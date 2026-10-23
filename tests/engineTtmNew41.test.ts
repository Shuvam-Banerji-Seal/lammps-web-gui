import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { parseThermalPositional, parseThermalProps } from '../src/engine/fix/ttm_thermal';
import { parseCascadeTable, interpTable, integralTable } from '../src/engine/fix/ttm_cascade';
import { parseTtmKeywords, ttmPositional } from '../src/engine/fix/ttm';
import { FIXES } from '../src/engine/registry/fix_ttm';

/*
 * fix ttm/thermal and fix ttm/cascade (wave 41): parsing of the positional arguments, the properties file
 * and the tables, the StyleErrors of the refused and invalid inputs. The coupled runs are oracle cases
 * (tests/oracle/w41ttm_*.in, checked by engineOracle.test.ts); this file does not repeat them.
 */

const BOX = `
units metal
atom_style atomic
lattice sc 3.0
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 10.0
displace_atoms all move 0.5 0.5 0.5 units box
velocity all set 1.0 0.5 -0.3 units box
pair_style lj/cut 3.0
pair_coeff 1 1 0.0 2.0
fix 1 all nve
timestep 0.001
`;

const cellsOf = (nx: number, ny: number, nz: number) => {
  const out: [number, number, number][] = [];
  for (let z = 1; z <= nz; z++) for (let y = 1; y <= ny; y++) for (let x = 1; x <= nx; x++) out.push([x, y, z]);
  return out;
};
const CELLS = cellsOf(2, 2, 2);
const props = (lines: string[] = CELLS.map(([x, y, z]) => `${x} ${y} ${z} 1.0 1.0 20.0 1.0`), first = '# UNITS: metal') =>
  [first, '# ix iy iz C_vol kappa_e gamma_p eta', ...lines].join('\n') + '\n';
const INIT = `# UNITS: metal\n${CELLS.map(([x, y, z]) => `${x} ${y} ${z} 1000`).join('\n')}\n`;

const runWith = async (text: string, files: Record<string, string>) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [name, t] of Object.entries(files)) session.addFile(name, t);
  await session.execute(text);
  return events;
};

const expectError = async (fix: string, files: Record<string, string>, msg: RegExp | string) => {
  await expect(runWith(`${BOX}\n${fix}\nrun 1`, files)).rejects.toThrow(msg);
};

describe('ttm/thermal positional arguments', () => {
  it('reads seed, properties file and the grid', () => {
    const p = parseThermalPositional('f', ['11111', 'props.in', '10', '10', '10']);
    expect(p).toMatchObject({ seed: 11111, propsFile: 'props.in', nx: 10, ny: 10, nz: 10, consumed: 5, scalar: null });
  });

  it('rejects a short argument list, a bad seed and a bad grid size', () => {
    expect(() => parseThermalPositional('f', ['1', 'p', '2', '2'])).toThrow(/Illegal fix ttm\/thermal command: missing argument\(s\)/);
    expect(() => parseThermalPositional('f', ['1.5', 'p', '2', '2', '2'])).toThrow(/seed must be a positive integer/);
    expect(() => parseThermalPositional('f', ['1', 'p', '0', '2', '2'])).toThrow(/Nx must be a positive integer/);
    expect(() => parseThermalPositional('f', ['1', 'p', '2', '2', '2.5'])).toThrow(/Nz must be a positive integer/);
  });
});

describe('ttm/thermal properties file', () => {
  const parse = (text: string) => parseThermalProps(text, 'f', 2, 2, 2, 'metal', 'p.in');

  it('reads the seven columns per grid point, comments and blank lines skipped, extra words ignored', () => {
    const r = parse(props(CELLS.map(([x, y, z], k) => `${x} ${y} ${z} ${1 + k} 2.0 3.0 0.5 extra words`)));
    expect(Array.from(r.cvol)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(Array.from(r.kappa)).toEqual(new Array(8).fill(2));
    expect(Array.from(r.gammaP)).toEqual(new Array(8).fill(3));
    expect(Array.from(r.eta)).toEqual(new Array(8).fill(0.5));
  });

  it('accepts zero conductivity, zero friction and zero absorption (measured: no error)', () => {
    const r = parse(props(CELLS.map(([x, y, z]) => `${x} ${y} ${z} 1.0 0.0 0.0 0.0`)));
    expect(Array.from(r.kappa)).toEqual(new Array(8).fill(0));
    expect(Array.from(r.gammaP)).toEqual(new Array(8).fill(0));
  });

  it('reads the UNITS tag on the first line only, and it must match the run units (measured)', () => {
    expect(() => parse(props(undefined, '# UNITS: lj'))).toThrow(/was written for units lj but metal units are in use/);
    // a tag on another line is not read (measured with native LAMMPS: no error)
    const body = CELLS.map(([x, y, z]) => `${x} ${y} ${z} 1 1 1 1`).join('\n');
    expect(() => parse(`# comment\n# UNITS: lj\n${body}\n`)).not.toThrow();
  });

  it('stops for a line with fewer than seven numbers, a non-number, or an index out of range (measured)', () => {
    expect(() => parse(props(['1 1 1 1.0 1.0 20.0']))).toThrow(/needs ix iy iz C_vol kappa_e gamma_p eta/);
    expect(() => parse(props(['1 1 1 x 1.0 20.0 1.0']))).toThrow(/non-numeric/);
    expect(() => parse(props(['3 1 1 1.0 1.0 20.0 1.0']))).toThrow(/grid index out of range/);
    expect(() => parse(props(['1.5 1 1 1.0 1.0 20.0 1.0']))).toThrow(/grid index out of range/);
  });

  it('stops for a negative specific heat, conductivity, friction or absorption (measured: each is an error)', () => {
    const with1 = (cols: string) => props([cols, ...CELLS.slice(1).map(([x, y, z]) => `${x} ${y} ${z} 1 1 1 1`)]);
    expect(() => parse(with1('1 1 1 -1.0 1.0 20.0 1.0'))).toThrow(/electron specific heat/);
    expect(() => parse(with1('1 1 1 1.0 -1.0 20.0 1.0'))).toThrow(/electron conductivity/);
    expect(() => parse(with1('1 1 1 1.0 1.0 -20.0 1.0'))).toThrow(/electron coupling/);
    expect(() => parse(with1('1 1 1 1.0 1.0 20.0 -1.0'))).toThrow(/electron inductive response/);
  });

  it('refuses C_vol = 0 (native accepts it and freezes the grid; not reproduced here)', () => {
    expect(() => parse(props(['1 1 1 0.0 1.0 20.0 1.0', ...CELLS.slice(1).map(([x, y, z]) => `${x} ${y} ${z} 1 1 1 1`)])))
      .toThrow(/C_vol = 0 at 1 1 1 is not reproduced/);
  });

  it('refuses a grid point given twice and a grid point never given (measured: a missing point is an error)', () => {
    expect(() => parse(props([...CELLS.map(([x, y, z]) => `${x} ${y} ${z} 1 1 1 1`), '1 1 1 2 2 2 2']))).toThrow(/sets grid point 1 1 1 twice/);
    expect(() => parse(props(CELLS.slice(1).map(([x, y, z]) => `${x} ${y} ${z} 1 1 1 1`)))).toThrow(/did not set all 2x2x2 grid points/);
  });
});

describe('ttm/cascade tables', () => {
  it('reads a two-column table with comments and blank lines', () => {
    const t = parseCascadeTable('# T C\n\n500 1.0 # first\n2000 3.0\n', 'f', 'C_e', 'ce.tab');
    expect(t.t).toEqual([500, 2000]);
    expect(t.v).toEqual([1, 3]);
  });

  it('stops for fewer than two rows, a temperature not strictly increasing, or a value not positive (measured)', () => {
    expect(() => parseCascadeTable('500 1.0\n', 'f', 'C_e', 'ce.tab')).toThrow(/less than 2 rows/);
    expect(() => parseCascadeTable('2000 3.0\n500 1.0\n', 'f', 'C_e', 'ce.tab')).toThrow(/strictly increasing/);
    expect(() => parseCascadeTable('500 1.0\n500 2.0\n', 'f', 'C_e', 'ce.tab')).toThrow(/strictly increasing/);
    expect(() => parseCascadeTable('500 -1.0\n4000 2.0\n', 'f', 'C_e', 'ce.tab')).toThrow(/must be > 0/);
    expect(() => parseCascadeTable('500 1.0\n4000 x\n', 'f', 'C_e', 'ce.tab')).toThrow(/needs a temperature and a value/);
  });

  it('interpolates linearly and holds the end values outside the table', () => {
    const t = parseCascadeTable('500 1.0\n2000 3.0\n4000 2.0\n', 'f', 'C_e', 'ce.tab');
    expect(interpTable(t, 100)).toBe(1);
    expect(interpTable(t, 500)).toBe(1);
    expect(interpTable(t, 1250)).toBeCloseTo(2, 12);
    expect(interpTable(t, 3000)).toBeCloseTo(2.5, 12);
    expect(interpTable(t, 9000)).toBe(2);
  });

  it('integrates from 0 with the table flat below its first and above its last temperature', () => {
    const t = parseCascadeTable('500 1.0\n2000 3.0\n4000 2.0\n', 'f', 'C_e', 'ce.tab');
    // 0..500 at 1.0; 500..2000 trapezoid (1+3)/2*1500 = 3000; 2000..4000 trapezoid (3+2)/2*2000 = 5000; 4000..5000 at 2.0
    expect(integralTable(t, 500)).toBeCloseTo(500, 9);
    expect(integralTable(t, 2000)).toBeCloseTo(3500, 9);
    expect(integralTable(t, 4000)).toBeCloseTo(8500, 9);
    expect(integralTable(t, 5000)).toBeCloseTo(10500, 9);
    // the integral at 1250 is 500 + (1 + 2)/2 * 750 (C at 1250 is 2.0... linear from 1.0 to 3.0 over 500..2000)
    expect(integralTable(t, 1250)).toBeCloseTo(500 + 750 * (1 + 2) / 2, 9);
  });
});

describe('ttm family keywords', () => {
  it('ttm/grid and ttm/cascade refuse the outfile keyword; ttm and ttm/thermal accept it', () => {
    expect(() => parseTtmKeywords('f', 'ttm/cascade', ['outfile', '1', 'x'], false)).toThrow(/does not support the outfile keyword/);
    expect(parseTtmKeywords('f', 'ttm', ['outfile', '2', 'x'], true).outNevery).toBe(2);
  });

  it('the positional part of ttm/cascade is the ttm one (ten arguments)', () => {
    const p = ttmPositional('f', 'ttm/cascade', ['77', '1', '1', '0.5', '20', '0', '2', '2', '2', '2']);
    expect(p).toMatchObject({ seed: 77, nx: 2, ny: 2, nz: 2, consumed: 10 });
  });
});

describe('ttm/thermal and ttm/cascade in the engine', () => {
  it('are registered as fix styles', () => {
    expect(Object.keys(FIXES)).toEqual(expect.arrayContaining(['ttm', 'ttm/grid', 'ttm/thermal', 'ttm/cascade']));
  });

  it('ttm/thermal: C_vol = 0 in the file is refused', async () => {
    await expectError('fix 2 all ttm/thermal 11 p.in 2 2 2 infile i.in', {
      'p.in': props(['1 1 1 0.0 1.0 20.0 1.0', ...CELLS.slice(1).map(([x, y, z]) => `${x} ${y} ${z} 1 1 1 1`)]),
      'i.in': INIT,
    }, /C_vol = 0 at 1 1 1 is not reproduced/);
  });

  it('ttm/thermal: the source keyword needs a value', async () => {
    await expectError('fix 2 all ttm/thermal 11 p.in 2 2 2 set 1000 source', { 'p.in': props() }, 'Illegal fix ttm/thermal source command: missing argument(s)');
  });

  it('ttm/thermal: an unknown keyword is an error', async () => {
    await expectError('fix 2 all ttm/thermal 11 p.in 2 2 2 set 1000 bogus 3', { 'p.in': props() }, 'Unknown fix ttm/thermal keyword bogus');
  });

  it('ttm/thermal: no stopping keywords (the doc: the effects of electron stopping have been removed)', async () => {
    await expectError('fix 2 all ttm/thermal 11 p.in 2 2 2 set 1000 cutoff', { 'p.in': props() }, 'Unknown fix ttm/thermal keyword cutoff');
  });

  it('ttm/cascade: needs set or infile (the doc: the temperature must be a non-zero positive value)', async () => {
    await expectError('fix 2 all ttm/cascade 77 1 1 0.5 20 0 2 2 2 2', {}, /needs the set or infile keyword/);
  });

  it('ttm/cascade: a zero electron temperature in infile is refused', async () => {
    await expectError('fix 2 all ttm/cascade 77 1 1 0.5 20 0 2 2 2 2 infile i.in', {
      'i.in': INIT.replace('1 1 1 1000', '1 1 1 0'),
    }, /must be a non-zero positive value/);
  });

  it('ttm/cascade: no outfile keyword (native: "Unknown fix ttm/cascade keyword outfile")', async () => {
    await expectError('fix 2 all ttm/cascade 77 1 1 0.5 20 0 2 2 2 2 set 1000 outfile 1 q.out', {}, /does not support the outfile keyword/);
  });

  it('ttm/cascade: cutoff takes no value (measured: the number after it is an unknown keyword)', async () => {
    await expectError('fix 2 all ttm/cascade 77 1 1 0.5 20 0 2 2 2 2 set 1000 cutoff 3', {}, 'Unknown fix ttm/cascade keyword 3');
  });

  it('ttm/cascade: a negative offset is an error; gamma_p = 0 and gamma_s < 0 are errors (measured)', async () => {
    await expectError('fix 2 all ttm/cascade 77 1 1 0.5 20 0 2 2 2 2 set 1000 offset -1.0', {}, /time offset must be >= 0/);
    await expectError('fix 2 all ttm/cascade 77 1 1 0.5 0 0 2 2 2 2 set 1000', {}, /gamma_p must be > 0/);
    await expectError('fix 2 all ttm/cascade 77 1 1 0.5 20 -1 2 2 2 2 2 set 1000', {}, /gamma_s must be >= 0/);
  });

  it('ttm/cascade: a missing table file is an error, and so is a table that is not sorted', async () => {
    await expectError('fix 2 all ttm/cascade 77 1 1 0.5 20 0 2 2 2 2 set 1000 cetab nothere.tab', {}, /nothere\.tab|cannot open|not found|No such/i);
    await expectError('fix 2 all ttm/cascade 77 1 1 0.5 20 0 2 2 2 2 set 1000 cetab ce.tab', {
      'ce.tab': '2000 3.0\n500 1.0\n',
    }, /strictly increasing/);
  });
});
