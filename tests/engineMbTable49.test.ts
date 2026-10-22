import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { readAngleSection, AngleTable } from '../src/engine/force/angleTable49';
import { PairThreebodyTable } from '../src/engine/force/pair/threebody_table';
import { PairSWAngleTable } from '../src/engine/force/pair/sw_angle_table';
import { PAIR_STYLES, ensureStyles } from '../src/engine/styles';

/*
 * threebody/table and sw/angle/table: file parsing, refusals, and a small engine run per style (the oracle cases
 * tests/oracle/w49mt_*.in carry the native comparisons). The table files below are generated, not LAMMPS files.
 */

/** Runs a script on a session with the given virtual files; returns the error messages and the thermo pe rows. */
const run = async (script: string, files: Record<string, string>) => {
  await ensureStyles(script.split(/\s+/));
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  let thrown = '';
  try {
    await session.execute(script);
  } catch (e) {
    thrown = String(e);
  }
  const errors = events.filter((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error').map((e) => e.message);
  const pe = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row.pe as number);
  return { errors: thrown ? [...errors, thrown] : errors, pe };
};

/** A symmetric N = 2 three-body table (M = 2*2*3 = 12) with energies 1..12 and force constants 0.1*index. */
const symTable = (n = 2, withEnergy = true): string => {
  const rmin = 2.0, rmax = 3.0;
  const dr = (rmax - rmin) / (n - 1);
  const rows: string[] = [];
  let idx = 0;
  for (let a = 0; a < n; a++) {
    for (let b = a; b < n; b++) {
      for (let m = 0; m < 2 * n; m++) {
        idx++;
        const f = 0.1 * idx;
        const e = withEnergy ? idx : 0;
        rows.push(`${idx} ${rmin + a * dr} ${rmin + b * dr} ${((m + 0.5) * 180) / (2 * n)} ${f} ${-f / 2} ${-f} ${f / 3} ${f / 2} ${-f / 3} ${e}`);
      }
    }
  }
  return `# test table\n\nENTRY1\nN ${n} rmin ${rmin} rmax ${rmax}\n\n${rows.join('\n')}\n`;
};

const potential3B = (cut = 3.7, style = 'linear', n = 2): string =>
  `type\ntype\ntype\n${cut}\ntest.table\nENTRY1\n${style}\n${n}\n`;

/**
 * Three atoms: the center at the origin, j at 2.4 on the x axis and k at 2.4 with a 150 degree angle; j and k are
 * 4.6 apart, beyond the cut, so the center is the only triplet.
 */
const pairThreebody = (potential: string, table: string) =>
  run(
    `units real
atom_style atomic
boundary f f f
lattice none 1.0
region b block -30 30 -30 30 -30 30
create_box 1 b
create_atoms 1 single 0 0 0 units box
create_atoms 1 single 2.4 0 0 units box
create_atoms 1 single -2.0784610 1.2 0 units box
mass 1 18
pair_style threebody/table
pair_coeff * * test.3b type
thermo_style custom step pe
thermo_modify format float %.15g
run 0`,
    { 'test.3b': potential, 'test.table': table },
  );

describe('threebody/table file handling and refusals', () => {
  it('returns the table energy of the one triplet: r = 2.4 -> grid 2.0, theta 150 -> the 157.5 bin (row 4)', async () => {
    const r = await pairThreebody(potential3B(), symTable());
    expect(r.errors).toEqual([]);
    expect(r.pe).toEqual([4]);
  });

  it('refuses a table style other than linear, naming it', async () => {
    const r = await pairThreebody(potential3B(3.7, 'spline'), symTable());
    expect(r.errors.join(' ')).toMatch(/table style 'spline' is not supported/);
  });

  it('refuses a pair_style with arguments', async () => {
    const r = await run('pair_style threebody/table 1.0\n', {});
    expect(r.errors.join(' ')).toMatch(/Illegal pair_style command/);
  });

  it('names a missing element of the potential file', async () => {
    const r = await run(
      `units real
atom_style atomic
region b block -30 30 -30 30 -30 30
create_box 2 b
mass * 18
pair_style threebody/table
pair_coeff * * test.3b type1 type2`,
      { 'test.3b': potential3B(), 'test.table': symTable() },
    );
    expect(r.errors.join(' ')).toMatch(/element 'type1' is not in threebody\/table potential file test.3b/);
  });

  it('refuses a table with too few rows for its layout', async () => {
    const t = symTable(2).split('\n').slice(0, -3).join('\n') + '\n';
    expect(() => {
      // the reader itself: build the layout directly through the style's coeff path
      const style = new PairThreebodyTable();
      style.allocate(1);
      const ctxFiles: Record<string, string> = { 'test.3b': potential3B(), 'test.table': t };
      style.coeff(['*', '*', 'test.3b', 'type'], { readFile: (n: string) => ctxFiles[n], log: () => {} } as never);
    }).toThrow(/has \d+ of 12 data lines/);
  });

  it('refuses an unknown table parameter', async () => {
    const t = symTable().replace('N 2 rmin 2 rmax 3', 'N 2 rmin 2 rmax 3 RSQ 1');
    const r = await pairThreebody(potential3B(), t);
    expect(r.errors.join(' ')).toMatch(/unknown parameter 'RSQ'/);
  });

  it('is registered in the style table once loaded', () => {
    expect(Object.keys(PAIR_STYLES)).toContain('threebody/table');
  });
});

describe('sw/angle/table refusals', () => {
  it('refuses settings', () => {
    const style = new PairSWAngleTable();
    expect(() => style.settings(['threebody', 'off'], {} as never)).toThrow(/takes no settings/);
  });

  it('refuses the spline style when N of the entry differs from N of the table', () => {
    const sec = readAngleSection('ANG\nN 3\n\n1 0 0 0\n2 90 1 0.5\n3 180 0 0\n', 'a.txt', 'ANG');
    expect(() => AngleTable.build(sec, 5, 'spline', 'test')).toThrow(/spline style of the angle table is only supported/);
    expect(AngleTable.build(sec, 3, 'spline', 'test').energy(90)).toBeCloseTo(1, 12);
  });
});

describe('angle table section reader', () => {
  it('reads FP and EQ and the values of the named section', () => {
    const sec = readAngleSection('# c\n\nOTHER\nN 2\n\n1 0 0 0\n2 180 0 0\n\nANG\nN 3 FP 0 0 EQ 90\n\n1 0 0 -1\n2 90 2 0\n3 180 0 1\n', 'a.txt', 'ANG');
    expect(sec.n).toBe(3);
    expect(Array.from(sec.x)).toEqual([0, 90, 180]);
    expect(Array.from(sec.e)).toEqual([0, 2, 0]);
  });

  it('refuses a section that does not start at 0 or end at 180 degrees', () => {
    expect(() => readAngleSection('ANG\nN 2\n\n1 10 0 0\n2 180 0 0\n', 'a.txt', 'ANG')).toThrow(/must begin with angle 0.0 and end with 180.0/);
  });

  it('refuses an unknown parameter and a missing keyword', () => {
    expect(() => readAngleSection('ANG\nN 2 XX 1\n\n1 0 0 0\n2 180 0 0\n', 'a.txt', 'ANG')).toThrow(/unknown parameter 'XX'/);
    expect(() => readAngleSection('ANG\nN 2\n\n1 0 0 0\n2 180 0 0\n', 'a.txt', 'NOPE')).toThrow(/no section with keyword 'NOPE'/);
  });

  it('interpolates linearly and on a resampled grid', () => {
    const sec = readAngleSection('ANG\nN 3\n\n1 0 0 0\n2 90 1 0\n3 180 0 0\n', 'a.txt', 'ANG');
    const lin = AngleTable.build(sec, 3, 'linear', 'test');
    expect(lin.energy(45)).toBeCloseTo(0.5, 12);
    expect(lin.energy(180)).toBeCloseTo(0, 12);
    // a finer grid passes through the file nodes, so the energy at 90 degrees is the node value
    const fine = AngleTable.build(sec, 5, 'linear', 'test');
    expect(fine.energy(90)).toBeCloseTo(1, 12);
  });
});
