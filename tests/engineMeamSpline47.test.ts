import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import { CubicSpline, PairMeamSpline, PairMeamSWSpline, pairIndex, parseMeamSplineFile } from '../src/engine/force/pair/meam_spline';
import { StyleError } from '../src/engine/force/types';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Unit checks for pair_style meam/spline and meam/sw/spline (src/engine/force/pair/meam_spline.ts):
 * file parsing, refusals, the spline itself, and analytic forces against finite differences of the
 * engine's own energy. Native LAMMPS agreement (thermo, pressure, minimisation) is covered by the
 * tests/oracle/w47ms_*.in cases.
 *
 * Measured with native LAMMPS (black box), used as expectations below:
 *   - Si_1.meam.spline, dimer and four-atom clusters: the energies of the engine agree with native to
 *     better than 1e-14 absolute; a single isolated atom has energy 0 (the -U(0) shift per atom).
 *   - Ti.meam.sw.spline: the first knot of the phi block has a second derivative of 0 in the file, and
 *     native energies follow the clamped spline (recomputed from the end slopes), not the file's column.
 *   - pair_coeff with two atom types on one element of Si_1.meam.spline: an error that says one atom type per element is needed.
 *   - units real: "requires metal units".
 */

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const readPot = (name: string): string => readFileSync(join(POT, name), 'utf8');

const runScript = async (
  script: string,
  files: Record<string, string>,
): Promise<{ rows: ThermoRow[]; errors: string[]; session: Session }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  try {
    await session.execute(script);
  } catch (e) {
    // a refused command ends the script; the message is reported like the error events
    events.push({ kind: 'error', message: (e as Error).message } as EngineEvent);
  }
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  const errors = events.filter((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error').map((e) => e.message ?? '');
  return { rows, errors, session };
};

/** Atoms created one by one (create_atoms single) in a non-periodic box large enough for isolated clusters. */
const clusterScript = (style: string, elems: string[], types: number[], pos: number[][], coeff = 'pot.file'): string => `
units metal
atom_style atomic
boundary f f f
region box block -100 100 -100 100 -100 100
create_box ${elems.length} box
${pos.map((p, i) => `create_atoms ${types[i]} single ${p[0]} ${p[1]} ${p[2]} units box`).join('\n')}
mass * 28.0
pair_style ${style}
pair_coeff * * ${coeff} ${elems.join(' ')}
thermo_style custom step pe
run 0
`;

const energyOf = async (style: string, file: string, elems: string[], types: number[], pos: number[][]): Promise<number> => {
  const { rows, errors } = await runScript(clusterScript(style, elems, types, pos), { 'pot.file': readPot(file) });
  expect(errors).toEqual([]);
  return rows[0].pe;
};

describe('spline functions', () => {
  it('reproduces the knots, the end slopes and the linear continuation outside the knots', () => {
    const knots = [[0, 1, 0], [1, 2, 0], [2, 0, 0], [3, -1, 0]];
    const s = new CubicSpline('test', knots, 0.5, -0.25);
    for (const k of knots) expect(s.value(k[0])).toBeCloseTo(k[1], 13);
    // slope agrees with a central difference inside the knots
    const h = 1e-6;
    for (const v of [0.3, 1.7, 2.5]) {
      const fd = (s.value(v + h) - s.value(v - h)) / (2 * h);
      expect(Math.abs(fd - s.slope(v))).toBeLessThan(1e-7);
    }
    // the end slopes are the derivatives at the ends, and the spline is linear beyond them
    expect(s.slope(0)).toBeCloseTo(0.5, 9);
    expect(s.slope(3)).toBeCloseTo(-0.25, 9);
    expect(s.value(-2)).toBeCloseTo(1 + 0.5 * -2, 12);
    expect(s.value(5)).toBeCloseTo(-1 + -0.25 * 2, 12);
    expect(s.slope(-2)).toBe(0.5);
  });

  it('indexes the two-element pairs as (0,0),(0,1),(1,1) and is symmetric in its arguments', () => {
    const idx = [[0, 0, 0], [0, 1, 1], [1, 1, 2]];
    for (const [a, b, k] of idx) {
      expect(pairIndex(a, b, 2)).toBe(k);
      expect(pairIndex(b, a, 2)).toBe(k);
    }
  });
});

describe('potential file parsing', () => {
  it('reads the old single-element meam/spline file (phi, rho, U, f, g) with cutoffs at the last knots', () => {
    const f = parseMeamSplineFile(readPot('Si_1.meam.spline'), 'Si_1.meam.spline', false);
    expect(f.multi).toBe(false);
    expect(f.phi[0].xmax).toBeCloseTo(4.5, 12);
    expect(f.rho[0].xmax).toBeCloseTo(3.5, 12);
    expect(f.f[0].xmax).toBeCloseTo(3.5, 12);
    expect(f.U[0].n).toBe(8);
    expect(f.g[0].xmin).toBeCloseTo(-1, 9);
    // the clamped second derivative at the first knot of phi matches the file's third column (165.33216)
    expect(f.phi[0].y2[0]).toBeCloseTo(165.33216025579176, 6);
  });

  it('reads the new multicomponent file: N(N+1)/2 phi and g, N rho, U and f', () => {
    const f = parseMeamSplineFile(readPot('TiO.meam.spline'), 'TiO.meam.spline', false);
    expect(f.multi).toBe(true);
    expect(f.names).toEqual(['Ti', 'O']);
    expect(f.phi).toHaveLength(3);
    expect(f.g).toHaveLength(3);
    expect(f.rho).toHaveLength(2);
    expect(f.U).toHaveLength(2);
    expect(f.f).toHaveLength(2);
    expect(f.phi[0].xmax).toBeCloseTo(5.5, 12);
  });

  it('reads the meam/sw/spline file with F and G after phi', () => {
    const f = parseMeamSplineFile(readPot('Si.b.meam.sw.spline'), 'Si.b.meam.sw.spline', true);
    expect(f.F).not.toBeNull();
    expect(f.G).not.toBeNull();
    expect(f.F!.xmax).toBeCloseTo(3.7711800000000002, 9);
    expect(f.G!.n).toBe(9);
  });

  it('refuses a file with more spline blocks than the style reads, and a wrong block count', () => {
    expect(() => parseMeamSplineFile(readPot('Si.b.meam.sw.spline'), 'sw.file', false)).toThrow(StyleError);
    expect(() => parseMeamSplineFile(readPot('Si_1.meam.spline'), 'si.file', true)).toThrow(StyleError);
    expect(() => parseMeamSplineFile(readPot('TiO.meam.spline'), 'tio.file', true)).toThrow(/single-element/);
    const truncated = readPot('Si_1.meam.spline').split('\n').slice(0, 20).join('\n');
    expect(() => parseMeamSplineFile(truncated, 'short.file', false)).toThrow(StyleError);
  });

  it('refuses knots that do not increase in x and malformed flags', () => {
    const text = readPot('Si_1.meam.spline').split('\n');
    const bad = [...text];
    bad[4] = '3.0 0.0 0.0';
    expect(() => parseMeamSplineFile(bad.join('\n'), 'bad.file', false)).toThrow(/increase/);
    const flags = [...text];
    flags[3] = '1 0 2 0';
    expect(() => parseMeamSplineFile(flags.join('\n'), 'flags.file', false)).toThrow(/flags/);
  });
});

describe('pair_style meam/spline and meam/sw/spline: refusals', () => {
  const coeff = async (style: string, file: string, script: string) => {
    const { errors } = await runScript(script.replace('STYLE', style).replace('FILE', 'pot.file'), { 'pot.file': readPot(file) });
    return errors;
  };
  const base = (types: number, elems: string, units = 'metal') => `
units ${units}
atom_style atomic
boundary f f f
region box block -20 20 -20 20 -20 20
create_box ${types} box
create_atoms 1 single 0 0 0 units box
create_atoms ${types} single 2.3 0 0 units box
mass * 28.0
pair_style STYLE
pair_coeff * * FILE ${elems}
run 0
`;

  it('refuses units other than metal', async () => {
    const e = await coeff('meam/spline', 'Si_1.meam.spline', base(1, 'Si', 'real'));
    expect(e.join(' ')).toMatch(/requires metal units/);
  });

  it('refuses two atom types on the one element of a single-element file', async () => {
    const e = await coeff('meam/spline', 'Si_1.meam.spline', base(2, 'Si Si'));
    expect(e.join(' ')).toMatch(/one atom type per element/);
  });

  it('refuses NULL mappings', async () => {
    const e = await coeff('meam/spline', 'TiO.meam.spline', base(2, 'O NULL'));
    expect(e.join(' ')).toMatch(/NULL/);
  });

  it('refuses an element name that the multicomponent file does not contain', async () => {
    const e = await coeff('meam/spline', 'TiO.meam.spline', base(2, 'Ti X'));
    expect(e.join(' ')).toMatch(/not in potential file/);
  });

  it('refuses a meam/sw/spline style with two different element names', async () => {
    const e = await coeff('meam/sw/spline', 'Si.b.meam.sw.spline', base(2, 'Si Ti'));
    expect(e.join(' ')).toMatch(/single element/);
  });

  it('refuses style arguments and a pair_coeff that is not * *', async () => {
    const extra = await runScript(
      'units metal\natom_style atomic\nregion box block -5 5 -5 5 -5 5\ncreate_box 1 box\npair_style meam/spline 1.0\n', {},
    );
    expect(extra.errors.join(' ')).toMatch(/takes no arguments/);
    const notStar = await coeff('meam/spline', 'Si_1.meam.spline', base(1, 'Si').replace('pair_coeff * *', 'pair_coeff 1 1'));
    expect(notStar.join(' ')).toMatch(/\* \*/);
  });

  it('constructs both styles with the documented names', () => {
    expect(new PairMeamSpline().name).toBe('meam/spline');
    expect(new PairMeamSWSpline().name).toBe('meam/sw/spline');
  });
});

describe('energies and forces', () => {
  it('a single isolated atom has zero energy (the embedding term is shifted by U(0))', async () => {
    const e = await energyOf('meam/spline', 'Si_1.meam.spline', ['Si'], [1], [[0, 0, 0]]);
    expect(Math.abs(e)).toBeLessThan(1e-12);
  });

  it('dimer and cluster energies agree with the values measured with native LAMMPS (black box)', async () => {
    // Measured with native LAMMPS (black box): Si_1.meam.spline, dimer at 4.2 along x, pe -0.00808511755718257
    const dimer = await energyOf('meam/spline', 'Si_1.meam.spline', ['Si'], [1, 1], [[0, 0, 0], [4.2, 0, 0]]);
    expect(dimer).toBeCloseTo(-0.00808511755718257, 14);
    // Measured with native LAMMPS (black box): TiO.meam.spline, type 1 is the first element of the file (Ti), type 2 is O;
    // triangle Ti-O-Ti (atoms at (0,0,0), (1.9,0,0), (0.9,1.7,0.4)): pe 4.35776363872048
    const tio = await energyOf('meam/spline', 'TiO.meam.spline', ['Ti', 'O'], [1, 2, 1], [[0, 0, 0], [1.9, 0, 0], [0.9, 1.7, 0.4]]);
    expect(tio).toBeCloseTo(4.35776363872048, 10);
    // Measured with native LAMMPS (black box): Ti.meam.sw.spline, three atoms at (0,0,0), (2.4,0.1,0), (0.2,2.6,0.3): pe -3.28834271400376
    const sw = await energyOf('meam/sw/spline', 'Ti.meam.sw.spline', ['Ti'], [1, 1, 1], [[0, 0, 0], [2.4, 0.1, 0], [0.2, 2.6, 0.3]]);
    expect(sw).toBeCloseTo(-3.28834271400376, 10);
  });

  it('forces equal minus the finite-difference gradient of the energy (meam/spline, multicomponent, meam/sw/spline)', async () => {
    const cases: { style: string; file: string; elems: string[]; types: number[]; pos: number[][] }[] = [
      { style: 'meam/spline', file: 'Si_1.meam.spline', elems: ['Si'], types: [1, 1, 1, 1, 1], pos: [[0, 0, 0], [2.3, 0.4, 0.1], [0.3, 2.4, 0.5], [1.1, 1.0, 2.6], [2.9, 2.2, 2.3]] },
      { style: 'meam/spline', file: 'TiO.meam.spline', elems: ['Ti', 'O'], types: [1, 2, 1, 2, 1], pos: [[0, 0, 0], [2.2, 0.3, 0.1], [0.3, 2.5, 0.4], [1.2, 1.1, 2.6], [3.0, 2.2, 2.3]] },
      { style: 'meam/sw/spline', file: 'Si.b.meam.sw.spline', elems: ['Si'], types: [1, 1, 1, 1], pos: [[0, 0, 0], [2.2, 0.3, 0.1], [0.3, 2.3, 0.4], [1.0, 1.0, 2.2]] },
    ];
    for (const c of cases) {
      const run = async (pos: number[][]) => {
        const { rows, errors, session } = await runScript(clusterScript(c.style, c.elems, c.types, pos), { 'pot.file': readPot(c.file) });
        expect(errors).toEqual([]);
        const f = Array.from(session.sys.state.f as Float64Array).slice(0, 3 * pos.length);
        return { pe: rows[0].pe, f };
      };
      const base = await run(c.pos);
      const h = 1e-5;
      let maxErr = 0;
      for (let a = 0; a < c.pos.length; a++) {
        for (let k = 0; k < 3; k++) {
          const p1 = c.pos.map((q) => q.slice()), p2 = c.pos.map((q) => q.slice());
          p1[a][k] += h;
          p2[a][k] -= h;
          const fd = -((await run(p1)).pe - (await run(p2)).pe) / (2 * h);
          maxErr = Math.max(maxErr, Math.abs(fd - base.f[3 * a + k]));
        }
      }
      expect(maxErr, `${c.style} ${c.file}`).toBeLessThan(1e-6);
    }
  });
});
