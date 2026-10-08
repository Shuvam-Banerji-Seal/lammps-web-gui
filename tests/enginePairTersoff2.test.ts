import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { PairTersoffMod, PairTersoffModC, PairTersoffZBL } from '../src/engine/force/pair/tersoff_variants';
import { StyleError, type Pair, type StyleContext } from '../src/engine/force/types';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Unit checks for pair_style tersoff/mod, tersoff/mod/c and tersoff/zbl, written against the
 * documented formulas (docs.lammps.org/pair_tersoff_mod.html, docs.lammps.org/pair_tersoff_zbl.html).
 * Each expectation is recomputed here from the formulas and the parameters written in the
 * test, independently of the engine. Native LAMMPS agreement (thermo, pressure and per-atom
 * forces) is covered by tests/oracle/w14tersoff_*.in.
 *
 * Measured with native LAMMPS (black box), on the single-element files of this test:
 *   - tersoff/zbl (two-element file of tests/oracle/w14tersoff_zbl): the dimer energy with
 *     "shift 0.05" at r = 2.6 is -0.954087657 and equals the unshifted energy at r = 2.65
 *     (the shift replaces r by r + delta); pe is zero beyond R + D.
 *   - tersoff/zbl Coulomb prefactor: see ZBL_QQR2E in tersoff_variants.ts.
 * Two-body entries of a pair that differ (A B B vs B A A) give energies native LAMMPS computes
 * differently from the documented rule; the engine logs a WARNING for them (tested below).
 */

const ctx = (log: string[] = []): StyleContext => ({ s: null, readFile: () => '', log: (t) => log.push(t) });

/** Reads a potential text through pair_coeff of the given style (one atom type per element name). */
const readText = (p: Pair, text: string, elems: string[]): void => {
  p.allocate(elems.length);
  p.coeff(['*', '*', 'pot.file', ...elems], { s: null, log: () => {}, readFile: () => text });
};

const runSession = async (script: string, files: Record<string, string>): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
};

const script = (atoms: [number, number, number][], style: string, coeff: string, extra = ''): string => `
units metal
atom_style atomic
region box block -20 20 -20 20 -20 20
create_box 1 box
${atoms.map((p) => `create_atoms 1 single ${p[0]} ${p[1]} ${p[2]}`).join('\n')}
mass * 1.0
pair_style ${style}
pair_coeff * * pot.file ${coeff}
${extra}
thermo_style custom step pe
run 0
`;

const pe = async (atoms: [number, number, number][], style: string, coeff: string, file: string, extra = ''): Promise<number> => {
  const rows = await runSession(script(atoms, style, coeff, extra), { 'pot.file': file });
  return rows[0].pe;
};

const close = (a: number, b: number, rel = 1e-9, abs = 1e-12): boolean =>
  Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

/* ---------- tersoff/mod: single element ---------- */

const MOD = { alpha: 0.5, beta: 1.0, h: -0.4, eta: 1.2, lam2: 1.5, B: 150.0, R: 2.5, D: 0.25, lam1: 2.9, A: 1200.0, n: 0.7, c1: 0.2, c2: 300.0, c3: 50.0, c4: 0.5, c5: 4.0, c0: 0.1 };
const MOD_LINE = `A A A 1.0 ${MOD.alpha} ${MOD.h} ${MOD.eta} 1.0 ${MOD.lam2} ${MOD.B} ${MOD.R} ${MOD.D} ${MOD.lam1} ${MOD.A} ${MOD.n} ${MOD.c1} ${MOD.c2} ${MOD.c3} ${MOD.c4} ${MOD.c5}`;
const MODC_LINE = `${MOD_LINE} ${MOD.c0}`;

/** Murty cutoff of tersoff/mod: 1/2 - 9/16 sin(pi/2 x) - 1/16 sin(3 pi/2 x). */
const fcMod = (r: number, R = MOD.R, D = MOD.D): number => {
  if (r < R - D) return 1;
  if (r > R + D) return 0;
  const x = (r - R) / D;
  return 0.5 - (9 / 16) * Math.sin((Math.PI / 2) * x) - (1 / 16) * Math.sin((3 * Math.PI / 2) * x);
};

/** Documented tersoff/mod (and mod/c when c0 is true) energy, E = 1/2 sum_i sum_{j!=i} V_ij. */
const modEnergy = (pos: [number, number, number][], withC0: boolean): number => {
  const d = (a: number[], b: number[]): number => Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  let e = 0;
  for (let i = 0; i < pos.length; i++) {
    for (let j = 0; j < pos.length; j++) {
      if (i === j) continue;
      const rij = d(pos[i], pos[j]);
      let zeta = 0;
      for (let k = 0; k < pos.length; k++) {
        if (k === i || k === j) continue;
        const rik = d(pos[i], pos[k]);
        const vij = [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1], pos[j][2] - pos[i][2]];
        const vik = [pos[k][0] - pos[i][0], pos[k][1] - pos[i][1], pos[k][2] - pos[i][2]];
        const cth = (vij[0] * vik[0] + vij[1] * vik[1] + vij[2] * vik[2]) / (rij * rik);
        const u = MOD.h - cth;
        const g = MOD.c1 + (MOD.c2 * u * u / (MOD.c3 + u * u)) * (1 + MOD.c4 * Math.exp(-MOD.c5 * u * u));
        zeta += fcMod(rik) * g * Math.exp(MOD.alpha * Math.pow(rij - rik, MOD.beta));
      }
      const b = zeta > 0 ? Math.pow(1 + Math.pow(zeta, MOD.eta), -1 / (2 * MOD.n)) : 1;
      const fR = MOD.A * Math.exp(-MOD.lam1 * rij);
      const fA = -MOD.B * Math.exp(-MOD.lam2 * rij);
      e += 0.5 * fcMod(rij) * (fR + b * fA + (withC0 ? MOD.c0 : 0));
    }
  }
  return e;
};

const CLUSTER: [number, number, number][] = [[0, 0, 0], [2.3, 0, 0], [0.63, 2.0, 0], [1.0, 0.5, 1.8]];

describe('tersoff/mod', () => {
  it('two-body term on a dimer equals f_C(r)(f_R(r) + f_A(r)) with b = 1', async () => {
    const e = await pe([[0, 0, 0], [2.6, 0, 0]], 'tersoff/mod', 'A', MOD_LINE);
    expect(close(e, modEnergy([[0, 0, 0], [2.6, 0, 0]], false))).toBe(true);
  });

  it('bond-order term: a four-atom cluster matches the documented sums (zeta, b, g(theta), exponential)', async () => {
    const e = await pe(CLUSTER, 'tersoff/mod', 'A', MOD_LINE);
    expect(close(e, modEnergy(CLUSTER, false), 1e-8, 1e-10)).toBe(true);
  });

  it('pe is zero beyond R + D', async () => {
    const e = await pe([[0, 0, 0], [MOD.R + MOD.D + 0.01, 0, 0]], 'tersoff/mod', 'A', MOD_LINE);
    expect(e).toBe(0);
  });

  it('shift keyword replaces r by r + delta in every term', async () => {
    const e = await pe([[0, 0, 0], [2.6, 0, 0]], 'tersoff/mod shift 0.1', 'A', MOD_LINE);
    expect(close(e, modEnergy([[0, 0, 0], [2.7, 0, 0]], false))).toBe(true);
  });

  it('refuses unknown keywords and a UNITS header', () => {
    expect(() => new PairTersoffMod().settings(['foo', '1'], ctx())).toThrow(StyleError);
    expect(() => readText(new PairTersoffMod(), 'UNITS: metal\n' + MOD_LINE + '\n', ['A'])).toThrow(StyleError);
  });
});

describe('tersoff/mod/c', () => {
  it('adds c0 inside the bracket: V = f_C(r)(f_R + b f_A + c0)', async () => {
    const e = await pe(CLUSTER, 'tersoff/mod/c', 'A', MODC_LINE);
    expect(close(e, modEnergy(CLUSTER, true), 1e-8, 1e-10)).toBe(true);
  });

  it('a file without the c0 column is refused', () => {
    expect(() => readText(new PairTersoffModC(), MOD_LINE + '\n', ['A'])).toThrow(StyleError);
  });
});

/* ---------- tersoff/zbl: single element ---------- */

const ZBL = { m: 1, gamma: 1.0, lambda3: 0.0, c: 50390.0, d: 12.0, cos0: -0.5, n: 0.78, beta: 1e-6, lam2: 1.7, B: 470.0, R: 2.5, D: 0.2, lam1: 2.5, A: 1800.0, Z: 14, rc: 1.9, af: 4.0 };
const ZBL_LINE = `A A A ${ZBL.m} ${ZBL.gamma} ${ZBL.lambda3} ${ZBL.c} ${ZBL.d} ${ZBL.cos0} ${ZBL.n} ${ZBL.beta} ${ZBL.lam2} ${ZBL.B} ${ZBL.R} ${ZBL.D} ${ZBL.lam1} ${ZBL.A} ${ZBL.Z} ${ZBL.Z} ${ZBL.rc} ${ZBL.af}`;
/** Coulomb prefactor measured with native LAMMPS for metal units (tersoff_variants.ts). */
const QQ_METAL = 14.3996438057;

/** Documented tersoff/zbl dimer energy (b = 1): (1-f_F) V_ZBL + f_F f_C (f_R + f_A). */
const zblDimer = (r: number): number => {
  const a = (0.8854 * 0.529) / (2 * Math.pow(ZBL.Z, 0.23));
  const x = r / a;
  const phi = 0.1818 * Math.exp(-3.2 * x) + 0.5099 * Math.exp(-0.9423 * x) + 0.2802 * Math.exp(-0.4029 * x) + 0.02817 * Math.exp(-0.2016 * x);
  const fF = 1 / (1 + Math.exp(-ZBL.af * (r - ZBL.rc)));
  const fc = r < ZBL.R - ZBL.D ? 1 : r > ZBL.R + ZBL.D ? 0 : 0.5 - 0.5 * Math.sin((Math.PI / 2) * (r - ZBL.R) / ZBL.D);
  const vt = fc * (ZBL.A * Math.exp(-ZBL.lam1 * r) - ZBL.B * Math.exp(-ZBL.lam2 * r));
  return (1 - fF) * QQ_METAL * ZBL.Z * ZBL.Z * phi / r + fF * vt;
};

describe('tersoff/zbl', () => {
  it('dimer at short range (ZBL dominates) and in the switching region match the documented form', async () => {
    for (const r of [0.9, 1.5, 2.0, 2.4]) {
      const e = await pe([[0, 0, 0], [r, 0, 0]], 'tersoff/zbl', 'A', ZBL_LINE);
      expect(close(e, zblDimer(r), 1e-9, 1e-10)).toBe(true);
    }
  });

  it('shift keyword: pe at r = 2.6 with shift 0.05 equals the unshifted pe at 2.65', async () => {
    const e = await pe([[0, 0, 0], [2.6, 0, 0]], 'tersoff/zbl shift 0.05', 'A', ZBL_LINE);
    expect(close(e, zblDimer(2.65), 1e-9, 1e-10)).toBe(true);
  });

  it('pe is zero beyond R + D', async () => {
    const e = await pe([[0, 0, 0], [ZBL.R + ZBL.D + 0.05, 0, 0]], 'tersoff/zbl', 'A', ZBL_LINE);
    expect(e).toBe(0);
  });

  it('refuses m other than 1 or 3 and a UNITS header', () => {
    expect(() => readText(new PairTersoffZBL(), ZBL_LINE.replace('A A A 1 ', 'A A A 2 ') + '\n', ['A'])).toThrow(StyleError);
    expect(() => readText(new PairTersoffZBL(), 'UNITS: metal\n' + ZBL_LINE + '\n', ['A'])).toThrow(StyleError);
  });

  it('refuses unit styles other than metal and real', () => {
    const p = new PairTersoffZBL();
    readText(p, ZBL_LINE + '\n', ['A']);
    const lj = { s: { units: { style: 'lj' } }, log: () => {}, readFile: () => '' } as unknown as StyleContext;
    expect(() => p.init(lj)).toThrow(StyleError);
  });
});

/* ---------- two-body entries that differ between A B B and B A A ---------- */

describe('tersoff variants: asymmetric two-body entries', () => {
  it('logs a warning (native LAMMPS gives other energies for such files, measured)', async () => {
    const lines = [
      MOD_LINE.replace('A A A', 'A A A'),
      MOD_LINE.replace('A A A', 'A B B').replace(' 1200 ', ' 900 '),
      MOD_LINE.replace('A A A', 'B A A'),
      MOD_LINE.replace('A A A', 'A A B'),
      MOD_LINE.replace('A A A', 'A B A'),
      MOD_LINE.replace('A A A', 'B B A'),
      MOD_LINE.replace('A A A', 'B A B'),
      MOD_LINE.replace('A A A', 'B B B'),
    ];
    const log: string[] = [];
    const s = new PairTersoffMod();
    s.allocate(2);
    const file = '# two-element file\n' + lines.join('\n') + '\n';
    s.coeff(['*', '*', 'two.mod', 'A', 'B'], { s: null, log: () => {}, readFile: () => file });
    s.init(ctx(log));
    expect(log.some((t) => t.startsWith('WARNING: tersoff/mod potential file two.mod'))).toBe(true);
  });
});
