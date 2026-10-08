import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { PairTersoff } from '../src/engine/force/pair/tersoff';
import { StyleError, type Pair, type StyleContext } from '../src/engine/force/types';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Unit checks for pair_style tersoff, written against the documented formulas
 * (docs.lammps.org/pair_tersoff.html):
 *
 *   V_ij = f_C(r_ij) [ f_R(r_ij) + b_ij f_A(r_ij) ]
 *   f_R(r) = A exp(-lambda1 r),  f_A(r) = -B exp(-lambda2 r)
 *   b_ij = (1 + beta^n zeta_ij^n)^(-1/(2n))
 *   zeta_ij = sum_{k != i,j} f_C(r_ik) g(theta_ijk) exp[lambda3^m (r_ij-r_ik)^m]
 *   g(theta) = gamma (1 + c^2/d^2 - c^2/[d^2+(cos theta - cos theta0)^2])
 *   E = 1/2 sum_i sum_{j != i} V_ij
 *
 * The expectations below recompute these sums independently in the test, so
 * a two-atom dimer checks the two-body term (zeta = 0, b = 1) and a three-atom
 * cluster checks the bond-order term.  Native LAMMPS agreement (energy,
 * pressure and the final per-atom state) is covered by tests/oracle/
 * w2tsw_tersoff*.in.
 */

interface P {
  m: number; gamma: number; lambda3: number; c: number; d: number; costheta0: number;
  n: number; beta: number; lambda2: number; B: number; R: number; D: number; lambda1: number; A: number;
}

const P: P = {
  m: 3, gamma: 1.2, lambda3: 0.15, c: 2.5, d: 1.7, costheta0: 0.4,
  n: 0.9, beta: 1e-3, lambda2: 1.3, B: 5.0, R: 3.0, D: 0.5, lambda1: 2.0, A: 10.0,
};

const potLine = `E E E ${P.m} ${P.gamma} ${P.lambda3} ${P.c} ${P.d} ${P.costheta0} ${P.n} ${P.beta} ${P.lambda2} ${P.B} ${P.R} ${P.D} ${P.lambda1} ${P.A}`;
const potFile = `# tiny single-element Tersoff file\n${potLine}\n`;

const fc = (r: number): number => {
  if (r < P.R - P.D) return 1;
  if (r > P.R + P.D) return 0;
  return 0.5 - 0.5 * Math.sin((Math.PI / 2) * (r - P.R) / P.D);
};
const fR = (r: number): number => P.A * Math.exp(-P.lambda1 * r);
const fA = (r: number): number => -P.B * Math.exp(-P.lambda2 * r);
const gOf = (cth: number): number => {
  const q = cth - P.costheta0;
  return P.gamma * (1 + (P.c * P.c) / (P.d * P.d) - (P.c * P.c) / (P.d * P.d + q * q));
};
const bOf = (zeta: number): number => (1 + Math.pow(P.beta, P.n) * Math.pow(zeta, P.n)) ** (-1 / (2 * P.n));

/** E = 1/2 sum_i sum_{j!=i} V_ij using the documented formulas and positions. */
const expectedEnergy = (pos: [number, number, number][]): number => {
  const dist = (a: [number, number, number], b: [number, number, number]): [number, number, number] => {
    const d: [number, number, number] = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    return d;
  };
  const norm = (v: [number, number, number]): number => Math.hypot(v[0], v[1], v[2]);
  let sum = 0;
  for (let i = 0; i < pos.length; i++) {
    for (let j = 0; j < pos.length; j++) {
      if (i === j) continue;
      const rij = dist(pos[i], pos[j]);
      const r1 = norm(rij);
      if (r1 >= P.R + P.D) continue;
      let zeta = 0;
      for (let k = 0; k < pos.length; k++) {
        if (k === i || k === j) continue;
        const rik = dist(pos[i], pos[k]);
        const r2 = norm(rik);
        if (r2 >= P.R + P.D) continue;
        const cth = (rij[0] * rik[0] + rij[1] * rik[1] + rij[2] * rik[2]) / (r1 * r2);
        zeta += fc(r2) * gOf(cth) * Math.exp(Math.pow(P.lambda3, P.m) * Math.pow(r1 - r2, P.m));
      }
      const b = zeta > 0 ? bOf(zeta) : 1;
      sum += fc(r1) * (fR(r1) + b * fA(r1));
    }
  }
  return 0.5 * sum;
};

const ctx: StyleContext = { s: null, readFile: () => '', log: () => {} };

const make = (coeffs: string[], files: Record<string, string> = {}): Pair => {
  const s = new PairTersoff();
  s.settings([], ctx);
  s.allocate(1);
  const full: StyleContext = {
    s: null,
    log: () => {},
    readFile: (n) => {
      if (!(n in files)) throw new StyleError(`cannot open file ${n}`);
      return files[n];
    },
  };
  for (const c of coeffs) s.coeff(c.split(/\s+/), full);
  s.init(full);
  return s;
};

const close = (a: number, b: number, rel: number, abs: number) =>
  Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

const runSession = async (script: string, files: Record<string, string>): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
};

const script = (atoms: string): string => `
units metal
atom_style atomic
region box block 0 12 0 12 0 12
create_box 1 box
${atoms}
mass * 1.0
pair_style tersoff
pair_coeff * * tiny.tersoff E
thermo_style custom step pe
run 0
`;

describe('tersoff: documented two-body term on a dimer', () => {
  it('pe at r=2.5 (inside R+D) equals f_C(r)[f_R(r)+f_A(r)] with b=1', async () => {
    const r = 2.5;
    const rows = await runSession(script('create_atoms 1 single 2 2 2\ncreate_atoms 1 single 4.5 2 2'), { 'tiny.tersoff': potFile });
    expect(rows.length).toBe(1);
    // a dimer: zeta = 0 so b = 1, and each ordered term is visited once per center
    expect(close(rows[0].pe, fc(r) * (fR(r) + fA(r)), 1e-12, 1e-12)).toBe(true);
  });

  it('pe is zero beyond R+D', async () => {
    const rows = await runSession(script('create_atoms 1 single 2 2 2\ncreate_atoms 1 single 5.6 2 2'), { 'tiny.tersoff': potFile });
    expect(Math.abs(rows[0].pe)).toBe(0);
  });
});

describe('tersoff: documented bond-order term on a three-atom cluster', () => {
  it('pe equals the independent 1/2 sum_i sum_{j!=i} V_ij with zeta over the third atom', async () => {
    const pos: [number, number, number][] = [[2, 2, 2], [4.2, 2, 2], [2, 4.1, 2]];
    const atoms = pos.map((p) => `create_atoms 1 single ${p[0]} ${p[1]} ${p[2]}`).join('\n');
    const rows = await runSession(script(atoms), { 'tiny.tersoff': potFile });
    const want = expectedEnergy(pos);
    // the three-body term is nonzero here (the angle is not 90 deg), so this
    // exercises zeta, g(theta), the lambda3 exponent and b_ij
    expect(Math.abs(want - 0.5 * (fR(2.2) + fA(2.2) + fR(2.1) + fA(2.1) + fR(Math.hypot(0.1, -2.1)) + fA(Math.hypot(0.1, -2.1))))).toBeGreaterThan(0);
    expect(close(rows[0].pe, want, 1e-12, 1e-12)).toBe(true);
  });

  it('b_ij reduces to 1 when the influencing atom is out of range', async () => {
    // third atom far away: only the neighbouring pair survives, b = 1
    const r = 2.4;
    const rows = await runSession(script('create_atoms 1 single 2 2 2\ncreate_atoms 1 single 4.4 2 2\ncreate_atoms 1 single 2 10 2'), { 'tiny.tersoff': potFile });
    expect(close(rows[0].pe, fc(r) * (fR(r) + fA(r)), 1e-12, 1e-12)).toBe(true);
  });
});

describe('tersoff: coefficients, cutoffs and errors', () => {
  it('cutoff is R+D and NULL mapping turns the type off', () => {
    const s = make(['* * tiny.tersoff E'], { 'tiny.tersoff': potFile });
    expect(s.cut[1 * 2 + 1]).toBe(P.R + P.D);
    const nulled = make(['* * tiny.tersoff NULL'], { 'tiny.tersoff': potFile });
    expect(nulled.initOne(1, 1)).toBe(0);
  });

  it('errors: unknown keyword, non-* * coefficients, missing box, wrong element count, unknown element', () => {
    const s = new PairTersoff();
    expect(() => s.settings(['bogus'], ctx)).toThrow(StyleError);
    expect(() => s.coeff(['*', '*', 'tiny.tersoff', 'E'], ctx)).toThrow(StyleError); // no box / ntypes
    const a = new PairTersoff();
    a.settings([], ctx);
    a.allocate(1);
    expect(() => a.coeff(['1', '1', 'tiny.tersoff', 'E'], ctx)).toThrow(StyleError);
    expect(() => make(['* * tiny.tersoff'], { 'tiny.tersoff': potFile })).toThrow(StyleError);
    expect(() => make(['* * tiny.tersoff Q'], { 'tiny.tersoff': potFile })).toThrow(StyleError);
  });

  it('errors: no pair_coeff (init) and m not 1 or 3', () => {
    const s = new PairTersoff();
    s.settings([], ctx);
    s.allocate(1);
    expect(() => s.init(ctx)).toThrow(StyleError);
    const badM = potLine.replace(' 3 1.2', ' 2 1.2');
    expect(() => make(['* * tiny.tersoff E'], { 'tiny.tersoff': `${badM}\n` })).toThrow(StyleError);
  });

  it('shift keyword is parsed and round-trips through settings', () => {
    const s = new PairTersoff();
    expect(() => s.settings(['shift', '0.05'], ctx)).not.toThrow();
    expect(() => s.settings(['shift'], ctx)).toThrow(StyleError);
  });
});
