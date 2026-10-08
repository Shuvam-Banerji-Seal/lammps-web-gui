import { describe, expect, it } from 'vitest';
import { PairColloid } from '../src/engine/force/pair/colloid';
import { PAIRS } from '../src/engine/registry/pair_colloid';
import { newAccum, type PairCompute, type StyleContext } from '../src/engine/force/types';
import type { Neighbor, NeighList } from '../src/engine/neighbor';
import { emptyState, addAtoms } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import type { SimState } from '../src/engine/types';

/*
 * Unit tests for pair_style colloid: the three documented energy formulas
 * (colloid-colloid, colloid-solvent, solvent-solvent, selected by the two
 * diameters d1 d2), central finite differences proving the forces are the
 * analytic derivatives, per-atom energy/virial splitting, per type pair
 * cutoffs, geometric mixing of A sigma d1 d2 cut, the pair_modify shift
 * offset, and StyleError on bad arguments. The oracle parity itself lives in
 * tests/engineOracle.test.ts (w5colloid_mix).
 */

/* Reference energies, transcribed from docs.lammps.org/pair_colloid.html. */

/** U_A + U_R for two colloids of radii a1, a2. */
const uCC = (A: number, sig: number, a1: number, a2: number, r: number): number => {
  const q1 = r * r - (a1 + a2) ** 2, q2 = r * r - (a1 - a2) ** 2;
  const ua = -(A / 6) * (2 * a1 * a2 / q1 + 2 * a1 * a2 / q2 + Math.log(q1 / q2));
  const kp = a1 * a1 + 7 * a1 * a2 + a2 * a2, km = a1 * a1 - 7 * a1 * a2 + a2 * a2;
  const t1 = (r * r - 7 * r * (a1 + a2) + 6 * kp) / (r - a1 - a2) ** 7;
  const t2 = (r * r + 7 * r * (a1 + a2) + 6 * kp) / (r + a1 + a2) ** 7;
  const t3 = (r * r + 7 * r * (a1 - a2) + 6 * km) / (r + a1 - a2) ** 7;
  const t4 = (r * r - 7 * r * (a1 - a2) + 6 * km) / (r - a1 + a2) ** 7;
  return ua + (A / 37800) * (sig ** 6 / r) * (t1 + t2 - t3 - t4);
};

/** Colloid (radius a) - solvent. */
const uCS = (A: number, sig: number, a: number, r: number): number => {
  const s = 5 * a ** 6 + 45 * a ** 4 * r * r + 63 * a * a * r ** 4 + 15 * r ** 6;
  return ((2 * a ** 3 * sig ** 3 * A) / (9 * (a * a - r * r) ** 3))
    * (1 - (s * sig ** 6) / (15 * (a - r) ** 6 * (a + r) ** 6));
};

/** Solvent-solvent Lennard-Jones form. */
const uSS = (A: number, sig: number, r: number): number => (A / 36) * ((sig / r) ** 12 - (sig / r) ** 6);

const CTX: StyleContext = { s: null, readFile: () => { throw new Error('no files in unit test'); }, log: () => {} };

/** Two owned atoms of the given types r apart on x. */
const pairState = (t1: number, t2: number, r: number): SimState => {
  const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [30, 30, 30], periodic: [true, true, true] }, 2, 'atomic');
  addAtoms(s, new Float64Array([0, 0, 0]), t1);
  addAtoms(s, new Float64Array([r, 0, 0]), t2);
  return s;
};

const COEFFS = [
  ['1', '1', '144.0', '1.0', '5.0', '5.0', '12.0'],
  ['1', '2', '75.4', '1.0', '5.0', '0.0', '8.0'],
  ['2', '2', '39.5', '1.0', '0.0', '0.0', '2.5'],
];

const make = (styleArgs: string[], coeffs: string[][], shift = false): PairColloid => {
  const st = new PairColloid();
  st.allocate(2);
  st.settings(styleArgs);
  for (const c of coeffs) st.coeff(c);
  st.shift = shift;
  st.init({ ...CTX, s: null });
  return st;
};

/** Full neighbor list of two owned atoms (each lists the other), no ghosts. */
const twoAtomList = (): NeighList => ({
  inum: 2, numneigh: new Int32Array([1, 1]), firstneigh: new Int32Array([0, 1]), neighbors: new Int32Array([1, 0]),
});

interface RunResult { e: number; fx0: number; fx1: number; e0: number; e1: number; vxx: number }

/** Runs compute() once on the two-atom system. */
const run = (st: PairColloid, t1: number, t2: number, r: number): RunResult => {
  const x = new Float64Array([0, 0, 0, r, 0, 0]);
  const f = new Float64Array(6);
  const vatom = new Float64Array(12);
  const pc = {
    s: null, nb: { owner: new Int32Array([0, 1]) } as unknown as Neighbor, geom: null,
    x, f,
    type: Int32Array.from([t1, t2]),
    q: new Float64Array(2),
    nlocal: 2, nall: 2,
    half: null,
    full: twoAtomList(),
    specialLJ: new Float64Array([1, 1, 1, 1]),
    specialCoul: new Float64Array([1, 1, 1, 1]),
    qqrd2e: 1,
    acc: newAccum(),
    eatom: new Float64Array(2),
    vatom,
  } as unknown as PairCompute;
  st.compute(pc);
  return { e: pc.acc.evdwl, fx0: f[0], fx1: f[3], e0: pc.eatom![0], e1: pc.eatom![1], vxx: vatom[0] + vatom[6] };
};

const near = (a: number, b: number, tol = 1e-11): void => {
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol);
};

describe('pair style colloid', () => {
  it('is registered under its LAMMPS style name', () => {
    expect(PAIRS.colloid).toBeDefined();
    expect(PAIRS.colloid()).toBeInstanceOf(PairColloid);
  });

  it('colloid-colloid (d1 d2 > 0) matches U_A + U_R', () => {
    const st = make(['12.0'], COEFFS);
    for (const r of [5.5, 7.0, 9.9, 11.5]) {
      const out = run(st, 1, 1, r);
      near(out.e, uCC(144.0, 1.0, 2.5, 2.5, r), 1e-10 * (1 + Math.abs(uCC(144.0, 1.0, 2.5, 2.5, r))));
      // per-atom energy split half and half
      near(out.e0, 0.5 * out.e, 1e-10);
      near(out.e1, 0.5 * out.e, 1e-10);
      // per-atom virial sums to the pair virial r * (-dU/dr)
      near(out.vxx, -r * (uCC(144.0, 1.0, 2.5, 2.5, r + 1e-7) - uCC(144.0, 1.0, 2.5, 2.5, r - 1e-7)) / 2e-7, 1e-5);
    }
  });

  it('colloid-solvent (one d zero) matches the documented formula inside and outside the colloid', () => {
    const st = make(['12.0'], COEFFS);
    for (const r of [2.0, 4.0, 6.06, 7.5]) {
      const want = uCS(75.4, 1.0, 2.5, r);
      const out = run(st, 1, 2, r);
      near(out.e, want, 1e-10 * (1 + Math.abs(want)));
      near(out.e0, 0.5 * out.e, 1e-10);
      near(out.e1, 0.5 * out.e, 1e-10);
    }
  });

  it('solvent-solvent (d1 = d2 = 0) is the A/36 Lennard-Jones form', () => {
    const st = make(['12.0'], COEFFS);
    for (const r of [1.2, 1.8, 2.4]) {
      const out = run(st, 2, 2, r);
      near(out.e, uSS(39.5, 1.0, r), 1e-12);
    }
    // beyond the 2.5 cutoff: no interaction
    const out = run(st, 2, 2, 2.6);
    near(out.e, 0, 0);
    near(out.fx0, 0, 0);
  });

  it('forces are the analytic derivatives by central finite differences (all three forms)', () => {
    const st = make(['12.0'], COEFFS);
    for (const [t1, t2, r] of [[1, 1, 7.0], [1, 1, 5.6], [1, 2, 6.06], [1, 2, 3.0], [2, 2, 1.5]] as const) {
      const h = 1e-6 * r;
      // moving atom 0 by -dx grows the separation, so F_0x = +dE/dr
      const fd = (run(st, t1, t2, r + h).e - run(st, t1, t2, r - h).e) / (2 * h);
      const out = run(st, t1, t2, r);
      expect(Math.abs(out.fx0 - fd), `types ${t1},${t2} r=${r}`).toBeLessThan(1e-5 * (1 + Math.abs(fd)));
      expect(Math.abs(out.fx0 + out.fx1), `newton pair ${t1},${t2} r=${r}`).toBeLessThan(1e-12);
    }
  });

  it('cutoffs are per type pair', () => {
    const st = make(['12.0'], COEFFS);
    near(st.cut[1 * 3 + 1], 12.0);
    near(st.cut[1 * 3 + 2], 8.0);
    near(st.cut[2 * 3 + 2], 2.5);
  });

  it('unset I != J pairs mix A like epsilon and sigma/d1/d2/cut like sigma (geometric default)', () => {
    const st = make(['10.0'], [
      ['1', '1', '144.0', '4.0', '5.0', '2.0', '12.0'],
      ['2', '2', '39.5', '1.0', '0.8', '0.0', '3.0'],
    ]);
    near(st.p.get('a', 1, 2), Math.sqrt(144.0 * 39.5));
    near(st.p.get('sigma', 1, 2), Math.sqrt(4.0 * 1.0));
    near(st.p.get('d1', 1, 2), Math.sqrt(5.0 * 0.8));
    near(st.p.get('d2', 1, 2), 0);
    near(st.p.get('cut', 1, 2), Math.sqrt(12.0 * 3.0));
    near(st.cut[1 * 3 + 2], Math.sqrt(12.0 * 3.0));
  });

  it('pair_modify shift yes: energy is 0 at the cutoff, force unchanged', () => {
    const st = make(['12.0'], COEFFS, true);
    const cutCC = 12.0, cutCS = 8.0, cutSS = 2.5;
    near(run(st, 1, 1, cutCC).e, 0, 1e-9);
    near(run(st, 1, 2, cutCS).e, 0, 1e-9);
    near(run(st, 2, 2, cutSS).e, 0, 1e-9);
    // shifted energy = unshifted - u(cutoff); force untouched
    const r = 7.0;
    near(run(st, 1, 1, r).e, uCC(144.0, 1.0, 2.5, 2.5, r) - uCC(144.0, 1.0, 2.5, 2.5, cutCC), 1e-10);
    const fd = (run(st, 1, 1, r + 1e-5).e - run(st, 1, 1, r - 1e-5).e) / 2e-5;
    near(run(st, 1, 1, r).fx0, fd, 1e-6);
  });

  it('throws StyleError on bad arguments', () => {
    expect(() => make([], COEFFS)).toThrow(/pair_style colloid cutoff/);
    expect(() => make(['12.0', '3.0'], COEFFS)).toThrow(/pair_style colloid cutoff/);
    expect(() => make(['0.0'], COEFFS)).toThrow(/cutoff must be > 0/);
    expect(() => make(['12.0'], [['1', '1', '144.0', '1.0', '-5.0', '5.0', '12.0']])).toThrow(/d1 and d2 must be values >= 0/);
    expect(() => make(['12.0'], [['1', '1', '144.0', '1.0', '5.0']])).toThrow(/usage: pair_coeff/);
    // tail is not supported
    const st = new PairColloid();
    st.allocate(2);
    st.settings(['12.0']);
    for (const c of COEFFS) st.coeff(c);
    st.tail = true;
    expect(() => st.init({ ...CTX, s: null })).toThrow(/tail is not supported for pair style colloid/);
    // a type pair without coefficients cannot be initialized
    const st2 = new PairColloid();
    st2.allocate(2);
    st2.settings(['12.0']);
    st2.coeff(['1', '1', '144.0', '1.0', '5.0', '5.0', '12.0']);
    expect(() => st2.init({ ...CTX, s: null })).toThrow(/all pair coeffs are not set \(pair 1 2\)/);
  });
});
