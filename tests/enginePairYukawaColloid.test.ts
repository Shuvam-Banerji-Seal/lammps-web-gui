import { describe, expect, it } from 'vitest';
import { PairYukawaColloid } from '../src/engine/force/pair/yukawa_colloid';
import { PAIRS } from '../src/engine/registry/pair_ycolloid';
import { newAccum, type PairCompute, type StyleContext } from '../src/engine/force/types';
import type { Neighbor, NeighList } from '../src/engine/neighbor';
import { emptyState, addAtoms } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import type { SimState } from '../src/engine/types';

/*
 * Unit tests for pair_style yukawa/colloid: the documented energy
 * E = (A/kappa) exp(-kappa (r - (ri + rj))) on a two-sphere system (radii of
 * BOTH particles enter), central finite differences proving force = -dE/dr,
 * the shift offset (energy -> 0 at the cutoff), geometric mixing, and
 * StyleError on bad arguments. The oracle parity itself lives in
 * tests/engineOracle.test.ts (w5yukawa_colloid).
 */

/** Two owned spheres (type 1 radius 0.5, type 2 radius 0.7) r apart on x. */
const sphereState = (r: number): SimState => {
  const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [10, 10, 10], periodic: [true, true, true] }, 2, 'sphere');
  addAtoms(s, new Float64Array([0, 0, 0]), 1);
  addAtoms(s, new Float64Array([r, 0, 0]), 2);
  s.radius![0] = 0.5;
  s.radius![1] = 0.7;
  return s;
};

const make = (st: PairYukawaColloid, s: SimState | null, styleArgs: string[], coeffs: string[][]): PairYukawaColloid => {
  st.allocate(2);
  st.settings(styleArgs, CTX);
  for (const c of coeffs) st.coeff(c, CTX);
  st.init({ ...CTX, s });
  return st;
};

const CTX: StyleContext = { s: null, readFile: () => { throw new Error('no files in unit test'); }, log: () => {} };

/** Full neighbor list of two owned atoms (each lists the other), no ghosts. */
const twoAtomList = (): NeighList => ({
  inum: 2, numneigh: new Int32Array([1, 1]), firstneigh: new Int32Array([0, 1]), neighbors: new Int32Array([1, 0]),
});

interface RunResult { e: number; fx0: number; fx1: number; e0: number; e1: number; vxx: number }

/** Runs compute() once on the two-sphere system with types 1 and 2. */
const run = (st: PairYukawaColloid, s: SimState, r: number): RunResult => {
  const x = new Float64Array([0, 0, 0, r, 0, 0]);
  const f = new Float64Array(6);
  const vatom = new Float64Array(12);
  const pc = {
    s, nb: { owner: new Int32Array([0, 1]) } as unknown as Neighbor, geom: null,
    x, f,
    type: s.type.subarray(0, 2),
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

const near = (a: number, b: number, tol = 1e-12): void => {
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol);
};

const COEFFS = [['1', '1', '100.0'], ['2', '2', '140.0', '3.0'], ['1', '2', '120.0', '3.2']];

describe('pair style yukawa/colloid', () => {
  it('matches the documented formula E = (A/kappa) exp(-kappa (r - (ri + rj))) and F = -dE/dr', () => {
    const s = sphereState(2.0);
    const st = make(new PairYukawaColloid(), s, ['2.0', '3.5'], COEFFS);
    const A = 120, kappa = 2.0, ri = 0.5, rj = 0.7;
    for (const r of [1.4, 2.0, 3.1]) {
      const out = run(st, s, r);
      const ex = Math.exp(-kappa * (r - ri - rj));
      near(out.e, (A / kappa) * ex, 1e-11);
      // repulsive: atom 0 pushed in -x, atom 1 in +x, magnitude -dE/dr
      near(out.fx0, -A * ex, 1e-10);
      near(out.fx1, A * ex, 1e-10);
      near(out.e0, 0.5 * (A / kappa) * ex, 1e-11);
      near(out.e1, 0.5 * (A / kappa) * ex, 1e-11);
      // per-atom virial sums to the pair virial r * (r * fforce) = r * (-dE/dr)
      near(out.vxx, A * ex * r, 1e-9);
    }
  });

  it('force = -dE/dr by central finite differences through compute()', () => {
    const s = sphereState(2.0);
    const st = make(new PairYukawaColloid(), s, ['2.0', '3.5'], COEFFS);
    for (const r of [1.5, 2.2, 3.0]) {
      const h = 1e-6 * r;
      // moving atom 0 by -dx grows the separation, so F_0x = +dE/dr
      const fd = (run(st, s, r + h).e - run(st, s, r - h).e) / (2 * h);
      const out = run(st, s, r);
      expect(Math.abs(out.fx0 - fd), `r=${r}`).toBeLessThan(1e-5 * (1 + Math.abs(fd)));
    }
  });

  it('cutoff is per type pair and both radii enter the energy (1-1 pair)', () => {
    const s = sphereState(2.0);
    const st = make(new PairYukawaColloid(), s, ['2.0', '3.5'], COEFFS);
    near(st.cut[1 * 3 + 1], 3.5);
    near(st.cut[2 * 3 + 2], 3.0);
    near(st.cut[1 * 3 + 2], 3.2);
    // type 1-1 with radius 0.5 + 0.5 at r = 2.0
    s.radius![1] = 0.5;
    const out = run(st, { ...s, type: new Int32Array([1, 1]) } as SimState, 2.0);
    near(out.e, (100 / 2.0) * Math.exp(-2.0 * (2.0 - 1.0)), 1e-11);
  });

  it('pair_modify shift yes: energy is 0 at the cutoff, force unchanged', () => {
    const s = sphereState(3.2);
    const st = make(new PairYukawaColloid(), s, ['2.0', '3.5'], COEFFS);
    st.shift = true;
    st.init({ ...CTX, s });
    near(run(st, s, 3.2).e, 0, 1e-12);
    const A = 120, kappa = 2.0;
    near(run(st, s, 2.0).e, (A / kappa) * Math.exp(-kappa * (2.0 - 1.2)) - (A / kappa) * Math.exp(-kappa * (3.2 - 1.2)), 1e-11);
  });

  it('mixes A like a LJ epsilon and the cutoff like a distance, geometrically', () => {
    const st = make(new PairYukawaColloid(), null, ['2.0', '3.5'], [['1', '1', '100.0'], ['2', '2', '140.0', '3.0']]);
    near(st.p.get('A', 1, 2), Math.sqrt(100 * 140));
    near(st.p.get('cut', 1, 2), Math.sqrt(3.5 * 3.0));
  });

  it('reads ghost radii through the owner map', () => {
    const s = sphereState(2.0);
    const st = make(new PairYukawaColloid(), s, ['2.0', '3.5'], COEFFS);
    // one owned atom (type 1, radius 0.5) plus one ghost whose owner carries the
    // radius; a wrong ghost-radius lookup (indexing the ghost slot) would see
    // radius[1] = 0.9 instead of the owner's 0.5
    s.radius![1] = 0.9;
    const x = new Float64Array([0, 0, 0, 2.0, 0, 0]);
    const f = new Float64Array(6);
    const pc = {
      s, nb: { owner: new Int32Array([0, 0]) } as unknown as Neighbor, geom: null,
      x, f,
      type: new Int32Array([1, 1]),
      q: new Float64Array(2),
      nlocal: 1, nall: 2,
      half: null,
      full: { inum: 1, numneigh: new Int32Array([1]), firstneigh: new Int32Array([0]), neighbors: new Int32Array([1]) },
      specialLJ: new Float64Array([1, 1, 1, 1]),
      specialCoul: new Float64Array([1, 1, 1, 1]),
      qqrd2e: 1,
      acc: newAccum(),
      eatom: null,
      vatom: null,
    } as unknown as PairCompute;
    st.compute(pc);
    // a self-image pair is one ordered term: half of the pair energy and half of
    // the force (the mirror image supplies the other half in a real system)
    near(pc.acc.evdwl, 0.5 * (100 / 2.0) * Math.exp(-2.0 * (2.0 - 1.0)), 1e-11);
    near(f[0], -0.5 * 100 * Math.exp(-2.0 * (2.0 - 1.0)), 1e-10);
    near(f[3], 0.5 * 100 * Math.exp(-2.0 * (2.0 - 1.0)), 1e-10);
  });

  it('throws on bad arguments', () => {
    expect(() => new PairYukawaColloid().settings(['2.0'], CTX)).toThrow(/usage: pair_style yukawa\/colloid kappa cutoff/);
    expect(() => new PairYukawaColloid().settings(['0.0', '3.5'], CTX)).toThrow(/kappa must be > 0/);
    expect(() => new PairYukawaColloid().settings(['2.0', '0'], CTX)).toThrow(/cutoff must be > 0/);
    expect(() => new PairYukawaColloid().coeff(['1', '1', '1.0'], CTX)).toThrow(/needs the simulation box/);
    const st = new PairYukawaColloid();
    st.allocate(2);
    expect(() => st.coeff(['1', '1'], CTX)).toThrow(/usage: pair_coeff I J A \[cutoff\]/);
    expect(() => st.coeff(['1', '1', '100.0', '3.5', '9'], CTX)).toThrow(/usage/);
    const st2 = new PairYukawaColloid();
    st2.allocate(2);
    st2.settings(['2.0', '3.5'], CTX);
    st2.coeff(['1', '1', '100.0'], CTX);
    expect(() => st2.init(CTX)).toThrow(/all pair coeffs are not set/);
    const st3 = make(new PairYukawaColloid(), null, ['2.0', '3.5'], COEFFS);
    st3.tail = true;
    expect(() => st3.init(CTX)).toThrow(/tail is not supported/);
  });

  it('requires atom_style sphere radii', () => {
    const st = new PairYukawaColloid();
    st.allocate(2);
    st.settings(['2.0', '3.5'], CTX);
    st.coeff(['1', '1', '100.0'], CTX);
    st.coeff(['2', '2', '100.0'], CTX);
    const atomic = { ...sphereState(2.0), radius: null, rmass: null, omega: null, torque: null } as unknown as SimState;
    expect(() => st.init({ ...CTX, s: atomic })).toThrow(/requires atom_style sphere/);
  });

  it('is registered as yukawa/colloid', () => {
    expect(PAIRS['yukawa/colloid']).toBeTypeOf('function');
    expect(PAIRS['yukawa/colloid']!().name).toBe('yukawa/colloid');
  });
});
