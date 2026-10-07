import { describe, expect, it } from 'vitest';
import {
  BondFene, BondFeneExpand, BondMorse, BondNonlinear, BondClass2,
  BondGromos, BondHarmonicShift, BondHarmonicShiftCut, BondZero,
} from '../src/engine/force/bond/styles';
import {
  AngleCosine, AngleCosineSquared, AngleCosinePeriodic, AngleCosineShift,
  AngleCosineDelta, AngleCharmm, AngleQuartic, AngleFourier,
  AngleFourierSimple, AngleZero,
} from '../src/engine/force/angle/styles';
import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms, buildAtomMap, pushTopo } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { newAccum, StyleError, type Bonded, type BondedCompute, type StyleContext } from '../src/engine/force/types';

/*
 * Wave-1 bond and angle styles: for every style the force on each atom must
 * equal minus the finite-difference derivative of the energy (central
 * differences), over several geometries — including bonds beyond the FENE
 * LJ cutoff (r > 2^(1/6) sigma) and beyond the harmonic/shift/cut rc, and
 * geometry across a periodic boundary. Also checked: per-atom energy/virial
 * tallies, coefficient unit conversion (degrees -> radians), data-file
 * coefficient round trips and the StyleError conditions of the zero styles.
 */

const CTX: StyleContext = { s: null, readFile: () => '', log: () => {} };

/** Several angle geometries, including one across a periodic boundary. */
const angleCases = [
  [[5, 5, 5], [6.0, 5, 5], [6.5, 5.9, 5.1]], // ~120 deg
  [[5, 5, 5], [6.0, 5, 5], [6.1, 5.2, 5.1]], // ~55 deg
  [[5, 5, 5], [6.0, 5, 5], [6.9, 5.4, 5.2]], // ~150 deg
  [[19.5, 5, 5], [0.5, 5, 5], [1.1, 5.9, 5.1]], // across the periodic boundary
];

interface Rig {
  s: ReturnType<typeof emptyState>;
  geom: Geometry;
  map: Int32Array;
}

const rig = (pts: number[][], ntypes: number, L = 20): Rig => {
  const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [L, L, L] }, ntypes, 'molecular');
  addAtoms(s, Float64Array.from(pts.flat()), 1);
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [L, L, L] }));
  return { s, geom, map: buildAtomMap(s) };
};

const makeBc = (r: Rig, kind: 'bond' | 'angle', tallies = false): BondedCompute => {
  const acc = newAccum();
  r.s.f.fill(0);
  const bc: BondedCompute = {
    s: r.s, geom: r.geom, map: r.map, f: r.s.f, acc,
    virial: kind === 'bond' ? acc.vbond : acc.vangle,
    eatom: tallies ? new Float64Array(r.s.n) : null,
    vatom: tallies ? new Float64Array(6 * r.s.n) : null,
  };
  return bc;
};

const energy = (bc: BondedCompute, kind: 'bond' | 'angle'): number => (kind === 'bond' ? bc.acc.ebond : bc.acc.eangle);

/** Central-difference check: f = -dE/dx for every atom of the term. */
const fdCheck = (style: Bonded, kind: 'bond' | 'angle', r: Rig, type: number, ids: number[], tol = 2e-5): number => {
  const bc = makeBc(r, kind);
  style.compute(bc);
  const e0 = energy(bc, kind);
  expect(Number.isFinite(e0)).toBe(true);
  const f0 = r.s.f.slice();
  const h = 1e-6;
  let worst = 0;
  for (const id of ids) {
    const a = r.map[id];
    for (let d = 0; d < 3; d++) {
      const x0 = r.s.x[3 * a + d];
      r.s.x[3 * a + d] = x0 + h;
      const ep = energy(makeBc(r, kind), kind);
      r.s.x[3 * a + d] = x0 - h;
      const em = energy(makeBc(r, kind), kind);
      r.s.x[3 * a + d] = x0;
      const num = -(ep - em) / (2 * h);
      const got = f0[3 * a + d];
      const err = Math.abs(got - num) / Math.max(1, Math.abs(num));
      if (err > worst) worst = err;
    }
  }
  expect(worst, `${style.name} force vs -dE/dr (type ${type})`).toBeLessThan(tol);
  return e0;
};

const initStyle = <T extends Bonded>(style: T, ntypes: number, coeff: number[]): T => {
  style.allocate(ntypes);
  style.coeff(['*', ...coeff.map(String)], CTX);
  style.init(CTX);
  return style;
};

describe('bond styles: force = -dE/dr', () => {
  it('fene (inside and beyond the LJ cutoff 2^(1/6) sigma)', () => {
    const st = initStyle(new BondFene(), 2, [30, 1.5, 1.0, 1.0]);
    const cases = [
      [[5, 5, 5], [6.05, 5.1, 5.05]],
      [[5, 5, 5], [6.4, 5.2, 5.1]], // r ~ 1.43 > 2^(1/6) sigma: LJ part off
      [[19.5, 5, 5], [0.6, 5.1, 5.05]], // across the periodic boundary, r ~ 1.1
    ];
    for (const pts of cases) fdCheck(st, 'bond', rig(pts, 1), 1, [1, 2]);
  });
  it('fene/expand', () => {
    const st = initStyle(new BondFeneExpand(), 1, [30, 1.5, 1.0, 1.0, 0.3]);
    for (const pts of [
      [[5, 5, 5], [6.05, 5.1, 5.05]],
      [[5, 5, 5], [6.6, 5.2, 5.1]], // r - Delta ~ 1.3 beyond 2^(1/6) sigma
    ]) fdCheck(st, 'bond', rig(pts, 1), 1, [1, 2]);
  });
  it('morse', () => {
    const st = initStyle(new BondMorse(), 1, [2.0, 1.8, 1.2]);
    for (const pts of [
      [[5, 5, 5], [6.1, 5.1, 5.05]],
      [[5, 5, 5], [6.8, 5.2, 5.1]],
    ]) fdCheck(st, 'bond', rig(pts, 1), 1, [1, 2]);
  });
  it('nonlinear', () => {
    const st = initStyle(new BondNonlinear(), 1, [100, 1.1, 1.5]);
    for (const pts of [
      [[5, 5, 5], [6.0, 5.1, 5.05]],
      [[5, 5, 5], [6.5, 5.2, 5.1]],
    ]) fdCheck(st, 'bond', rig(pts, 1), 1, [1, 2]);
  });
  it('class2', () => {
    const st = initStyle(new BondClass2(), 1, [1.05, 100, 80, 80]);
    for (const pts of [
      [[5, 5, 5], [6.0, 5.1, 5.05]],
      [[5, 5, 5], [6.4, 5.2, 5.1]],
    ]) fdCheck(st, 'bond', rig(pts, 1), 1, [1, 2]);
  });
  it('gromos', () => {
    const st = initStyle(new BondGromos(), 1, [20, 1.1]);
    for (const pts of [
      [[5, 5, 5], [6.0, 5.1, 5.05]],
      [[5, 5, 5], [6.4, 5.2, 5.1]],
    ]) fdCheck(st, 'bond', rig(pts, 1), 1, [1, 2]);
  });
  it('harmonic/shift', () => {
    const st = initStyle(new BondHarmonicShift(), 1, [10, 1.0, 1.5]);
    for (const pts of [
      [[5, 5, 5], [6.1, 5.1, 5.05]],
      [[5, 5, 5], [6.35, 5.2, 5.1]],
    ]) fdCheck(st, 'bond', rig(pts, 1), 1, [1, 2]);
  });
  it('harmonic/shift/cut (zero energy and force beyond rc)', () => {
    const st = initStyle(new BondHarmonicShiftCut(), 1, [10, 1.0, 1.5]);
    fdCheck(st, 'bond', rig([[5, 5, 5], [6.1, 5.1, 5.05]], 1), 1, [1, 2]);
    // beyond rc: E = 0 and F = 0 ("The bond potential is zero and thus its force also zero for distances r > rc")
    const r = rig([[5, 5, 5], [6.8, 5.2, 5.1]], 1);
    const bc = makeBc(r, 'bond');
    st.compute(bc);
    expect(bc.acc.ebond).toBe(0);
    expect(Array.from(r.s.f)).toEqual(new Array(6).fill(0));
  });
  it('zero computes nothing', () => {
    const st = initStyle(new BondZero(), 1, []);
    const r = rig([[5, 5, 5], [6.1, 5.1, 5.05]], 1);
    const bc = makeBc(r, 'bond');
    st.compute(bc);
    expect(bc.acc.ebond).toBe(0);
    expect(Array.from(r.s.f)).toEqual(new Array(6).fill(0));
  });
});

describe('angle styles: force = -dE/dtheta', () => {
  it('cosine', () => {
    const st = initStyle(new AngleCosine(), 1, [10]);
    for (const pts of angleCases) fdCheck(st, 'angle', rig(pts, 1), 1, [1, 2, 3]);
  });
  it('cosine/squared', () => {
    const st = initStyle(new AngleCosineSquared(), 1, [30, 109.5]);
    for (const pts of angleCases) fdCheck(st, 'angle', rig(pts, 1), 1, [1, 2, 3]);
  });
  it('cosine/periodic (B = 1 and B = -1, several n)', () => {
    for (const [C, B, n] of [[10, 1, 3], [15, -1, 4], [10, 1, 1], [12, -1, 6]]) {
      const st = initStyle(new AngleCosinePeriodic(), 1, [C, B, n]);
      for (const pts of angleCases) fdCheck(st, 'angle', rig(pts, 1), 1, [1, 2, 3]);
    }
  });
  it('cosine/shift', () => {
    const st = initStyle(new AngleCosineShift(), 1, [10, 45]);
    for (const pts of angleCases) fdCheck(st, 'angle', rig(pts, 1), 1, [1, 2, 3]);
  });
  it('cosine/delta', () => {
    const st = initStyle(new AngleCosineDelta(), 1, [20, 60]);
    for (const pts of angleCases) fdCheck(st, 'angle', rig(pts, 1), 1, [1, 2, 3]);
  });
  it('charmm (harmonic + Urey-Bradley on the 1-3 distance)', () => {
    const st = initStyle(new AngleCharmm(), 1, [50, 109.5, 30, 2.0]);
    for (const pts of angleCases) fdCheck(st, 'angle', rig(pts, 1), 1, [1, 2, 3]);
    const st0 = initStyle(new AngleCharmm(), 1, [50, 109.5, 0, 0]);
    for (const pts of angleCases) fdCheck(st0, 'angle', rig(pts, 1), 1, [1, 2, 3]);
  });
  it('quartic', () => {
    const st = initStyle(new AngleQuartic(), 1, [90, 40, -25, -14]);
    for (const pts of angleCases) fdCheck(st, 'angle', rig(pts, 1), 1, [1, 2, 3]);
  });
  it('fourier', () => {
    const st = initStyle(new AngleFourier(), 1, [10, 1, -1, 0.5]);
    for (const pts of angleCases) fdCheck(st, 'angle', rig(pts, 1), 1, [1, 2, 3]);
  });
  it('fourier/simple', () => {
    const st = initStyle(new AngleFourierSimple(), 1, [10, -1, 2]);
    for (const pts of angleCases) fdCheck(st, 'angle', rig(pts, 1), 1, [1, 2, 3]);
  });
  it('zero computes nothing', () => {
    const st = initStyle(new AngleZero(), 1, []);
    const r = rig(angleCases[0], 1);
    pushTopo(r.s.topo.angles, 1, [1, 2, 3]);
    const bc = makeBc(r, 'angle');
    st.compute(bc);
    expect(bc.acc.eangle).toBe(0);
    expect(Array.from(r.s.f)).toEqual(new Array(9).fill(0));
  });
});

describe('per-atom energy/virial tallies', () => {
  it('bond tallies split energy in half and match the virial sum r.f', () => {
    const st = initStyle(new BondMorse(), 1, [2.0, 1.8, 1.2]);
    const r = rig([[5, 5, 5], [6.1, 5.1, 5.05]], 1);
    pushTopo(r.s.topo.bonds, 1, [1, 2]);
    const bc = makeBc(r, 'bond', true);
    st.compute(bc);
    const e = bc.acc.ebond;
    expect(bc.eatom![0]).toBeCloseTo(e / 2, 12);
    expect(bc.eatom![1]).toBeCloseTo(e / 2, 12);
    // single bond: virial = (r_i - r_j) . F_i
    const d = [r.s.x[0] - r.s.x[3], r.s.x[1] - r.s.x[4], r.s.x[2] - r.s.x[5]];
    const w = [d[0] * r.s.f[0], d[1] * r.s.f[1], d[2] * r.s.f[2], d[0] * r.s.f[1], d[0] * r.s.f[2], d[1] * r.s.f[2]];
    for (let c = 0; c < 6; c++) expect(bc.virial[c]).toBeCloseTo(w[c], 10);
    for (let c = 0; c < 6; c++) expect(bc.vatom![c]).toBeCloseTo(w[c] / 2, 10);
  });
  it('angle tallies split energy in thirds and match the virial sum r.f', () => {
    const st = initStyle(new AngleCharmm(), 1, [50, 109.5, 30, 2.0]);
    const r = rig(angleCases[0], 1);
    pushTopo(r.s.topo.angles, 1, [1, 2, 3]);
    const bc = makeBc(r, 'angle', true);
    st.compute(bc);
    const e = bc.acc.eangle;
    // charmm: theta part split in thirds, Urey-Bradley part split in halves;
    // the per-atom energies must still sum to the total
    let esum = 0;
    for (const a of [0, 1, 2]) esum += bc.eatom![a];
    expect(esum).toBeCloseTo(e, 10);
    // virial = sum over atoms of (r_k - r_j) . f_k (rel to middle atom)
    const x = r.s.x, f = r.s.f;
    const rel = [x[0] - x[3], x[1] - x[4], x[2] - x[5], 0, 0, 0, x[6] - x[3], x[7] - x[4], x[8] - x[5]];
    const w = [0, 0, 0, 0, 0, 0];
    for (let a = 0; a < 3; a++) {
      w[0] += rel[3 * a] * f[3 * a]; w[1] += rel[3 * a + 1] * f[3 * a + 1]; w[2] += rel[3 * a + 2] * f[3 * a + 2];
      w[3] += rel[3 * a] * f[3 * a + 1]; w[4] += rel[3 * a] * f[3 * a + 2]; w[5] += rel[3 * a + 1] * f[3 * a + 2];
    }
    for (let c = 0; c < 6; c++) expect(bc.virial[c]).toBeCloseTo(w[c], 9);
  });
});

describe('coefficients', () => {
  it('convert degrees to radians and round-trip through dataCoeffs', () => {
    const st = initStyle(new AngleCosineSquared(), 1, [30, 109.5]);
    expect(st.equilibrium(1)).toBeCloseTo(109.5, 10);
    expect(st.dataCoeffs()).toEqual(['1 30 109.5']);
    const sh = initStyle(new AngleCosineShift(), 1, [10, 45]);
    expect(sh.equilibrium(1)).toBeCloseTo(45, 10);
    const q = initStyle(new AngleQuartic(), 1, [90, 40, -25, -14]);
    expect(q.equilibrium(1)).toBeCloseTo(90, 10);
    expect(q.dataCoeffs()).toEqual(['1 90 40 -25 -14']);
    const c2 = initStyle(new BondClass2(), 1, [1.05, 100, 80, 80]);
    expect(c2.equilibrium(1)).toBe(1.05);
    expect(c2.dataCoeffs()).toEqual(['1 1.05 100 80 80']);
  });
  it('cosine/periodic n stays an integer', () => {
    const st = initStyle(new AngleCosinePeriodic(), 1, [10, -1, 3]);
    expect(st.dataCoeffs()).toEqual(['1 10 -1 3']);
  });
  it('zero styles: coeff must cover all types and takes at most one value', () => {
    const bz = new BondZero();
    bz.allocate(2);
    expect(() => bz.init(CTX)).toThrow(StyleError); // "bond_coeff must be used for all bond types"
    bz.coeff(['*', '1.1'], CTX);
    bz.init(CTX);
    expect(bz.equilibrium(2)).toBe(1.1);
    expect(() => bz.coeff(['1', '1.0', '2.0'], CTX)).toThrow(StyleError);
    expect(() => bz.settings(['bogus'], CTX)).toThrow(StyleError);
    const az = new AngleZero();
    az.allocate(2);
    expect(() => az.init(CTX)).toThrow(StyleError);
    az.coeff(['*'], CTX);
    az.coeff(['2', '120'], CTX);
    az.init(CTX);
    expect(az.equilibrium(2)).toBe(120);
    expect(() => az.coeff(['1', '120', '5'], CTX)).toThrow(StyleError);
  });
  it('missing or wrong-count coefficients throw StyleError', () => {
    const fene = new BondFene();
    fene.allocate(1);
    expect(() => fene.init(CTX)).toThrow(StyleError);
    expect(() => fene.coeff(['*', '30', '1.5', '1.0'], CTX)).toThrow(StyleError); // needs 4 values
    const cs = new AngleCosineSquared();
    cs.allocate(1);
    expect(() => cs.init(CTX)).toThrow(StyleError);
    expect(() => cs.coeff(['*', '30'], CTX)).toThrow(StyleError);
    const ch = new AngleCharmm();
    ch.allocate(1);
    expect(() => ch.coeff(['*', '50', '109.5', '30'], CTX)).toThrow(StyleError); // needs 4 values
  });
  it('wildcard ranges set a range of types', () => {
    const st = initStyle(new BondGromos(), 3, [20, 1.1]);
    expect(st.dataCoeffs()).toEqual(['1 20 1.1', '2 20 1.1', '3 20 1.1']);
  });
});
