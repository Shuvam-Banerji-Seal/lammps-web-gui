import { describe, expect, it } from 'vitest';
import { BondGaussian } from '../src/engine/force/bond/gaussian';
import { AngleGaussian } from '../src/engine/force/angle/gaussian';
import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms, buildAtomMap, pushTopo } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { newAccum, StyleError, type Bonded, type BondedCompute, type StyleContext } from '../src/engine/force/types';

/*
 * Gaussian bonded styles: the force on every atom must equal minus the
 * finite-difference derivative of the energy (central differences), the
 * energy must match the documented sum-of-Gaussians expression, theta_i must
 * be converted from degrees to radians and coefficient round trips must work.
 */

const CTX: StyleContext = { s: null, readFile: () => '', log: () => {} };
const SQRT_PI_2 = Math.sqrt(Math.PI / 2);

interface Rig {
  s: ReturnType<typeof emptyState>;
  geom: Geometry;
  map: Int32Array;
}

const rig = (pts: number[][], L = 20): Rig => {
  const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [L, L, L] }, 1, 'molecular');
  addAtoms(s, Float64Array.from(pts.flat()), 1);
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [L, L, L] }));
  return { s, geom, map: buildAtomMap(s) };
};

const bondRig = (pts: number[][], L = 20): Rig => {
  const r = rig(pts, L);
  pushTopo(r.s.topo.bonds, 1, [1, 2]);
  return r;
};

const angleRig = (pts: number[][], L = 20): Rig => {
  const r = rig(pts, L);
  pushTopo(r.s.topo.angles, 1, [1, 2, 3]);
  return r;
};

const makeBc = (r: Rig, kind: 'bond' | 'angle', tallies = false): BondedCompute => {
  const acc = newAccum();
  r.s.f.fill(0);
  return {
    s: r.s, geom: r.geom, map: r.map, f: r.s.f, acc,
    virial: kind === 'bond' ? acc.vbond : acc.vangle,
    eatom: tallies ? new Float64Array(r.s.n) : null,
    vatom: tallies ? new Float64Array(6 * r.s.n) : null,
  };
};

const energy = (bc: BondedCompute, kind: 'bond' | 'angle'): number => (kind === 'bond' ? bc.acc.ebond : bc.acc.eangle);

/** Central-difference check: f = -dE/dx for every atom of the term. */
const fdCheck = (style: Bonded, kind: 'bond' | 'angle', r: Rig, ids: number[], tol = 2e-5): number => {
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
      const bp = makeBc(r, kind);
      style.compute(bp);
      const ep = energy(bp, kind);
      r.s.x[3 * a + d] = x0 - h;
      const bm = makeBc(r, kind);
      style.compute(bm);
      const em = energy(bm, kind);
      r.s.x[3 * a + d] = x0;
      const num = -(ep - em) / (2 * h);
      const err = Math.abs(f0[3 * a + d] - num) / Math.max(1, Math.abs(num));
      if (err > worst) worst = err;
    }
  }
  expect(worst, `${style.kind}_style ${style.name} force vs -dE/dx`).toBeLessThan(tol);
  return e0;
};

const initStyle = <T extends Bonded>(style: T, ntypes: number, coeff: (string | number)[]): T => {
  style.allocate(ntypes);
  style.coeff(['*', ...coeff.map(String)], CTX);
  style.init(CTX);
  return style;
};

// T = 1, n = 2: (A, w, r0) = (1.0, 0.3, 2.0), (0.5, 0.4, 3.0)
const BOND_COEFF = [1.0, 2, 1.0, 0.3, 2.0, 0.5, 0.4, 3.0];
// T = 1, n = 2: (A, w, theta_deg) = (1.0, 0.3, 100), (0.5, 0.5, 90)
const ANGLE_COEFF = [1.0, 2, 1.0, 0.3, 100.0, 0.5, 0.5, 90.0];

describe('bond_style gaussian', () => {
  it('force = -dE/dr for a two-term gaussian, generic and periodic', () => {
    const st = initStyle(new BondGaussian(), 1, BOND_COEFF);
    for (const pts of [
      [[5, 5, 5], [7.1, 5.2, 5.1]],
      [[5, 5, 5], [8.0, 5.3, 5.2]],
      [[19.5, 5, 5], [0.6, 5.1, 5.05]],
    ]) fdCheck(st, 'bond', bondRig(pts), [1, 2]);
  });

  it('energy matches the documented sum-of-Gaussians', () => {
    const st = initStyle(new BondGaussian(), 1, BOND_COEFF);
    const r = bondRig([[5, 5, 5], [7.1, 5, 5]]); // r = 2.1
    const bc = makeBc(r, 'bond');
    st.compute(bc);
    const terms = [[1.0, 0.3, 2.0], [0.5, 0.4, 3.0]];
    let S = 0;
    for (const [A, w, r0] of terms) S += (A / (w * SQRT_PI_2)) * Math.exp((-2 * (2.1 - r0) ** 2) / w ** 2);
    expect(bc.acc.ebond).toBeCloseTo(-Math.log(S), 12);
  });

  it('coefficients, data round trip and errors', () => {
    const st = initStyle(new BondGaussian(), 1, BOND_COEFF);
    expect(st.dataCoeffs()).toEqual(['1 1 2 1 0.3 2 0.5 0.4 3']);
    const unset = new BondGaussian();
    unset.allocate(1);
    expect(() => unset.init(CTX)).toThrow(StyleError);
    expect(() => unset.coeff(['*', '1', '2', '1', '0.3'], CTX)).toThrow(StyleError);
    expect(() => unset.coeff(['*', '1', '0', '1', '0.3', '2'], CTX)).toThrow(StyleError);
    expect(() => st.settings(['bogus'], CTX)).toThrow(StyleError);
  });
});

describe('angle_style gaussian', () => {
  const angleCases = [
    [[6.0, 5, 5], [5, 5, 5], [4.82635182233307, 5.984807753012208, 5]], // exactly 100 deg at atom 2
    [[5, 5, 5], [6.0, 5, 5], [6.1, 5.2, 5.1]],
    [[5, 5, 5], [6.0, 5, 5], [6.9, 5.4, 5.2]],
    [[19.5, 5, 5], [0.5, 5, 5], [1.1, 5.9, 5.1]], // across the periodic boundary
  ];

  it('force = -dE/dtheta for a two-term gaussian', () => {
    const st = initStyle(new AngleGaussian(), 1, ANGLE_COEFF);
    for (const pts of angleCases) fdCheck(st, 'angle', angleRig(pts), [1, 2, 3]);
  });

  it('energy matches the documented sum-of-Gaussians (theta_i in degrees)', () => {
    const st = initStyle(new AngleGaussian(), 1, ANGLE_COEFF);
    const r = angleRig(angleCases[0]); // theta = 100 deg
    const bc = makeBc(r, 'angle');
    st.compute(bc);
    const theta = Math.PI / 2 + (10 * Math.PI) / 180; // 100 deg
    const terms = [[1.0, 0.3, 100], [0.5, 0.5, 90]];
    let S = 0;
    for (const [A, w, deg] of terms) S += (A / (w * SQRT_PI_2)) * Math.exp((-2 * (theta - (deg * Math.PI) / 180) ** 2) / w ** 2);
    expect(bc.acc.eangle).toBeCloseTo(-Math.log(S), 12);
  });

  it('coefficients convert degrees to radians and round trip', () => {
    const st = initStyle(new AngleGaussian(), 1, ANGLE_COEFF);
    expect(st.dataCoeffs()).toEqual(['1 1 2 1 0.3 100 0.5 0.5 90']);
    const unset = new AngleGaussian();
    unset.allocate(1);
    expect(() => unset.init(CTX)).toThrow(StyleError);
    expect(() => unset.coeff(['*', '1', '2', '1', '0.3'], CTX)).toThrow(StyleError);
    expect(() => unset.coeff(['*', '1', '0', '1', '0.3', '100'], CTX)).toThrow(StyleError);
    expect(() => st.settings(['bogus'], CTX)).toThrow(StyleError);
  });
});
