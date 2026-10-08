import { describe, expect, it } from 'vitest';
import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms, buildAtomMap, pushTopo } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { newAccum, clearAccum, StyleError, type BondedCompute } from '../src/engine/force/types';
import type { SimpleBonded } from '../src/engine/force/bonded_util';
import { AngleCosineSquaredRestricted, AngleMm3 } from '../src/engine/force/angle/misc18';

/*
 * Unit tests for the wave-18 angle styles: cosine/squared/restricted and mm3.
 *
 * Measured with native LAMMPS (black box): for cosine/squared/restricted the
 * reported energy is K [cos(theta) - cos(theta0)]^2 / sin^2(theta) and the
 * forces are the exact gradient of that energy, so the finite-difference check
 * below expects f = -dE/dx for the documented energy. The native values for
 * one geometry and coefficients are hard coded as literals.
 *
 * For mm3 the native energy is K dR^2 bE(dDeg) and the native forces are the
 * gradient of the SAME leading term with a slightly different bracket bF
 * (native's energy and force are not consistent to 1e-12; see the header of
 * angle/misc18.ts). The finite-difference check therefore expects
 * f = -dE_F/dx for the force bracket energy E_F = K dR^2 bF(dDeg), and the
 * energy check uses bE. Both brackets are the measured native ones.
 */

const DEG = 180 / Math.PI;
const GEOMS: number[][][] = [
  [[5, 5, 5], [6.0, 5, 5], [6.5, 5.9, 5.1]], // ~118.9 deg
  [[5, 5, 5], [6.0, 5, 5], [6.1, 5.2, 5.1]], // ~55 deg
  [[5, 5, 5], [6.0, 5, 5], [6.9, 5.4, 5.2]], // ~150 deg
  [[19.5, 5, 5], [0.5, 5, 5], [1.1, 5.9, 5.1]], // across the periodic boundary
];

const make = (style: SimpleBonded, pts: number[][]) => {
  const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [20, 20, 20] }, 1, 'molecular');
  addAtoms(s, Float64Array.from(pts.flat()), 1);
  pushTopo(s.topo.angles, 1, [1, 2, 3]);
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [20, 20, 20] }));
  const bc: BondedCompute = { s, geom, map: buildAtomMap(s), f: s.f, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null };
  return { s, bc, geom };
};

/** Zeroes accumulators, runs compute, returns the eangle (forces left in bc.f). */
const energy = (style: SimpleBonded, bc: BondedCompute): number => {
  clearAccum(bc.acc);
  bc.f.fill(0);
  bc.virial.fill(0);
  style.compute(bc);
  return bc.acc.eangle;
};

const points = (s: ReturnType<typeof make>['s']): number[][] => [0, 1, 2].map((a) => [s.x[3 * a], s.x[3 * a + 1], s.x[3 * a + 2]]);

/** 20-box wrap so the reference sees the same minimum-image deltas as the engine. */
const wrap = (P: number[][]): number[][] => P.map((q) => q.map((c) => c - 20 * Math.round(c / 20)));
const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: number[]) => Math.sqrt(dot(a, a));
const cosTheta = (raw: number[][]): number => {
  const p = wrap(raw);
  const d1 = sub(p[0], p[1]), d2 = sub(p[2], p[1]);
  return Math.max(-1, Math.min(1, dot(d1, d2) / (norm(d1) * norm(d2))));
};

/** K [cos(theta) - cos(theta0)]^2 / sin^2(theta), theta0 in degrees. */
const csrDoc = (K: number, th0deg: number) => (P: number[][]): number => {
  const c = cosTheta(P), u = c - Math.cos((th0deg * Math.PI) / 180);
  return (K * u * u) / (1 - c * c);
};

// measured native mm3 brackets (see src/engine/force/angle/misc18.ts header)
const BE = [1, -1.400000151523997e-2, 5.599995260133318e-5, -7.000017854655080e-7, 2.200000132720880e-8];
const BF = [1, -1.399999569747566e-2, 5.5999952601329771e-5, -6.8884952102526184e-7, 2.2000001327209518e-8];
const poly = (b: number[], d: number): number => b[0] + b[1] * d + b[2] * d * d + b[3] * d ** 3 + b[4] * d ** 4;

/** mm3 energy with a given bracket: K (theta-theta0)^2 b(theta-theta0), d in degrees. */
const mm3Energy = (b: number[], K: number, th0deg: number) => (P: number[][]): number => {
  const theta = (Math.acos(cosTheta(P)) * 180) / Math.PI;
  const dR = ((theta - th0deg) * Math.PI) / 180;
  return K * dR * dR * poly(b, theta - th0deg);
};

const fresh = (make2: () => SimpleBonded, coeffs: string[]) => {
  const style = make2();
  style.allocate(1);
  style.coeff(['1', ...coeffs]);
  style.init();
  return style;
};

/** f = -dE/dx by central differences for the given energy reference. */
const checkForces = (style: SimpleBonded, pts: number[][], docEnergy: (P: number[][]) => number, tol = 5) => {
  const { s, bc, geom } = make(style, pts);
  energy(style, bc); // run compute so bc.f holds the analytic forces
  const fAna = Array.from(bc.f);
  const h = 1e-6;
  for (let a = 0; a < 3; a++) {
    for (let d = 0; d < 3; d++) {
      const k = 3 * a + d;
      const x0 = s.x[k];
      s.x[k] = x0 + h;
      const ep = docEnergy(points(s));
      s.x[k] = x0 - h;
      const em = docEnergy(points(s));
      s.x[k] = x0;
      expect(fAna[k], `atom ${a} dim ${d}`).toBeCloseTo(-(ep - em) / (2 * h), tol);
    }
  }
  // virial = sum_k (r_k - r_j) . f_k (minimum image), j is the middle atom
  const ref = new Float64Array(6);
  for (let a = 0; a < 3; a++) {
    const rel = [0, 0, 0];
    for (let c = 0; c < 3; c++) rel[c] = s.x[3 * a + c] - s.x[3 + c];
    geom.minimumImage(rel);
    const [rx, ry, rz] = rel, [fx, fy, fz] = [bc.f[3 * a], bc.f[3 * a + 1], bc.f[3 * a + 2]];
    ref[0] += rx * fx; ref[1] += ry * fy; ref[2] += rz * fz;
    ref[3] += rx * fy; ref[4] += rx * fz; ref[5] += ry * fz;
  }
  for (let c = 0; c < 6; c++) expect(bc.virial[c], `virial ${c}`).toBeCloseTo(ref[c], 6);
};

describe('angle_style cosine/squared/restricted: f = -dE/dx by central finite differences', () => {
  it('all four geometries (K 30, theta0 109.5)', () => {
    const style = fresh(() => new AngleCosineSquaredRestricted(), ['30.0', '109.5']);
    for (const pts of GEOMS) checkForces(style, pts, csrDoc(30.0, 109.5));
  });
  it('other coefficients (K 100, theta0 180 and K 2.5, theta0 60)', () => {
    const a = fresh(() => new AngleCosineSquaredRestricted(), ['100.0', '180']);
    const b = fresh(() => new AngleCosineSquaredRestricted(), ['2.5', '60']);
    for (const pts of GEOMS) {
      checkForces(a, pts, csrDoc(100.0, 180));
      checkForces(b, pts, csrDoc(2.5, 60));
    }
  });
  it('reproduces native LAMMPS energy and forces (K 30, theta0 109.5)', () => {
    const style = fresh(() => new AngleCosineSquaredRestricted(), ['30.0', '109.5']);
    const { bc } = make(style, GEOMS[0]);
    expect(energy(style, bc)).toBeCloseTo(0.87564916563854811, 13);
    const native = [
      0, 11.149133224450088, 1.2387925804944493,
      9.4935506168733852, -16.359008562978165, -1.8176676181086777,
      -9.4935506168733852, 5.209875338528076, 0.57887503761422832,
    ];
    for (let k = 0; k < 9; k++) expect(bc.f[k]).toBeCloseTo(native[k], 12);
  });
  it('theta0 is read in degrees and written back in degrees', () => {
    const style = fresh(() => new AngleCosineSquaredRestricted(), ['30.0', '109.5']);
    expect(style.equilibrium(1)).toBeCloseTo(109.5, 10);
    expect(style.dataCoeffs()).toEqual(['1 30 109.5']);
  });
});

describe('angle_style mm3', () => {
  it('energy is the measured native bracket (K 30, theta0 109.5)', () => {
    const style = fresh(() => new AngleMm3(), ['30.0', '109.5']);
    const { bc } = make(style, GEOMS[0]);
    expect(energy(style, bc)).toBeCloseTo(0.7056639473359354, 13);
    // and it is exactly K dR^2 bE(dDeg)
    const P = GEOMS[0];
    const th = (Math.acos(cosTheta(P)) * 180) / Math.PI;
    const dR = ((th - 109.5) * Math.PI) / 180;
    expect(energy(style, bc)).toBeCloseTo(30 * dR * dR * poly(BE, th - 109.5), 13);
  });
  it('forces are the gradient of the measured force bracket (all four geometries)', () => {
    const style = fresh(() => new AngleMm3(), ['30.0', '109.5']);
    for (const pts of GEOMS) checkForces(style, pts, mm3Energy(BF, 30.0, 109.5));
  });
  it('other coefficients (K 100, theta0 90 and K 25, theta0 120)', () => {
    const a = fresh(() => new AngleMm3(), ['100.0', '90']);
    const b = fresh(() => new AngleMm3(), ['25.0', '120']);
    for (const pts of GEOMS) {
      checkForces(a, pts, mm3Energy(BF, 100.0, 90));
      checkForces(b, pts, mm3Energy(BF, 25.0, 120));
    }
  });
  it('reproduces native LAMMPS energy and forces (K 30, theta0 109.5)', () => {
    const style = fresh(() => new AngleMm3(), ['30.0', '109.5']);
    const { bc } = make(style, GEOMS[0]);
    expect(energy(style, bc)).toBeCloseTo(0.7056639473359354, 13);
    const native = [
      0, 7.9437597746629498, 0.88263997496254654,
      6.764156817470008, -11.65579705254283, -1.2950885613936429,
      -6.764156817470008, 3.7120372778798809, 0.41244858643109628,
    ];
    for (let k = 0; k < 9; k++) expect(bc.f[k]).toBeCloseTo(native[k], 12);
  });
  it('theta0 is read in degrees and written back in degrees', () => {
    const style = fresh(() => new AngleMm3(), ['30.0', '109.5']);
    expect(style.equilibrium(1)).toBeCloseTo(109.5, 10);
    expect(style.dataCoeffs()).toEqual(['1 30 109.5']);
  });
});

describe('wave-18 angle styles: missing or bad coefficients throw', () => {
  it('init without coefficients throws for every style', () => {
    for (const mk of [() => new AngleCosineSquaredRestricted(), () => new AngleMm3()]) {
      const style = mk();
      style.allocate(1);
      expect(() => style.init()).toThrow(StyleError);
    }
  });
  it('wrong coefficient counts and non-numeric values throw', () => {
    const csr = new AngleCosineSquaredRestricted(); csr.allocate(1);
    expect(() => csr.coeff(['1', '30'])).toThrow(StyleError);
    expect(() => csr.coeff(['1', '30', '109.5', '2'])).toThrow(StyleError);
    expect(() => csr.coeff(['1', 'k', 'x'])).toThrow(StyleError);
    const mm3 = new AngleMm3(); mm3.allocate(1);
    expect(() => mm3.coeff(['1', '30'])).toThrow(StyleError);
    expect(() => mm3.coeff(['1', '30', '109.5', '2'])).toThrow(StyleError);
  });
});
