import { describe, expect, it } from 'vitest';
import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms, buildAtomMap, pushTopo } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { newAccum, clearAccum, StyleError, type BondedCompute } from '../src/engine/force/types';
import type { SimpleBonded } from '../src/engine/force/bonded_util';
import {
  DihedralOpls, DihedralMultiHarmonic, DihedralFourier, DihedralQuadratic,
  DihedralNHarmonic, DihedralCosineShiftExp, DihedralHelix, DihedralZero,
} from '../src/engine/force/dihedral/styles';

/*
 * Unit tests for the wave-1 dihedral styles: central finite differences must
 * reproduce f = -dE/dx for every atom at three geometries (two generic and
 * one crossing a periodic boundary, exercising minimum image), and a style
 * without coefficients must refuse to init.
 */

const GEOMS: number[][][] = [
  [[5.1, 5.2, 4.7], [6.0, 5.1, 5.05], [6.6, 6.1, 4.8], [7.4, 6.3, 5.7]],
  [[5, 5, 5], [6, 5, 5], [6.5, 5.9, 5.1], [7.5, 6.0, 4.0]],
  [[19.6, 5.2, 4.7], [0.5, 5.1, 5.05], [1.1, 6.1, 4.8], [1.9, 6.3, 5.7]],
];

const make = (style: SimpleBonded, pts: number[][]) => {
  const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [20, 20, 20] }, 1, 'molecular');
  addAtoms(s, Float64Array.from(pts.flat()), 1);
  pushTopo(s.topo.dihedrals, 1, [1, 2, 3, 4]);
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [20, 20, 20] }));
  const bc: BondedCompute = { s, geom, map: buildAtomMap(s), f: s.f, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null };
  return { s, bc };
};

/** Zeroes accumulators, runs compute, returns edihed (forces left in bc.f). */
const energy = (style: SimpleBonded, bc: BondedCompute): number => {
  clearAccum(bc.acc);
  bc.f.fill(0);
  bc.virial.fill(0);
  style.compute(bc);
  return bc.acc.edihed;
};

const checkForces = (style: SimpleBonded, pts: number[][]) => {
  const { s, bc } = make(style, pts);
  energy(style, bc);
  const fAna = Array.from(bc.f);
  const h = 1e-6;
  for (let a = 0; a < 4; a++) {
    for (let d = 0; d < 3; d++) {
      const k = 3 * a + d;
      const x0 = s.x[k];
      s.x[k] = x0 + h;
      const ep = energy(style, bc);
      s.x[k] = x0 - h;
      const em = energy(style, bc);
      s.x[k] = x0;
      expect(fAna[k], `atom ${a} dim ${d}`).toBeCloseTo(-(ep - em) / (2 * h), 5);
    }
  }
};

/** coeff + init + force checks for one style at all three geometries. */
const checkStyle = (make2: () => SimpleBonded, coeffs: string[]) => {
  const style = make2();
  style.allocate(1);
  style.coeff(['1', ...coeffs]);
  style.init();
  for (const pts of GEOMS) checkForces(style, pts);
};

describe('dihedral styles: f = -dE/dx by central finite differences', () => {
  it('opls', () => checkStyle(() => new DihedralOpls(), ['1.2', '-0.3', '0.6', '0.1']));
  it('multi/harmonic', () => checkStyle(() => new DihedralMultiHarmonic(), ['1.0', '-0.5', '0.8', '0.2', '-0.1']));
  it('fourier', () => checkStyle(() => new DihedralFourier(), ['2', '1.0', '1', '0.0', '0.6', '3', '180.0']));
  it('quadratic', () => checkStyle(() => new DihedralQuadratic(), ['2.0', '120.0']));
  it('nharmonic', () => checkStyle(() => new DihedralNHarmonic(), ['4', '1.0', '-0.5', '0.8', '0.2']));
  it('cosine/shift/exp', () => checkStyle(() => new DihedralCosineShiftExp(), ['1.5', '45.0', '2.0']));
  it('cosine/shift/exp in the linear-order small-a branch', () =>
    checkStyle(() => new DihedralCosineShiftExp(), ['1.5', '45.0', '0.0005']));
  it('helix', () => checkStyle(() => new DihedralHelix(), ['1.0', '0.8', '0.5']));
  it('zero computes no energy and no force', () => {
    const style = new DihedralZero();
    style.allocate(1);
    style.coeff(['1']);
    style.init();
    for (const pts of GEOMS) {
      const { bc } = make(style, pts);
      expect(energy(style, bc)).toBe(0);
      for (let k = 0; k < 12; k++) expect(bc.f[k]).toBe(0);
    }
  });
});

describe('dihedral styles: missing or bad coefficients throw', () => {
  const fresh = (make2: () => SimpleBonded) => {
    const style = make2();
    style.allocate(1);
    return style;
  };

  it('init without coefficients throws for every style', () => {
    const makers = [
      () => new DihedralOpls(), () => new DihedralMultiHarmonic(), () => new DihedralFourier(),
      () => new DihedralQuadratic(), () => new DihedralNHarmonic(), () => new DihedralCosineShiftExp(),
      () => new DihedralHelix(), () => new DihedralZero(),
    ];
    for (const make2 of makers) expect(() => fresh(make2).init()).toThrow(StyleError);
  });

  it('wrong coefficient counts throw', () => {
    expect(() => fresh(() => new DihedralOpls()).coeff(['1', '1.0', '2.0'])).toThrow(StyleError);
    expect(() => fresh(() => new DihedralMultiHarmonic()).coeff(['1', '1.0'])).toThrow(StyleError);
    expect(() => fresh(() => new DihedralFourier()).coeff(['1', '2', '1.0', '1', '0.0'])).toThrow(StyleError);
    expect(() => fresh(() => new DihedralFourier()).coeff(['1', '0', '1.0'])).toThrow(StyleError);
    expect(() => fresh(() => new DihedralQuadratic()).coeff(['1', '2.0'])).toThrow(StyleError);
    expect(() => fresh(() => new DihedralNHarmonic()).coeff(['1', '3', '1.0', '2.0'])).toThrow(StyleError);
    expect(() => fresh(() => new DihedralCosineShiftExp()).coeff(['1', '1.0', '2.0'])).toThrow(StyleError);
    expect(() => fresh(() => new DihedralHelix()).coeff(['1', '1.0', '2.0'])).toThrow(StyleError);
  });

  it('zero rejects coefficient values unless nocoeff was set, and unknown keywords', () => {
    const z = fresh(() => new DihedralZero());
    expect(() => z.coeff(['1', '5.0'])).toThrow(StyleError);
    expect(() => z.settings(['bogus'])).toThrow(StyleError);
    z.settings(['nocoeff']);
    expect(() => z.coeff(['1', '5.0', 'junk'])).not.toThrow();
  });
});
