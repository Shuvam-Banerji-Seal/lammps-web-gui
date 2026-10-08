import { describe, expect, it } from 'vitest';
import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms, buildAtomMap, pushTopo } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { newAccum, clearAccum, StyleError, type BondedCompute } from '../src/engine/force/types';
import type { SimpleBonded } from '../src/engine/force/bonded_util';
import {
  ImproperCvff, ImproperUmbrella, ImproperCossq, ImproperFourier,
  ImproperDistance, ImproperZero,
} from '../src/engine/force/improper/styles';

/*
 * Unit tests for the wave-1 improper styles: central finite differences must
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
  pushTopo(s.topo.impropers, 1, [1, 2, 3, 4]);
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [20, 20, 20] }));
  const bc: BondedCompute = { s, geom, map: buildAtomMap(s), f: s.f, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null };
  return { s, bc };
};

/** Zeroes accumulators, runs compute, returns eimp (forces left in bc.f). */
const energy = (style: SimpleBonded, bc: BondedCompute): number => {
  clearAccum(bc.acc);
  bc.f.fill(0);
  bc.virial.fill(0);
  style.compute(bc);
  return bc.acc.eimp;
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

describe('improper styles: f = -dE/dx by central finite differences', () => {
  it('cvff', () => checkStyle(() => new ImproperCvff(), ['1.2', '-1', '2']));
  it('umbrella', () => checkStyle(() => new ImproperUmbrella(), ['3.0', '25.0']));
  it('umbrella in the omega0 = 0 branch', () => checkStyle(() => new ImproperUmbrella(), ['3.0', '0.0']));
  // with chi0 = 0 native's cossq forces are the energy gradient (see styles.ts)
  it('cossq (chi0 = 0)', () => checkStyle(() => new ImproperCossq(), ['2.0', '0.0']));
  it('cossq (chi0 = 30) reproduces native LAMMPS forces, which are not -dE/dx there', () => {
    const style = new ImproperCossq();
    style.allocate(1);
    style.coeff(['1', '2.0', '30.0']);
    style.init();
    const { bc } = make(style, [[5, 5, 5], [6.1, 5.2, 4.8], [5.3, 6.2, 5.1], [4.8, 5.3, 6.15]]);
    expect(energy(style, bc)).toBeCloseTo(0.00549138452467154, 14);
    // native LAMMPS 2 Sep 2026, improper_coeff 1 2.0 30.0, forces on atoms 1..4
    const native = [
      -0.02675826, 0.06693005, -0.08024039, 0.02675826, -0.06693005, 0.08024039,
      -0.07832329, 0.01696748, -0.02275325, 0.07832329, -0.01696748, 0.02275325,
    ];
    for (let k = 0; k < 12; k++) expect(bc.f[k]).toBeCloseTo(native[k], 7);
  });
  it('fourier', () => checkStyle(() => new ImproperFourier(), ['2.5', '0.3', '0.7', '-0.4']));
  it('fourier with all = 1', () => checkStyle(() => new ImproperFourier(), ['2.5', '0.3', '0.7', '-0.4', '1']));
  it('distance', () => checkStyle(() => new ImproperDistance(), ['1.5', '0.8']));
  it('zero computes no energy and no force', () => {
    const style = new ImproperZero();
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

describe('improper styles: missing or bad coefficients throw', () => {
  const fresh = (make2: () => SimpleBonded) => {
    const style = make2();
    style.allocate(1);
    return style;
  };

  it('init without coefficients throws for every style', () => {
    const makers = [
      () => new ImproperCvff(), () => new ImproperUmbrella(), () => new ImproperCossq(),
      () => new ImproperFourier(), () => new ImproperDistance(), () => new ImproperZero(),
    ];
    for (const make2 of makers) expect(() => fresh(make2).init()).toThrow(StyleError);
  });

  it('wrong coefficient counts throw', () => {
    expect(() => fresh(() => new ImproperCvff()).coeff(['1', '1.0', '2.0'])).toThrow(StyleError);
    expect(() => fresh(() => new ImproperCvff()).coeff(['1', '1.0', '2', '3'])).toThrow(StyleError);
    expect(() => fresh(() => new ImproperUmbrella()).coeff(['1', '1.0'])).toThrow(StyleError);
    expect(() => fresh(() => new ImproperCossq()).coeff(['1', '2.0'])).toThrow(StyleError);
    expect(() => fresh(() => new ImproperFourier()).coeff(['1', '1.0', '2.0', '3.0'])).toThrow(StyleError);
    expect(() => fresh(() => new ImproperFourier()).coeff(['1', '1.0', '2.0', '3.0', '0.5', '1', '1'])).toThrow(StyleError);
    expect(() => fresh(() => new ImproperDistance()).coeff(['1', '1.0'])).toThrow(StyleError);
  });

  it('cvff rejects d other than +/-1 and n outside 0,1,2,3,4,6', () => {
    expect(() => fresh(() => new ImproperCvff()).coeff(['1', '2.0', '2'])).toThrow(StyleError);
    expect(() => fresh(() => new ImproperCvff()).coeff(['1', '-1', '5'])).toThrow(StyleError);
    expect(() => fresh(() => new ImproperCvff()).coeff(['1', '1', '1', '6'])).not.toThrow();
  });

  it('fourier rejects all other than 0 or 1', () => {
    expect(() => fresh(() => new ImproperFourier()).coeff(['1', '1.0', '0.2', '0.3', '0.4', '2'])).toThrow(StyleError);
    expect(() => fresh(() => new ImproperFourier()).coeff(['1', '1.0', '0.2', '0.3', '0.4', '0'])).not.toThrow();
    expect(() => fresh(() => new ImproperFourier()).coeff(['1', '1.0', '0.2', '0.3', '0.4', '1'])).not.toThrow();
  });

  it('zero rejects coefficient values unless nocoeff was set, and unknown keywords', () => {
    const z = fresh(() => new ImproperZero());
    expect(() => z.coeff(['1', '5.0'])).toThrow(StyleError);
    expect(() => z.settings(['bogus'])).toThrow(StyleError);
    z.settings(['nocoeff']);
    expect(() => z.coeff(['1', '5.0', 'junk'])).not.toThrow();
  });
});
