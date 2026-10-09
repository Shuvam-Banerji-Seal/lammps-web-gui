import { describe, expect, it } from 'vitest';
import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms, buildAtomMap, pushTopo } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { newAccum, clearAccum, StyleError, type BondedCompute } from '../src/engine/force/types';
import type { Bonded } from '../src/engine/force/types';
import { AngleClass2 } from '../src/engine/force/angle/class2';
import { ImproperClass2 } from '../src/engine/force/improper/class2';

/*
 * Unit tests for angle_style class2 and improper_style class2: central finite
 * differences must reproduce f = -dE/dx for every atom at several geometries
 * (one crossing a periodic boundary, exercising minimum image), and a style
 * missing one of its coefficient blocks must refuse to init. Native parity is
 * checked by tests/engineOracle.test.ts (w35c2angle_tri/_coeff, w35c2imp).
 */

const GEOMS: number[][][] = [
  [[5.1, 5.2, 4.7], [6.0, 5.1, 5.05], [6.6, 6.1, 4.8], [7.4, 6.3, 5.7]],
  [[5, 5, 5], [6, 5, 5], [6.5, 5.9, 5.1], [7.5, 6.0, 4.0]],
  [[19.6, 5.2, 4.7], [0.5, 5.1, 5.05], [1.1, 6.1, 4.8], [1.9, 6.3, 5.7]],
];

const build = (style: Bonded, pts: number[][], kind: 'angle' | 'improper') => {
  const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [20, 20, 20] }, 1, 'molecular');
  addAtoms(s, Float64Array.from(pts.flat()), 1);
  if (kind === 'angle') pushTopo(s.topo.angles, 1, [1, 2, 3]);
  else pushTopo(s.topo.impropers, 1, [1, 2, 3, 4]);
  const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [20, 20, 20] }));
  const bc: BondedCompute = { s, geom, map: buildAtomMap(s), f: s.f, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null };
  return { s, bc };
};

const energy = (style: Bonded, bc: BondedCompute, kind: 'angle' | 'improper'): number => {
  clearAccum(bc.acc);
  bc.f.fill(0);
  bc.virial.fill(0);
  style.compute(bc);
  return kind === 'angle' ? bc.acc.eangle : bc.acc.eimp;
};

const checkForces = (style: Bonded, pts: number[][], kind: 'angle' | 'improper') => {
  const { s, bc } = build(style, pts, kind);
  energy(style, bc, kind);
  const fAna = Array.from(bc.f);
  const h = 1e-6;
  for (let a = 0; a < pts.length; a++) {
    for (let d = 0; d < 3; d++) {
      const k = 3 * a + d;
      const x0 = s.x[k];
      s.x[k] = x0 + h;
      const ep = energy(style, bc, kind);
      s.x[k] = x0 - h;
      const em = energy(style, bc, kind);
      s.x[k] = x0;
      expect(fAna[k], `atom ${a} dim ${d}`).toBeCloseTo(-(ep - em) / (2 * h), 5);
    }
  }
};

describe('angle_style class2', () => {
  const make = (): AngleClass2 => {
    const a = new AngleClass2();
    a.allocate(1);
    a.coeff(['1', '109.5', '25.0', '3.0', '-0.5']);
    a.coeff(['1', 'bb', '12.0', '1.05', '1.10']);
    a.coeff(['1', 'ba', '3.5', '-2.0', '1.05', '1.10']);
    return a;
  };

  it('f = -dE/dx by central finite differences', () => {
    for (const pts of GEOMS) checkForces(make(), pts, 'angle');
  });

  it('contains the K2/K3/K4 angle, bond-bond and bond-angle terms', () => {
    // A right angle with r_ij and r_jk at their equilibria: only E_a contributes.
    const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [-5, -5, -5], hi: [5, 5, 5] }, 1, 'molecular');
    addAtoms(s, Float64Array.from([1, 0, 0, 0, 0, 0, 0, 1, 0]), 1);
    pushTopo(s.topo.angles, 1, [1, 2, 3]);
    const geom = new Geometry(makeBox({ lo: [-5, -5, -5], hi: [5, 5, 5] }));
    const bc: BondedCompute = { s, geom, map: buildAtomMap(s), f: s.f, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null };
    const a = new AngleClass2();
    a.allocate(1);
    a.coeff(['1', '90.0', '4.0', '0.0', '0.0']);
    a.coeff(['1', 'bb', '7.0', '1.0', '1.0']);
    a.coeff(['1', 'ba', '5.0', '6.0', '1.0', '1.0']);
    a.init();
    clearAccum(bc.acc);
    a.compute(bc);
    expect(bc.acc.eangle).toBeCloseTo(0, 12);
  });

  it('refuses to init without the bb or ba block', () => {
    const a = new AngleClass2();
    a.allocate(1);
    a.coeff(['1', '109.5', '25.0', '3.0', '-0.5']);
    a.coeff(['1', 'bb', '12.0', '1.05', '1.10']);
    expect(() => a.init()).toThrow(StyleError);
    const b = new AngleClass2();
    b.allocate(1);
    b.coeff(['1', '109.5', '25.0', '3.0', '-0.5']);
    b.coeff(['1', 'ba', '3.5', '-2.0', '1.05', '1.10']);
    expect(() => b.init()).toThrow(StyleError);
  });
});

describe('improper_style class2', () => {
  const make = (K: string): ImproperClass2 => {
    const i = new ImproperClass2();
    i.allocate(1);
    i.coeff(['1', K, '15.0']);
    i.coeff(['1', 'aa', '2.0', '1.5', '-1.0', '108.0', '109.5', '110.0']);
    return i;
  };

  it('f = -dE/dx by central finite differences (K nonzero)', () => {
    for (const pts of GEOMS) checkForces(make('12.0'), pts, 'improper');
  });

  it('f = -dE/dx by central finite differences (K = 0)', () => {
    for (const pts of GEOMS) checkForces(make('0.0'), pts, 'improper');
  });

  it('the K = 0 branch ignores chi0 and keeps E_aa', () => {
    const s = emptyState(UNIT_SYSTEMS.lj, 3, { lo: [0, 0, 0], hi: [20, 20, 20] }, 1, 'molecular');
    addAtoms(s, Float64Array.from(GEOMS[1].flat()), 1);
    pushTopo(s.topo.impropers, 1, [1, 2, 3, 4]);
    const geom = new Geometry(makeBox({ lo: [0, 0, 0], hi: [20, 20, 20] }));
    const bc: BondedCompute = { s, geom, map: buildAtomMap(s), f: s.f, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null };
    const i = new ImproperClass2();
    i.allocate(1);
    i.coeff(['1', '0.0', '40.0']);
    i.coeff(['1', 'aa', '2.0', '1.5', '-1.0', '108.0', '109.5', '110.0']);
    i.init();
    clearAccum(bc.acc);
    i.compute(bc);
    expect(Number.isFinite(bc.acc.eimp)).toBe(true);
  });

  it('refuses to init without the aa or plain block', () => {
    const a = new ImproperClass2();
    a.allocate(1);
    a.coeff(['1', '12.0', '15.0']);
    expect(() => a.init()).toThrow(StyleError);
    const b = new ImproperClass2();
    b.allocate(1);
    b.coeff(['1', 'aa', '2.0', '1.5', '-1.0', '108.0', '109.5', '110.0']);
    expect(() => b.init()).toThrow(StyleError);
  });
});
