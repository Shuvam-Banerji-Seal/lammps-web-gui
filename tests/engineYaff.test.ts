import { describe, expect, it } from 'vitest';
import { Geometry, makeBox } from '../src/engine/domain';
import { emptyState, addAtoms, buildAtomMap, pushTopo } from '../src/engine/atoms';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { newAccum, clearAccum, StyleError, type BondedCompute } from '../src/engine/force/types';
import { BONDS, ANGLES, PAIRS } from '../src/engine-gpl/registry/yaff';
import { BondMM3 } from '../src/engine-gpl/yaff/bond_mm3';
import { AngleCross } from '../src/engine-gpl/yaff/angle_cross';
import { PairLJSwitch3CoulGaussLong } from '../src/engine-gpl/yaff/pair_lj_switch3_coulgauss_long';

/*
 * Unit tests for the YAFF styles ported from LAMMPS (src/engine-gpl/yaff). Native parity is in the oracle cases
 * tests/oracle/w118yaff_*.in; these tests cover the registry names, the closed-form energies on simple geometries,
 * and the refusals (pairwise energies, pair_modify tail).
 */

const box: { lo: [number, number, number]; hi: [number, number, number] } = { lo: [0, 0, 0], hi: [20, 20, 20] };

/** A system of three atoms (1 2 3) with one angle 1-2-3 (vertex 2) and, optionally, one bond 1-2. */
const angleSystem = (pts: number[][], bond = false) => {
  const s = emptyState(UNIT_SYSTEMS.real, 3, box, 1, 'molecular');
  addAtoms(s, Float64Array.from(pts.flat()), 1);
  pushTopo(s.topo.angles, 1, [1, 2, 3]);
  if (bond) pushTopo(s.topo.bonds, 1, [1, 2]);
  const geom = new Geometry(makeBox(box));
  const bc: BondedCompute = { s, geom, map: buildAtomMap(s), f: s.f, acc: newAccum(), virial: new Float64Array(6), eatom: null, vatom: null };
  return { s, bc };
};

describe('YAFF style registry (src/engine-gpl/registry/yaff.ts)', () => {
  it('registers the names listed in lazyStyles.ts', () => {
    expect(Object.keys(BONDS)).toEqual(['mm3']);
    expect(Object.keys(ANGLES)).toEqual(['cross']);
    expect(Object.keys(PAIRS)).toEqual(['lj/switch3/coulgauss/long']);
  });
});

describe('bond_style mm3 (docs.lammps.org/bond_mm3.html)', () => {
  it('energy is K d^2 (1 - 2.55 d + 7/12 2.55^2 d^2), d = r - r0, in real units', () => {
    const style = new BondMM3();
    style.allocate(1);
    style.coeff(['1', '100', '1.0']);
    const { s, bc } = angleSystem([[0, 0, 0], [1.1, 0, 0], [5, 5, 5]], true);
    const ctx = { s, readFile: () => '', log: () => {} };
    style.init(ctx as never);
    clearAccum(bc.acc);
    bc.f.fill(0);
    style.compute(bc);
    const d = 0.1;
    const expected = 100 * d * d * (1 - 2.55 * d + (7 / 12) * 2.55 * 2.55 * d * d);
    expect(bc.acc.ebond).toBeCloseTo(expected, 12);
  });
});

describe('angle_style cross (docs.lammps.org/angle_cross.html)', () => {
  it('energy is Kss dr1 dr2 + Kbs0 dr1 dtheta + Kbs1 dr2 dtheta', () => {
    const style = new AngleCross();
    style.allocate(1);
    // theta0 = 80 degrees; the right-angle geometry gives theta = 90 degrees, so dtheta = 10 degrees
    style.coeff(['1', '50', '100', '30', '1.0', '1.0', '80']);
    // r1 = 1.2 (bond 1-2), r2 = 1.0 (bond 3-2), vertex atom 2 at the origin
    const { bc } = angleSystem([[1.2, 0, 0], [0, 0, 0], [0, 1.0, 0]]);
    style.init({ s: bc.s, readFile: () => '', log: () => {} } as never);
    clearAccum(bc.acc);
    bc.f.fill(0);
    style.compute(bc);
    const dr1 = 0.2, dr2 = 0, dth = (10 * Math.PI) / 180;
    expect(bc.acc.eangle).toBeCloseTo(50 * dr1 * dr2 + 100 * dr1 * dth + 30 * dr2 * dth, 12);
    // forces sum to zero (internal term)
    for (let d = 0; d < 3; d++) expect(bc.f[d] + bc.f[3 + d] + bc.f[6 + d]).toBeCloseTo(0, 12);
  });
});

describe('pair_style lj/switch3/coulgauss/long (docs.lammps.org/pair_lj_switch3_coulgauss_long.html)', () => {
  it('reads "cutoff width" and "cutoff cutoff2 width"', () => {
    const p = new PairLJSwitch3CoulGaussLong();
    p.settings(['12.0', '4.0'] as never);
    expect(p.cutGlobal).toBe(12);
    expect(p.cutCoul).toBe(12);
    expect(p.truncw).toBe(4);
    p.settings(['12.0', '10.0', '3.0'] as never);
    expect(p.cutCoul).toBe(10);
    expect(p.truncw).toBe(3);
  });

  it('refuses malformed settings, pairwise energies and pair_modify tail', () => {
    const p = new PairLJSwitch3CoulGaussLong();
    expect(() => p.settings(['12.0'] as never)).toThrow(StyleError);
    p.settings(['12.0', '4.0'] as never);
    p.allocate(1);
    expect(() => p.coeff(['1', '1', '0.1', '1.0'])).toThrow(StyleError);
    p.coeff(['1', '1', '0.1', '1.0', '1.0']);
    expect(() => p.single()).toThrow(StyleError);
    p.tail = true;
    expect(() => p.initOne(1, 1)).toThrow(StyleError);
  });
});
