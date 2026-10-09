import { describe, expect, it } from 'vitest';
import { DihedralClass2 } from '../src/engine/force/dihedral/class2';
import { newAccum, StyleError, type BondedCompute } from '../src/engine/force/types';

/*
 * Focused tests for dihedral_style class2: the documented energy formulas, the
 * analytic forces (checked against finite differences of the same energy) and
 * the six-set coefficient requirements. Native parity is covered separately by
 * the w35c2dih* oracle cases in engineOracle.test.ts.
 */

/** Type 1 coefficients of the w35c2dih oracle data (docs.lammps.org/dihedral_class2.html example shape). */
const T1: string[][] = [
  ['1', '100.0', '75.0', '70.0', '100.0', '80.0', '60.0'],
  ['1', 'mbt', '3.5', '0.17', '-0.55', '1.02'],
  ['1', 'ebt', '0.34', '0.33', '-0.90', '0.14', '0.0', '-0.81', '0.95', '1.03'],
  ['1', 'at', '0.0', '-0.185', '-0.796', '-2.02', '0.0', '-0.399', '110.2', '105.1'],
  ['1', 'aat', '-13.5', '110.2', '105.1'],
  ['1', 'bb13', '0.5', '0.95', '1.03'],
];

const ZERO: Record<string, string> = { mbt: '0 0 0 1', ebt: '0 0 0 0 0 0 1 1', at: '0 0 0 0 0 0 90 90', aat: '0 90 90', bb13: '0 1 1' };
const T1CROSS: Record<string, string> = {
  mbt: '3.5 0.17 -0.55 1.02',
  ebt: '0.34 0.33 -0.90 0.14 0.0 -0.81 0.95 1.03',
  at: '0.0 -0.185 -0.796 -2.02 0.0 -0.399 110.2 105.1',
  aat: '-13.5 110.2 105.1',
  bb13: '0.5 0.95 1.03',
};

const style = (main: string, cross: Record<string, string>, ntypes = 1): DihedralClass2 => {
  const all = { ...ZERO, ...cross };
  const d = new DihedralClass2();
  d.allocate(ntypes);
  d.coeff(['1', ...main.split(' ')]);
  for (const kw of ['mbt', 'ebt', 'at', 'aat', 'bb13']) d.coeff(['1', kw, ...all[kw].split(' ')]);
  return d;
};

const bcFor = (x: number[], extra = 0): BondedCompute => {
  const acc = newAccum();
  const n = x.length / 3;
  const map = new Int32Array(n + 1);
  for (let i = 1; i <= n; i++) map[i] = i - 1;
  return {
    s: { x: Float64Array.from(x), topo: { dihedrals: { n: 1, atoms: new Int32Array([1, 2, 3, 4]), type: new Int32Array([1]) } } } as unknown as BondedCompute['s'],
    geom: { minimumImage: () => {} } as unknown as BondedCompute['geom'],
    map, f: new Float64Array(3 * n + 3 * extra), acc, virial: acc.vdihed, eatom: null, vatom: null,
  };
};

const energy = (d: DihedralClass2, x: number[]): number => {
  const bc = bcFor(x);
  d.compute(bc);
  return bc.acc.edihed;
};

// non-planar 4-atom chains: general (phi = 90), near-trans (phi ~ 172) and near-cis (phi ~ 8)
const GENERAL = [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1];
const NEAR180 = [-0.5, 0.866, 0, 0, 0, 0, 1, 0, 0, 1.5, -0.866, 0.12];
const NEAR0 = [-0.5, 0.866, 0, 0, 0, 0, 1, 0, 0, 1.5, 0.866, 0.12];

describe('dihedral_style class2', () => {
  it('reproduces the documented energy terms', () => {
    // E_d only: K1 = 4, phi1 = 0, phi = 90 -> 4 (1 - cos 90) = 4
    expect(energy(style('4 0 0 0 0 0', ZERO), GENERAL)).toBeCloseTo(4, 12);
    // phi1 = 90 -> 4 (1 - cos 0) = 0
    expect(energy(style('4 90 0 0 0 0', ZERO), GENERAL)).toBeCloseTo(0, 12);
    // E_mbt only: (r_jk - 1.1)[cos 90 + cos 180 + cos 270] = (-0.1)(-1) = 0.1
    expect(energy(style('0 0 0 0 0 0', { ...ZERO, mbt: '1 1 1 1.1' }), GENERAL)).toBeCloseTo(0.1, 12);
    // E_bb13 only: 1 (1 - 1.1)(1 - 1.2) = 0.02
    expect(energy(style('0 0 0 0 0 0', { ...ZERO, bb13: '1 1.1 1.2' }), GENERAL)).toBeCloseTo(0.02, 12);
    // E_at only, theta in degrees: (90-100deg)[cos 90+cos 180+cos 270] + (90-110deg)[same] = 30deg
    const at = energy(style('0 0 0 0 0 0', { at: '1 1 1 1 1 1 100 110' }), GENERAL);
    expect(at).toBeCloseTo((30 * Math.PI) / 180, 12);
    // all six terms together (hand-evaluated from the doc formulas)
    expect(energy(style('100 75 70 100 80 60', T1CROSS), GENERAL)).toBeCloseTo(210.45500428301608, 9);
  });

  it('rejects a type whose coefficients are incomplete', () => {
    const d = new DihedralClass2();
    d.allocate(1);
    for (const c of T1.slice(0, 5)) d.coeff(c); // no bb13 line
    expect(() => d.init()).toThrow(StyleError);
    expect(() => d.init()).toThrow(/not set/);
  });

  it('rejects an unknown cross-term keyword and a wrong coefficient count', () => {
    const d = style('0 0 0 0 0 0', ZERO);
    expect(() => d.coeff(['1', 'foo', '1', '2', '3'])).toThrow(/unknown dihedral_coeff class2 keyword 'foo'/);
    expect(() => d.coeff(['1', 'mbt', '1', '2'])).toThrow(/mbt needs 4 coefficients/);
    expect(() => d.coeff(['1', '1', '2'])).toThrow(/K1 phi1 K2 phi2 K3 phi3/);
  });

  it('write_data prints the main line with phi in degrees', () => {
    expect(style('100 75 70 100 80 60', {}).dataCoeffs()).toEqual(['1 100 75 70 100 80 60']);
  });

  it('analytic forces match finite differences of the energy', () => {
    const d = style('100 75 70 100 80 60', T1CROSS);
    for (const x0 of [GENERAL, NEAR180, NEAR0]) {
      const bc = bcFor(x0);
      d.compute(bc);
      const f = Array.from(bc.f);
      // internal forces sum to zero
      for (let c = 0; c < 3; c++) expect(Math.abs(f[c] + f[3 + c] + f[6 + c] + f[9 + c])).toBeLessThan(1e-9);
      const h = 1e-6;
      for (let k = 0; k < 12; k++) {
        const xp = x0.slice(), xm = x0.slice();
        xp[k] += h; xm[k] -= h;
        const num = -(energy(d, xp) - energy(d, xm)) / (2 * h);
        expect(Math.abs(num - f[k])).toBeLessThan(1e-5 * Math.max(1, Math.abs(f[k])));
      }
    }
  });
});
