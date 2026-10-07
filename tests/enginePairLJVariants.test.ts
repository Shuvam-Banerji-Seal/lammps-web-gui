import { describe, expect, it } from 'vitest';
import { PairLJ96Cut, PairLJExpand, PairLJSmooth, PairLJSmoothLinear, PairLJVariant } from '../src/engine/force/pair/lj_variants';
import { PAIRS } from '../src/engine/registry/pair_lj';
import type { StyleContext } from '../src/engine/force/types';

/*
 * Unit tests for the LJ variant pair styles: central finite differences
 * proving force = -dE/dr at three geometries per style, the shift offset
 * (energy -> 0 at the cutoff), mixing rules, and StyleError on bad input.
 * The oracle parity itself lives in tests/engineOracle.test.ts (w1ljv_*).
 */

const CTX: StyleContext = { s: null, readFile: () => { throw new Error('no files in unit test'); }, log: () => {} };

const make = (st: PairLJVariant, styleArgs: string[], coeffs: string[][]): PairLJVariant => {
  st.settings(styleArgs, CTX);
  st.allocate(2);
  for (const c of coeffs) st.coeff(c, CTX);
  st.init(CTX);
  return st;
};

/** Central finite difference: |fforce*r - (-dE/dr)| within tolerance. */
const checkFD = (st: PairLJVariant, itype: number, jtype: number, radii: readonly number[]): void => {
  for (const r of radii) {
    const h = 1e-6;
    const ePlus = st.single(0, 0, itype, jtype, (r + h) ** 2, 1, 1).eng;
    const eMinus = st.single(0, 0, itype, jtype, (r - h) ** 2, 1, 1).eng;
    const fd = -(ePlus - eMinus) / (2 * h); // F_r = -dE/dr
    const { fforce } = st.single(0, 0, itype, jtype, r * r, 1, 1);
    expect(Math.abs(fforce * r - fd), `r=${r}`).toBeLessThan(1e-6 * (1 + Math.abs(fd)));
  }
};

const near = (a: number, b: number, tol = 1e-12): void => {
  expect(Math.abs(a - b)).toBeLessThanOrEqual(tol);
};

const nearRel = (a: number, b: number, rel: number): void => {
  expect(Math.abs(a - b)).toBeLessThanOrEqual(rel * Math.max(Math.abs(a), Math.abs(b)));
};

describe('pair style lj96/cut', () => {
  it('force = -dE/dr at three geometries (both types)', () => {
    const st = make(new PairLJ96Cut(), ['2.5'], [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05']]);
    checkFD(st, 1, 1, [1.1, 1.6, 2.2]);
    checkFD(st, 2, 2, [1.15, 1.8, 2.3]);
  });

  it('shift yes: energy is 0 at the cutoff, force unchanged', () => {
    const st = make(new PairLJ96Cut(), ['2.5'], [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05']]);
    st.shift = true;
    st.init(CTX);
    near(st.single(0, 0, 1, 1, 2.5 * 2.5, 1, 1).eng, 0);
    checkFD(st, 1, 1, [1.1, 1.6, 2.2]);
  });

  it('tail correction matches numeric integration of the Sun formula', () => {
    const st = new PairLJ96Cut();
    st.settings(['2.5'], CTX);
    st.allocate(1);
    st.coeff(['1', '1', '1.0', '1.0'], CTX);
    st.tail = true;
    st.init(CTX);
    const t = st.tailSums(new Float64Array([0, 108]));
    // etail = 2 pi N^2 int_rc^inf u(r) r^2 dr ; ptail = -2 pi/3 N^2 int r^3 u'(r) dr
    const u = (r: number) => 4 * (r ** -9 - r ** -6);
    const du = (r: number) => 4 * (-9 * r ** -10 + 6 * r ** -7);
    let e = 0, p = 0;
    const dr = 1e-4;
    for (let r = 2.5 + dr / 2; r < 120; r += dr) {
      e += u(r) * r * r * dr;
      p += du(r) * r ** 3 * dr;
    }
    nearRel(t.etail, 2 * Math.PI * 108 * 108 * e, 1e-4);
    nearRel(t.ptail, -2 * Math.PI / 3 * 108 * 108 * p, 1e-4);
  });

  it('mixes epsilon, sigma and cutoff geometrically', () => {
    const st = make(new PairLJ96Cut(), ['2.5'], [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05']]);
    near(st.p.get('epsilon', 1, 2), Math.sqrt(0.8));
    near(st.p.get('sigma', 1, 2), Math.sqrt(1.05));
    near(st.p.get('cut', 1, 2), 2.5);
  });

  it('throws on missing cross coefficients', () => {
    const st = new PairLJ96Cut();
    st.settings(['2.5'], CTX);
    st.allocate(2);
    st.coeff(['1', '1', '1.0', '1.0'], CTX);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
  });

  it('throws on wrong pair_coeff arity', () => {
    const st = new PairLJ96Cut();
    st.allocate(2);
    expect(() => st.coeff(['1', '1', '1.0'], CTX)).toThrow(/usage/);
  });
});

describe('pair style lj/expand', () => {
  it('force = -dE/dr at three geometries (three type pairs)', () => {
    const st = make(new PairLJExpand(), ['2.5'], [
      ['1', '1', '1.0', '1.0', '0.0'], ['2', '2', '0.9', '0.9', '0.15'], ['1', '2', '1.0', '0.95', '0.08'],
    ]);
    checkFD(st, 1, 1, [1.2, 1.9, 2.4]);
    checkFD(st, 2, 2, [1.3, 2.0, 2.5]);
    checkFD(st, 1, 2, [1.25, 2.1, 2.5]);
  });

  it('force cutoff is rc + delta and shift zeroes the energy there', () => {
    const st = make(new PairLJExpand(), ['2.5'], [
      ['1', '1', '1.0', '1.0', '0.0'], ['2', '2', '0.9', '0.9', '0.15'], ['1', '2', '1.0', '0.95', '0.08'],
    ]);
    near(st.cut[1 * 3 + 2], 2.5 + 0.08);
    st.shift = true;
    st.init(CTX);
    const rc = 2.5 + 0.15;
    near(st.single(0, 0, 2, 2, rc * rc, 1, 1).eng, 0);
    checkFD(st, 2, 2, [1.3, 2.0, 2.5]);
  });

  it('mixes delta arithmetically, epsilon/sigma geometrically', () => {
    const st = make(new PairLJExpand(), ['2.5'], [['1', '1', '1.0', '1.0', '0.0'], ['2', '2', '0.81', '1.0', '0.2']]);
    near(st.p.get('delta', 1, 2), 0.1);
    near(st.p.get('epsilon', 1, 2), 0.9);
    near(st.p.get('sigma', 1, 2), 1.0);
    near(st.cut[1 * 3 + 2], 2.5 + 0.1);
  });

  it('throws on missing cross coefficients', () => {
    const st = new PairLJExpand();
    st.settings(['2.5'], CTX);
    st.allocate(2);
    st.coeff(['1', '1', '1.0', '1.0', '0.0'], CTX);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
  });
});

describe('pair style lj/smooth', () => {
  it('force = -dE/dr across both branches at three geometries', () => {
    const st = make(new PairLJSmooth(), ['2.0', '2.5'], [
      ['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05'], ['1', '2', '0.9', '1.0', '1.8', '2.3'],
    ]);
    checkFD(st, 1, 1, [1.3, 1.95, 2.15]); // LJ branch, LJ branch, smooth branch
    checkFD(st, 1, 2, [1.4, 1.75, 2.25]);
    checkFD(st, 2, 2, [1.5, 2.2, 2.38]);  // mixed rin/rc
  });

  it('force goes to 0 at the outer cutoff; shift zeroes the energy there', () => {
    const st = make(new PairLJSmooth(), ['2.0', '2.5'], [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05']]);
    near(st.single(0, 0, 1, 1, 2.5 * 2.5, 1, 1).fforce * 2.5, 0, 1e-12);
    st.shift = true;
    st.init(CTX);
    near(st.single(0, 0, 1, 1, 2.5 * 2.5, 1, 1).eng, 0);
    checkFD(st, 1, 1, [1.3, 1.95, 2.15]);
  });

  it('mixes rin and rc like the cutoff (geometric)', () => {
    const st = make(new PairLJSmooth(), ['2.0', '2.5'], [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05', '1.8', '2.3']]);
    near(st.p.get('rin', 1, 2), Math.sqrt(2.0 * 1.8), 1e-12);
    near(st.p.get('cut', 1, 2), Math.sqrt(2.5 * 2.3));
  });

  it('throws on missing cross coefficients', () => {
    const st = new PairLJSmooth();
    st.settings(['2.0', '2.5'], CTX);
    st.allocate(2);
    st.coeff(['1', '1', '1.0', '1.0'], CTX);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
  });

  it('throws on rin = 0 and on rin >= rc', () => {
    const st = new PairLJSmooth();
    st.allocate(2);
    expect(() => st.settings(['0.0', '2.5'], CTX)).toThrow(/inner cutoff cannot be 0.0/);
    expect(() => st.settings(['2.5', '2.5'], CTX)).toThrow(/must be less than/);
    void st;
  });

  it('throws on pair_modify tail yes (unsupported)', () => {
    const st = new PairLJSmooth();
    st.settings(['2.0', '2.5'], CTX);
    st.allocate(2);
    st.coeff(['1', '1', '1.0', '1.0'], CTX);
    st.coeff(['2', '2', '1.0', '1.0'], CTX);
    st.tail = true;
    expect(() => st.init(CTX)).toThrow(/tail/);
  });
});

describe('pair style lj/smooth/linear', () => {
  it('force = -dE/dr at three geometries (three type pairs)', () => {
    const st = make(new PairLJSmoothLinear(), ['2.5'], [
      ['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05'], ['1', '2', '0.9', '1.0', '2.2'],
    ]);
    checkFD(st, 1, 1, [1.1, 1.7, 2.3]);
    checkFD(st, 2, 2, [1.2, 2.0, 2.45]);
    checkFD(st, 1, 2, [1.3, 1.9, 2.19]);
  });

  it('energy and force go to 0 at the cutoff', () => {
    const st = make(new PairLJSmoothLinear(), ['2.5'], [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05']]);
    const at = st.single(0, 0, 1, 1, 2.5 * 2.5, 1, 1);
    near(at.eng, 0);
    near(at.fforce, 0);
  });

  it('throws on pair_modify shift yes (unsupported)', () => {
    const st = new PairLJSmoothLinear();
    st.settings(['2.5'], CTX);
    st.allocate(2);
    st.coeff(['1', '1', '1.0', '1.0'], CTX);
    st.coeff(['2', '2', '1.0', '1.0'], CTX);
    st.shift = true;
    expect(() => st.init(CTX)).toThrow(/shift/);
  });

  it('throws on missing cross coefficients', () => {
    const st = new PairLJSmoothLinear();
    st.settings(['2.5'], CTX);
    st.allocate(2);
    st.coeff(['1', '1', '1.0', '1.0'], CTX);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
  });
});

describe('registry', () => {
  it('registers the four styles under their LAMMPS names', () => {
    for (const n of ['lj96/cut', 'lj/expand', 'lj/smooth', 'lj/smooth/linear']) {
      expect(PAIRS[n], n).toBeTypeOf('function');
      expect(PAIRS[n]!().name).toBe(n);
    }
  });
});
