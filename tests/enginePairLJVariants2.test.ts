import { describe, expect, it } from 'vitest';
import { PairLJClass2, PairLJCubic, PairLJGromacs, PairMieCut, PairLJ2Variant } from '../src/engine/force/pair/lj_variants2';
import { PAIRS } from '../src/engine/registry/pair_lj2';
import type { StyleContext } from '../src/engine/force/types';

/*
 * Unit tests for the wave-2 LJ variant pair styles: central finite differences
 * proving force = -dE/dr at three geometries per style (covering every branch
 * of each potential), the shift offset (energy -> 0 at the cutoff), the
 * documented mixing rules, tail corrections against numeric integration, and
 * StyleError on missing coefficients and unsupported options. The oracle
 * parity itself lives in tests/engineOracle.test.ts (w1ljx_*).
 */

const CTX: StyleContext = { s: null, readFile: () => { throw new Error('no files in unit test'); }, log: () => {} };

const make = (st: PairLJ2Variant, settingsArgs: readonly string[] | null, coeffs: string[][]): PairLJ2Variant => {
  st.allocate(2);
  if (settingsArgs) st.settings([...settingsArgs], CTX);
  for (const c of coeffs) st.coeff(c, CTX);
  st.init(CTX);
  return st;
};

/** Central finite difference: |fforce*r - (-dE/dr)| within tolerance. */
const checkFD = (st: PairLJ2Variant, itype: number, jtype: number, radii: readonly number[]): void => {
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

describe('pair style lj/class2', () => {
  it('force = -dE/dr at three geometries (both types)', () => {
    const st = make(new PairLJClass2(), ['2.5'], [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05']]);
    checkFD(st, 1, 1, [1.1, 1.6, 2.2]);
    checkFD(st, 2, 2, [1.15, 1.8, 2.3]);
  });

  it('shift yes: energy is 0 at the cutoff, force unchanged', () => {
    const st = make(new PairLJClass2(), ['2.5'], [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05']]);
    st.shift = true;
    st.init(CTX);
    near(st.single(0, 0, 1, 1, 2.5 * 2.5, 1, 1).eng, 0);
    checkFD(st, 1, 1, [1.1, 1.6, 2.2]);
  });

  it('epsilon and sigma always mix sixthpower; cutoff follows pair_modify mix', () => {
    const st = make(new PairLJClass2(), ['2.5'], [['1', '1', '1.0', '1.0', '2.4'], ['2', '2', '0.8', '1.05', '2.6']]);
    const sixthEps = (2 * Math.sqrt(1.0 * 0.8) * 1.0 ** 3 * 1.05 ** 3) / (1.0 ** 6 + 1.05 ** 6);
    const sixthSig = (0.5 * (1.0 ** 6 + 1.05 ** 6)) ** (1 / 6);
    near(st.p.get('epsilon', 1, 2), sixthEps);
    near(st.p.get('sigma', 1, 2), sixthSig);
    near(st.p.get('cut', 1, 2), Math.sqrt(2.4 * 2.6));
    st.mix = 'sixthpower';
    st.init(CTX);
    near(st.p.get('epsilon', 1, 2), sixthEps);
    near(st.p.get('sigma', 1, 2), sixthSig);
    near(st.p.get('cut', 1, 2), (0.5 * (2.4 ** 6 + 2.6 ** 6)) ** (1 / 6));
  });

  it('tail correction matches numeric integration of the Sun formula', () => {
    const st = new PairLJClass2();
    st.allocate(1);
    st.settings(['2.5']);
    st.coeff(['1', '1', '1.0', '1.0']);
    st.tail = true;
    st.init(CTX);
    const t = st.tailSums(new Float64Array([0, 108]));
    // etail = 2 pi N^2 int_rc^inf u(r) r^2 dr ; ptail = -2 pi/3 N^2 int r^3 u'(r) dr
    const u = (r: number) => 2 * r ** -9 - 3 * r ** -6;
    const du = (r: number) => -18 * r ** -10 + 18 * r ** -7;
    let e = 0, p = 0;
    const dr = 1e-4;
    for (let r = 2.5 + dr / 2; r < 120; r += dr) {
      e += u(r) * r * r * dr;
      p += du(r) * r ** 3 * dr;
    }
    nearRel(t.etail, 2 * Math.PI * 108 * 108 * e, 1e-4);
    nearRel(t.ptail, -2 * Math.PI / 3 * 108 * 108 * p, 1e-4);
  });

  it('throws on missing cross coefficients', () => {
    const st = new PairLJClass2();
    st.allocate(2);
    st.settings(['2.5']);
    st.coeff(['1', '1', '1.0', '1.0']);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
  });

  it('throws on wrong pair_coeff arity and on two cutoffs', () => {
    const st = new PairLJClass2();
    st.allocate(2);
    expect(() => st.coeff(['1', '1', '1.0'])).toThrow(/usage/);
    expect(() => st.coeff(['1', '1', '1.0', '1.0', '2.5', '2.5'])).toThrow(/2 cutoffs/);
  });
});

describe('pair style lj/cubic', () => {
  const rhoS = (26 / 7) ** (1 / 6);
  const rhoC = (67 / 48) * rhoS;

  it('force = -dE/dr across both branches at three geometries', () => {
    const st = make(new PairLJCubic(), null, [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05'], ['1', '2', '0.9', '1.02']]);
    // sigma = 1: r_s = 1.2445 (LJ branch below, cubic above); r_c = 1.737
    checkFD(st, 1, 1, [1.0, 1.45, 1.7]);
    // sigma = 1.05: r_s = 1.3067, r_c = 1.8239
    checkFD(st, 2, 2, [1.1, 1.5, 1.8]);
    // mixed sigma = sqrt(1.02): r_s = 1.2577, r_c = 1.7590
    checkFD(st, 1, 2, [1.05, 1.45, 1.75]);
  });

  it('cutoff is r_s*67/48 and energy and force are 0 there', () => {
    const st = make(new PairLJCubic(), null, [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05']]);
    near(st.cut[1 * 3 + 1], rhoC * 1.0, 1e-14);
    near(st.cut[2 * 3 + 2], rhoC * 1.05, 1e-14);
    for (const [i, sig] of [[1, 1.0], [2, 1.05]] as const) {
      const at = st.single(0, 0, i, i, (rhoC * sig) ** 2, 1, 1);
      near(at.eng, 0, 1e-12);
      near(at.fforce, 0, 1e-12);
    }
  });

  it('mixes epsilon and sigma geometrically', () => {
    const st = make(new PairLJCubic(), null, [['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05']]);
    near(st.p.get('epsilon', 1, 2), Math.sqrt(0.8));
    near(st.p.get('sigma', 1, 2), Math.sqrt(1.05));
  });

  it('throws on missing cross coefficients', () => {
    const st = new PairLJCubic();
    st.allocate(2);
    st.coeff(['1', '1', '1.0', '1.0']);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
  });

  it('throws on pair_style args, extra pair_coeff cutoff, shift and tail', () => {
    const st = new PairLJCubic();
    st.allocate(2);
    expect(() => st.settings(['2.5'])).toThrow(/usage/);
    expect(() => st.coeff(['1', '1', '1.0', '1.0', '2.5'])).toThrow(/usage/);
    const full = make(new PairLJCubic(), null, [['1', '1', '1.0', '1.0'], ['2', '2', '1.0', '1.0']]);
    full.shift = true;
    expect(() => full.init(CTX)).toThrow(/shift/);
    full.shift = false;
    full.tail = true;
    expect(() => full.init(CTX)).toThrow(/tail/);
  });
});

describe('pair style lj/gromacs', () => {
  it('force = -dE/dr across inner, switching and outer regions at three geometries', () => {
    const st = make(new PairLJGromacs(), ['2.0', '2.5'], [
      ['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05'], ['1', '2', '0.9', '1.0', '1.9', '2.4'],
    ]);
    checkFD(st, 1, 1, [1.0, 2.2, 2.45]);
    checkFD(st, 2, 2, [1.05, 2.1, 2.45]);
    checkFD(st, 1, 2, [0.9, 2.1, 2.35]); // per-pair inner/outer 1.9 / 2.4
  });

  it('energy and force go to 0 at the outer cutoff', () => {
    const st = make(new PairLJGromacs(), ['2.0', '2.5'], [
      ['1', '1', '1.0', '1.0'], ['2', '2', '0.8', '1.05'], ['1', '2', '0.9', '1.0', '1.9', '2.4'],
    ]);
    for (const [i, j, rc] of [[1, 1, 2.5], [2, 2, 2.5], [1, 2, 2.4]] as const) {
      const at = st.single(0, 0, i, j, rc * rc, 1, 1);
      near(at.eng, 0, 1e-12);
      near(at.fforce, 0, 1e-12);
    }
  });

  it('mixes inner and outer cutoffs geometrically', () => {
    const st = make(new PairLJGromacs(), ['2.0', '2.5'], [
      ['1', '1', '1.0', '1.0', '1.8', '2.3'], ['2', '2', '0.8', '1.05', '2.2', '2.7'],
    ]);
    near(st.p.get('epsilon', 1, 2), Math.sqrt(0.8));
    near(st.p.get('inner', 1, 2), Math.sqrt(1.8 * 2.2));
    near(st.p.get('outer', 1, 2), Math.sqrt(2.3 * 2.7));
  });

  it('throws on missing cross coefficients', () => {
    const st = new PairLJGromacs();
    st.allocate(2);
    st.settings(['2.0', '2.5']);
    st.coeff(['1', '1', '1.0', '1.0']);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
  });

  it('throws on bad settings and pair_coeff arity', () => {
    const st = new PairLJGromacs();
    st.allocate(2);
    expect(() => st.settings(['0.0', '2.5'])).toThrow(/inner cutoff must be > 0/);
    expect(() => st.settings(['2.5', '2.5'])).toThrow(/must be less than/);
    expect(() => st.settings(['2.0'])).toThrow(/usage/);
    expect(() => st.coeff(['1', '1', '1.0', '1.0', '2.0'])).toThrow(/usage/);
  });

  it('throws on pair_modify shift yes and tail yes (unsupported)', () => {
    const st = make(new PairLJGromacs(), ['2.0', '2.5'], [['1', '1', '1.0', '1.0'], ['2', '2', '1.0', '1.0']]);
    st.shift = true;
    expect(() => st.init(CTX)).toThrow(/shift/);
    st.shift = false;
    st.tail = true;
    expect(() => st.init(CTX)).toThrow(/tail/);
  });
});

describe('pair style mie/cut', () => {
  it('force = -dE/dr at three geometries (three type pairs)', () => {
    const st = make(new PairMieCut(), ['2.5'], [
      ['1', '1', '1.0', '1.0', '12.0', '6.0'], ['2', '2', '0.8', '1.05', '14.0', '7.0'], ['1', '2', '0.9', '1.02', '13.0', '6.5'],
    ]);
    checkFD(st, 1, 1, [1.0, 1.6, 2.3]);
    checkFD(st, 2, 2, [1.05, 1.7, 2.3]);
    checkFD(st, 1, 2, [1.0, 1.6, 2.3]);
  });

  it('12/6 exponents give C = 4 and the standard LJ value', () => {
    const st = new PairMieCut();
    st.allocate(1);
    st.settings(['2.5']);
    st.coeff(['1', '1', '1.0', '1.0', '12.0', '6.0']);
    st.init(CTX);
    near(st.single(0, 0, 1, 1, 1.3 * 1.3, 1, 1).eng, 4 * (1.3 ** -12 - 1.3 ** -6));
  });

  it('shift yes: energy is 0 at the cutoff, force unchanged', () => {
    const st = make(new PairMieCut(), ['2.5'], [['1', '1', '1.0', '1.0', '12.0', '6.0'], ['2', '2', '0.8', '1.05', '14.0', '7.0']]);
    st.shift = true;
    st.init(CTX);
    near(st.single(0, 0, 1, 1, 2.5 * 2.5, 1, 1).eng, 0);
    checkFD(st, 1, 1, [1.0, 1.6, 2.3]);
  });

  it('tail correction matches numeric integration of the Sun formula', () => {
    const st = new PairMieCut();
    st.allocate(1);
    st.settings(['2.5']);
    st.coeff(['1', '1', '1.0', '1.0', '12.0', '6.0']);
    st.tail = true;
    st.init(CTX);
    const t = st.tailSums(new Float64Array([0, 108]));
    const u = (r: number) => 4 * (r ** -12 - r ** -6);
    const du = (r: number) => 4 * (-12 * r ** -13 + 6 * r ** -7);
    let e = 0, p = 0;
    const dr = 1e-4;
    for (let r = 2.5 + dr / 2; r < 120; r += dr) {
      e += u(r) * r * r * dr;
      p += du(r) * r ** 3 * dr;
    }
    nearRel(t.etail, 2 * Math.PI * 108 * 108 * e, 1e-4);
    nearRel(t.ptail, -2 * Math.PI / 3 * 108 * 108 * p, 1e-4);
  });

  it('mixes epsilon, sigma and both gammas geometrically', () => {
    const st = make(new PairMieCut(), ['2.5'], [['1', '1', '1.0', '1.0', '12.0', '6.0'], ['2', '2', '0.8', '1.05', '14.0', '7.0']]);
    near(st.p.get('epsilon', 1, 2), Math.sqrt(0.8));
    near(st.p.get('sigma', 1, 2), Math.sqrt(1.05));
    near(st.p.get('gammaR', 1, 2), Math.sqrt(12 * 14));
    near(st.p.get('gammaA', 1, 2), Math.sqrt(6 * 7));
  });

  it('throws on missing cross coefficients', () => {
    const st = new PairMieCut();
    st.allocate(2);
    st.settings(['2.5']);
    st.coeff(['1', '1', '1.0', '1.0', '12.0', '6.0']);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
  });

  it('throws on gammaR <= gammaA and wrong pair_coeff arity', () => {
    const st = new PairMieCut();
    st.allocate(1);
    st.settings(['2.5']);
    expect(() => st.coeff(['1', '1', '1.0', '1.0', '6.0', '12.0'])).not.toThrow();
    expect(() => st.init(CTX)).toThrow(/gammaR > gammaA/);
    const st2 = new PairMieCut();
    st2.allocate(2);
    expect(() => st2.coeff(['1', '1', '1.0', '1.0', '12.0'])).toThrow(/usage/);
  });
});

describe('registry', () => {
  it('registers the four styles under their LAMMPS names', () => {
    for (const n of ['lj/gromacs', 'lj/class2', 'lj/cubic', 'mie/cut']) {
      expect(PAIRS[n], n).toBeTypeOf('function');
      expect(PAIRS[n]!().name).toBe(n);
    }
  });
});
