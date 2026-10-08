import { describe, expect, it } from 'vitest';
import { PAIR_STYLES } from '../src/engine/styles';
import {
  PairBornCoulLongCS, PairBuckCoulLongCS, PairCoulLongCS, csCoulPair, erfAcc, hOverX3,
} from '../src/engine/force/pair/coul_cs';
import { PairBornCoulLong } from '../src/engine/force/pair/coul_long2';
import { erfcExact } from '../src/engine/force/erfc';
import type { Pair, StyleContext } from '../src/engine/force/types';

/*
 * Unit checks for pair_style lj/sf (an alias of lj/smooth/linear) and the core-shell /cs styles
 * (born/coul/long/cs, buck/coul/long/cs, coul/long/cs). For every style the
 * radial force returned must be the central finite difference of the energy
 * (fpair * r = -dE/dr). Native agreement is covered by tests/oracle/w9pair_*.in.
 */

const ctx: StyleContext = { s: null, readFile: () => '', log: () => {} };

const make = <T extends Pair>(s: T, styleArgs: string[], coeffs: string[]): T => {
  s.settings(styleArgs, ctx);
  s.allocate(2);
  for (const c of coeffs) s.coeff(c.split(/\s+/), ctx);
  s.init(ctx);
  return s;
};

const close = (a: number, b: number, rel: number, abs: number) => Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

/** Central difference with a step proportional to r (roundoff ~1e-11 |E|/r, truncation ~1e-10 |E|/r). */
const dEdr = (e: (r: number) => number, r: number): number => {
  const h = r * 1e-5;
  return (e(r + h) - e(r - h)) / (2 * h);
};

/** fpair * r must equal -dE/dr up to the natural scale |E|/r of a derivative of E at r. */
const forceOk = (fpair: number, e: (r: number) => number, r: number): boolean => {
  const d = dEdr(e, r);
  return Math.abs(fpair * r + d) <= 1e-7 * Math.abs(e(r)) / r + 1e-12;
};

describe('lj/sf: an alias of lj/smooth/linear', () => {
  it('gives the lj/smooth/linear energy and force (measured with native LAMMPS: identical to the last digit)', () => {
    const a = make(PAIR_STYLES['lj/sf']() as Pair, ['2.5'], ['1 1 1.0 1.0', '2 2 0.8 1.1 2.2']);
    const b = make(PAIR_STYLES['lj/smooth/linear']() as Pair, ['2.5'], ['1 1 1.0 1.0', '2 2 0.8 1.1 2.2']);
    expect(a.name).toBe('lj/sf');
    for (const [i, j] of [[1, 1], [1, 2], [2, 2]]) {
      for (const r of [0.95, 1.1, 1.7, 2.1]) {
        const ea = a.single!(0, 1, i, j, r * r, 1, 1, 0, 0), eb = b.single!(0, 1, i, j, r * r, 1, 1, 0, 0);
        expect(ea).toEqual(eb);
      }
    }
  });
});

describe('core-shell /cs styles: force = -dE/dr', () => {
  // The fc < 1 form (core-shell special pair) and the fc = 1 damped form, with the
  // accurate erfc and the series used at small g r.
  it('csCoulPair: fpair * r equals -dE/dr for fc = 0, 0.5, 1 over 1e-3 .. 2.5 A', () => {
    const g = 0.28, qqrd2e = 332.06371;
    for (const [qi, qj] of [[1.0, -1.0], [0.3, 0.3], [-0.8, 0.5]]) {
      for (const fc of [0, 0.5, 1]) {
        for (const r of [1e-3, 0.02, 0.1, 0.8, 1.6, 2.5]) {
          const term = (x: number) => csCoulPair(x * x, qi, qj, g, qqrd2e, fc).e;
          const t = csCoulPair(r * r, qi, qj, g, qqrd2e, fc);
          if (!forceOk(t.f, term, r)) {
            throw new Error(`cs coul qi=${qi} qj=${qj} fc=${fc} r=${r}: fpair*r=${t.f * r} vs -dE/dr=${-dEdr(term, r)}`);
          }
        }
      }
    }
  });

  it('csCoulPair fc = 0 reproduces C qi qj erf(g r)/r (native value at r = 1e-7: 112.408131753734)', () => {
    // qi = +1, qj = -1: E = C erf(g r)/r with C = 332.06371, g = 0.3 (measured with native LAMMPS, black box)
    const e = csCoulPair(1e-14, 1, -1, 0.3, 332.06371, 0).e;
    expect(Math.abs(e - 112.408131753734)).toBeLessThan(1e-9);
    const e1 = csCoulPair(1, 1, -1, 0.3, 332.06371, 0).e;
    expect(Math.abs(e1 - 109.125020951275)).toBeLessThan(1e-9);
  });

  it('erfAcc and hOverX3 agree with the exact erfc and the direct formula', () => {
    for (const x of [0, 0.01, 0.3, 0.5, 0.99, 1, 1.5, 2, 4]) {
      expect(Math.abs(erfAcc(x) + erfcExact(x) - 1), `erf+erfc at ${x}`).toBeLessThan(1e-14);
    }
    const x = 0.5, s = 2 / Math.sqrt(Math.PI);
    const direct = (erfAcc(x) - s * x * Math.exp(-x * x)) / (x * x * x);
    expect(Math.abs(hOverX3(x) - direct)).toBeLessThan(1e-12);
    // continuity of the series/direct switch at x = 1
    expect(Math.abs(hOverX3(0.999999) - hOverX3(1.000001))).toBeLessThan(1e-5);
  });

  it('fc = 1 is the base born/coul/long term with the accurate erfc', () => {
    // the base style now follows native's table / polynomial; the /cs styles keep the accurate erfc
    // (measured, see coul_cs.ts), so compare with the closed form: Born part from table 0 minus its
    // polynomial Coulomb part, plus C q q erfc(g r)/r
    const cs = make(new PairBornCoulLongCS(), ['4.0', '3.0'], ['* * 1.5 0.25 0.9 0.7 0.3']);
    cs.gEwald = 0.28; cs.qqrd2e = 332.06371; cs.table = 12;
    const born = make(new PairBornCoulLong(), ['4.0', '3.0'], ['* * 1.5 0.25 0.9 0.7 0.3']);
    born.gEwald = 0.28; born.qqrd2e = 332.06371; born.table = 0;
    for (const r of [0.8, 1.4, 2.6]) {
      const x = 0.28 * r;
      const coulExact = 332.06371 * 0.3 * -0.3 * erfcExact(x) / r;
      const bornOnly = born.single!(0, 0, 1, 1, r * r, 0, 1, 0, 0).eng; // charges 0: Born term only
      const b = cs.single!(0, 0, 1, 1, r * r, 1, 1, 0.3, -0.3);
      expect(close(b.eng, bornOnly + coulExact, 1e-10, 1e-10), `energy at ${r}`).toBe(true); // erfcFast is ~1e-12
    }
  });

  it('born/coul/long/cs single(): total force = -dE/dr for a core-shell pair (factorCoul 0)', () => {
    const s = make(new PairBornCoulLongCS(), ['4.0', '3.0'], ['* * 1.5 0.25 0.9 0.7 0.3']);
    s.gEwald = 0.28; s.qqrd2e = 332.06371;
    for (const r of [1e-3, 0.05, 0.4, 1.2]) {
      const term = (x: number) => s.single!(0, 0, 1, 1, x * x, 0, 1, 1.0, -1.0).eng;
      const m = s.single!(0, 0, 1, 1, r * r, 0, 1, 1.0, -1.0);
      expect(forceOk(m.fforce, term, r), `r=${r}`).toBe(true);
    }
  });

  it('buck/coul/long/cs single(): total force = -dE/dr for a core-shell pair (factorCoul 0)', () => {
    const s = make(new PairBuckCoulLongCS(), ['4.0', '3.0'], ['* * 500.0 0.27 90.0']);
    s.gEwald = 0.28; s.qqrd2e = 332.06371;
    for (const r of [1e-3, 0.05, 0.4, 1.2]) {
      const term = (x: number) => s.single!(0, 0, 1, 1, x * x, 0, 1, 1.0, -1.0).eng;
      const m = s.single!(0, 0, 1, 1, r * r, 0, 1, 1.0, -1.0);
      expect(forceOk(m.fforce, term, r), `r=${r}`).toBe(true);
    }
  });

  it('write_data: born/buck cs write their base coefficient lines; coul/long/cs writes none', () => {
    const b = make(new PairBornCoulLongCS(), ['10.0', '8.0'], ['* * 6.08 0.317 2.340 24.18 11.51']);
    expect(b.dataCoeffs()).toEqual(['1 6.08 0.317 2.34 24.18 11.51', '2 6.08 0.317 2.34 24.18 11.51']);
    expect(b.dataCoeffsIJ()).toEqual(['1 1 6.08 0.317 2.34 24.18 11.51 10', '1 2 6.08 0.317 2.34 24.18 11.51 10', '2 2 6.08 0.317 2.34 24.18 11.51 10']);
    const u = make(new PairBuckCoulLongCS(), ['10.0'], ['* * 100.0 1.5 200.0']);
    expect(u.dataCoeffs()).toEqual(['1 100 1.5 200', '2 100 1.5 200']);
    const c = make(new PairCoulLongCS(), ['10.0'], ['* *']);
    expect(c.dataCoeffs()).toBeNull();
  });

  it('argument errors throw StyleError-style messages', () => {
    expect(() => new PairBornCoulLongCS().settings([], ctx)).toThrow(/usage: pair_style born\/coul\/long\/cs/);
    expect(() => new PairBornCoulLongCS().settings(['10', '8', '6'], ctx)).toThrow(/usage: pair_style born\/coul\/long\/cs/);
    expect(() => new PairBuckCoulLongCS().settings([], ctx)).toThrow(/usage: pair_style buck\/coul\/long\/cs/);
    expect(() => new PairBuckCoulLongCS().settings(['10', '8', '6'], ctx)).toThrow(/usage: pair_style buck\/coul\/long\/cs/);
    expect(() => new PairCoulLongCS().settings(['10', '8'])).toThrow(/usage: pair_style coul\/long\/cs cutoff/);
    expect(() => new PairCoulLongCS().settings([])).toThrow(/usage: pair_style coul\/long\/cs cutoff/);
    const c = new PairCoulLongCS();
    c.settings(['10']);
    c.allocate(2);
    expect(() => c.coeff(['1', '1', '1.0'])).toThrow(/usage: pair_coeff I J/);
    // the buck and born cs styles reject a non-positive rho like their base styles
    const b = new PairBornCoulLongCS();
    b.settings(['10'], ctx);
    b.allocate(2);
    expect(() => b.coeff(['*', '*', '1.5', '-0.25', '0.9', '0.7', '0.3'], ctx)).toThrow(/rho must be > 0/);
  });
});
