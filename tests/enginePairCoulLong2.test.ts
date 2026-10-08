import { describe, expect, it } from 'vitest';
import { PairBornCoulLong, PairBuckCoulLong } from '../src/engine/force/pair/coul_long2';
import type { Pair, StyleContext } from '../src/engine/force/types';

/*
 * Unit checks for the born/coul/long and buck/coul/long styles: the force
 * returned by single() must be the central finite difference of the energy
 * (-dE/dr, where the radial force is fforce * r), including the damped
 * Ewald real-space Coulomb term (gEwald > 0) and the special-bonds weight
 * (factorCoul < 1). Incomplete or invalid pair_coeff input must raise a
 * StyleError. Native LAMMPS agreement is covered by the
 * tests/oracle/w3coullong_*.in cases.
 */

const ctx: StyleContext = { s: null, readFile: () => '', log: () => {} };

const make = <T extends Pair>(s: T, styleArgs: string[], coeffs: string[]): T => {
  s.settings(styleArgs, ctx);
  s.allocate(2);
  for (const c of coeffs) s.coeff(c.split(/\s+/), ctx);
  s.init(ctx);
  return s;
};

/** |a - b| <= abs + rel * max(|a|, |b|) */
const close = (a: number, b: number, rel: number, abs: number) => Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

/** fforce * r must equal -dE/dr (central difference) at each geometry. */
const finiteDiff = (s: Pair, itype: number, jtype: number, radii: number[], qi = 0, qj = 0, factorCoul = 1, factorLJ = 1, rel = 1e-6) => {
  const single = s.single!;
  // the erfc table interpolates at float32(r^2) (as native does), so its difference quotient needs a
  // step well above the float32 spacing of r^2
  const h = s.table > 0 ? 1e-3 : 1e-6;
  for (const r of radii) {
    const em = single.call(s, 0, 0, itype, jtype, (r - h) * (r - h), factorCoul, factorLJ, qi, qj).eng;
    const ep = single.call(s, 0, 0, itype, jtype, (r + h) * (r + h), factorCoul, factorLJ, qi, qj).eng;
    const dedr = (ep - em) / (2 * h);
    const m = single.call(s, 0, 0, itype, jtype, r * r, factorCoul, factorLJ, qi, qj);
    if (!close(m.fforce * r, -dedr, rel, 1e-7)) {
      throw new Error(`${s.name}: force != -dE/dr at r=${r}: fforce*r=${m.fforce * r} vs -dE/dr=${-dedr}`);
    }
  }
};

describe('born/coul/long and buck/coul/long: force = -dE/dr (central differences)', () => {
  for (const table of [12, 0]) {
    // With table 0 the energy uses the Abramowitz & Stegun 7.1.26 erfc fit
    // (erfc.ts, absolute error <= 1.5e-7) whose slope is not exactly erfc' =
    // -2/sqrt(pi) exp(-x^2), while the force formula assumes it is — so the
    // finite difference of the energy disagrees with the force at ~1e-6
    // relative, exactly as in native LAMMPS, which uses the same published
    // fit. A real term/sign error would be O(1) relative; 1e-4 catches it.
    // With table 12 the energy and the force are separate linear interpolations in r^2 (as native
    // LAMMPS tabulates them, erfc.ts makeErfcTable); inside a bin the difference quotient is the
    // chord slope, so they agree to ~1e-3 relative. A term or sign error is still O(1).
    const rel = table === 0 ? 1e-4 : 5e-3;
    it(`born/coul/long (pair_modify table ${table})`, () => {
      const s = make(new PairBornCoulLong(), ['4.0', '3.0'], ['* * 1.5 0.25 0.9 0.7 0.3', '1 2 2.0 0.28 0.85 0.9 0.4']);
      s.gEwald = 0.28;
      s.qqrd2e = 332.06371;
      s.table = table;
      finiteDiff(s, 1, 1, [0.8, 1.4, 2.6], 0.3, -0.3, 1, 1, rel);
      finiteDiff(s, 2, 2, [0.8, 1.4, 2.6], -0.3, -0.3, 1, 1, rel);
      finiteDiff(s, 1, 2, [0.8, 1.4, 2.6], 0.3, -0.3, 1, 1, rel);
      // special_bonds weight (factorCoul < 1) must stay force = -dE/dr
      finiteDiff(s, 1, 2, [0.8, 1.4, 2.6], 0.3, -0.3, 0.5, 1, rel);
    });

    it(`buck/coul/long (pair_modify table ${table})`, () => {
      const s = make(new PairBuckCoulLong(), ['4.0', '3.0'], ['* * 500.0 0.27 90.0', '1 2 450.0 0.28 85.0 3.4']);
      s.gEwald = 0.28;
      s.qqrd2e = 332.06371;
      s.table = table;
      finiteDiff(s, 1, 1, [0.9, 1.5, 2.7], 0.3, -0.3, 1, 1, rel);
      finiteDiff(s, 2, 2, [0.9, 1.5, 2.7], -0.3, -0.3, 1, 1, rel);
      finiteDiff(s, 1, 2, [0.9, 1.5, 2.7], 0.3, -0.3, 1, 1, rel);
      finiteDiff(s, 1, 2, [0.9, 1.5, 2.7], 0.3, -0.3, 0.5, 1, rel);
    });
  }

  it('pair_modify shift changes the non-Coulomb energy by E(rc) but not the force', () => {
    const cases: [() => Pair, string[], string[]][] = [
      [() => new PairBornCoulLong(), ['4.0', '3.0'], ['* * 1.5 0.25 0.9 0.7 0.3']],
      [() => new PairBuckCoulLong(), ['4.0', '3.0'], ['* * 500.0 0.27 90.0']],
    ];
    for (const [build, styleArgs, coeffs] of cases) {
      const plain = build(), shifted = build();
      make(plain, styleArgs, coeffs);
      shifted.shift = true;
      make(shifted, styleArgs, coeffs);
      const r = 1.4, rc = 4.0;
      const qi = 0.3, qj = -0.3;
      const a = plain.single!(0, 0, 1, 1, r * r, 1, 1, qi, qj);
      const b = shifted.single!(0, 0, 1, 1, r * r, 1, 1, qi, qj);
      // shift applies only to the exp(), 1/r^6 (and 1/r^8) part: E(rc) without Coulomb
      const atRc = plain.single!(0, 0, 1, 1, rc * rc, 1, 1, 0, 0).eng;
      if (!close(b.eng, a.eng - atRc, 1e-12, 1e-12) || !close(b.fforce, a.fforce, 1e-12, 1e-14)) {
        throw new Error(`${plain.name}: shift yes must subtract E(rc) (non-Coulomb only) and leave the force unchanged`);
      }
    }
  });
});

describe('born/coul/long and buck/coul/long: errors', () => {
  it('an unset I,J pair throws (these styles do not support mixing)', () => {
    const cases: [string, Pair, string[], string[]][] = [
      ['born/coul/long', new PairBornCoulLong(), ['4.0'], ['1 1 1.5 0.25 0.9 0.7 0.3', '2 2 2.0 0.28 0.85 0.9 0.4']],
      ['buck/coul/long', new PairBuckCoulLong(), ['4.0'], ['1 1 500.0 0.27 90.0', '2 2 450.0 0.28 85.0']],
    ];
    for (const [name, s, styleArgs, coeffs] of cases) {
      s.settings(styleArgs, ctx);
      s.allocate(2);
      for (const c of coeffs) s.coeff(c.split(/\s+/), ctx);
      expect(() => s.init(ctx), name).toThrow(/all pair coeffs are not set \(pair 1 2\)/);
    }
  });

  it('bad pair_style / pair_coeff arguments throw', () => {
    expect(() => new PairBornCoulLong().settings([], ctx)).toThrow(/usage: pair_style born\/coul\/long/);
    expect(() => new PairBornCoulLong().settings(['4.0', '3.0', '2.0'], ctx)).toThrow(/usage: pair_style born\/coul\/long/);
    expect(() => new PairBuckCoulLong().settings([], ctx)).toThrow(/usage: pair_style buck\/coul\/long/);
    expect(() => new PairBuckCoulLong().settings(['4.0', '3.0', '2.0'], ctx)).toThrow(/usage: pair_style buck\/coul\/long/);
    // no per-pair Coulomb cutoff for coul/long styles (docs pair_born/pair_buck)
    const born = new PairBornCoulLong();
    born.settings(['4.0', '3.0'], ctx);
    born.allocate(2);
    expect(() => born.coeff(['1', '1', '1.5', '0.25', '0.9', '0.7', '0.3', '3.5', '3.0'], ctx)).toThrow(/usage: pair_coeff/);
    const buck = new PairBuckCoulLong();
    buck.settings(['4.0', '3.0'], ctx);
    buck.allocate(2);
    expect(() => buck.coeff(['1', '1', '500.0', '0.27', '90.0', '3.5', '3.0'], ctx)).toThrow(/usage: pair_coeff/);
    // rho must be > 0
    const buck2 = new PairBuckCoulLong();
    buck2.settings(['4.0'], ctx);
    buck2.allocate(2);
    expect(() => buck2.coeff(['*', '*', '500.0', '0.0', '90.0'], ctx)).toThrow(/rho must be > 0/);
    const born2 = new PairBornCoulLong();
    born2.settings(['4.0'], ctx);
    born2.allocate(2);
    expect(() => born2.coeff(['*', '*', '1.5', '-0.25', '0.9', '0.7', '0.3'], ctx)).toThrow(/rho must be > 0/);
    // missing coefficients
    expect(() => born.coeff(['1', '1', '1.5', '0.25', '0.9', '0.7'], ctx)).toThrow(/usage: pair_coeff/);
    expect(() => buck.coeff(['1', '1', '500.0', '0.27'], ctx)).toThrow(/usage: pair_coeff/);
  });

  it('extract cut_coul follows the pair_style Coulomb cutoff', () => {
    const two = make(new PairBornCoulLong(), ['4.0', '3.0'], ['* * 1.5 0.25 0.9 0.7 0.3']);
    expect(two.extract('cut_coul')).toBe(3.0);
    const one = make(new PairBuckCoulLong(), ['4.0'], ['* * 500.0 0.27 90.0']);
    expect(one.extract('cut_coul')).toBe(4.0);
  });
});
