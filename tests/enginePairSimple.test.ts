import { describe, expect, it } from 'vitest';
import { PairBuck, PairBuckCoulCut, PairBorn, PairMorse } from '../src/engine/force/pair/simple';
import type { Pair, StyleContext } from '../src/engine/force/types';

/*
 * Unit checks for the simple pair styles: the force returned by single()
 * must be the central finite difference of the energy (-dE/dr, where the
 * radial force is fforce * r), and incomplete or invalid pair_coeff input
 * must raise a StyleError. Native LAMMPS agreement is covered by the
 * tests/oracle/w1buck_*.in cases.
 */

const ctx: StyleContext = { s: null, readFile: () => '', log: () => {} };

const make = (s: Pair, styleArgs: string[], coeffs: string[]): Pair => {
  s.settings(styleArgs, ctx);
  s.allocate(2);
  for (const c of coeffs) s.coeff(c.split(/\s+/), ctx);
  s.init(ctx);
  return s;
};

/** |a - b| <= abs + rel * max(|a|, |b|) */
const close = (a: number, b: number, rel: number, abs: number) => Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

/** fforce * r must equal -dE/dr (central difference) at each geometry. */
const finiteDiff = (s: Pair, itype: number, jtype: number, radii: number[], qi = 0, qj = 0, factorCoul = 1, factorLJ = 1) => {
  const single = s.single!;
  const h = 1e-6;
  for (const r of radii) {
    const em = single.call(s, 0, 0, itype, jtype, (r - h) * (r - h), factorCoul, factorLJ, qi, qj).eng;
    const ep = single.call(s, 0, 0, itype, jtype, (r + h) * (r + h), factorCoul, factorLJ, qi, qj).eng;
    const dedr = (ep - em) / (2 * h);
    const m = single.call(s, 0, 0, itype, jtype, r * r, factorCoul, factorLJ, qi, qj);
    if (!close(m.fforce * r, -dedr, 1e-6, 1e-7)) {
      throw new Error(`force != -dE/dr at r=${r}: fforce*r=${m.fforce * r} vs -dE/dr=${-dedr}`);
    }
  }
};

describe('simple pair styles: force = -dE/dr (central differences)', () => {
  it('buck', () => {
    const s = make(new PairBuck(), ['2.5'], ['* * 600.0 0.15 1.0', '2 2 800.0 0.16 1.2 2.2']);
    finiteDiff(s, 1, 1, [0.9, 1.3, 1.9]);
    finiteDiff(s, 2, 2, [0.9, 1.3, 1.9]);
    finiteDiff(s, 1, 2, [0.9, 1.3, 1.9]);
  });

  it('buck/coul/cut (Buckingham and Coulomb parts)', () => {
    const s = make(new PairBuckCoulCut(), ['2.5', '3.0'], ['* * 600.0 0.15 1.0', '1 2 700.0 0.155 1.1 2.3 2.8']);
    finiteDiff(s, 1, 1, [0.9, 1.3, 1.9], 0.5, -0.5);
    finiteDiff(s, 1, 2, [0.9, 1.3, 1.9], 1.5, -2.0);
    finiteDiff(s, 2, 2, [0.9, 1.3, 1.9], 0.5, -0.5);
  });

  it('born', () => {
    const s = make(new PairBorn(), ['2.5'], ['* * 1.0 0.1 1.0 1.0 0.5', '2 2 0.8 0.12 1.05 0.8 0.4']);
    finiteDiff(s, 1, 1, [0.7, 1.1, 1.6]);
    finiteDiff(s, 2, 2, [0.7, 1.1, 1.6]);
    finiteDiff(s, 1, 2, [0.7, 1.1, 1.6]);
  });

  it('morse', () => {
    const s = make(new PairMorse(), ['2.5'], ['* * 1.0 3.0 1.12', '1 2 0.8 2.8 1.15 2.2']);
    finiteDiff(s, 1, 1, [0.9, 1.12, 1.8]);
    finiteDiff(s, 2, 2, [0.9, 1.12, 1.8]);
    finiteDiff(s, 1, 2, [0.9, 1.12, 1.8]);
  });

  it('pair_modify shift changes the energy by E(rc) but not the force', () => {
    for (const build of [
      () => new PairBuck(),
      () => new PairBuckCoulCut(),
      () => new PairBorn(),
      () => new PairMorse(),
    ]) {
      const plain = build(), shifted = build();
      const styleArgs = build() instanceof PairBuckCoulCut ? ['2.5', '3.0'] : ['2.5'];
      const coeffs = build() instanceof PairBuckCoulCut
        ? ['* * 600.0 0.15 1.0']
        : build() instanceof PairBorn
          ? ['* * 1.0 0.1 1.0 1.0 0.5']
          : build() instanceof PairMorse
            ? ['* * 1.0 3.0 1.12']
            : ['* * 600.0 0.15 1.0'];
      make(plain, styleArgs, coeffs);
      shifted.shift = true;
      make(shifted, styleArgs, coeffs);
      const r = 1.4, rc = 2.5;
      const a = plain.single!(0, 0, 1, 1, r * r, 1, 1, 0, 0);
      const b = shifted.single!(0, 0, 1, 1, r * r, 1, 1, 0, 0);
      const atRc = plain.single!(0, 0, 1, 1, rc * rc, 1, 1, 0, 0).eng;
      if (!close(b.eng, a.eng - atRc, 1e-12, 1e-12) || !close(b.fforce, a.fforce, 1e-12, 1e-14)) {
        throw new Error(`${plain.name}: shift yes must subtract E(rc) and leave the force unchanged`);
      }
    }
  });
});

describe('simple pair styles: errors', () => {
  it('an unset I,J pair throws (no mixing for these styles)', () => {
    const cases: [string, Pair, string[], string[]][] = [
      ['buck', new PairBuck(), ['2.5'], ['1 1 600.0 0.15 1.0', '2 2 800.0 0.16 1.2']],
      ['buck/coul/cut', new PairBuckCoulCut(), ['2.5', '3.0'], ['1 1 600.0 0.15 1.0', '2 2 800.0 0.16 1.2']],
      ['born', new PairBorn(), ['2.5'], ['1 1 1.0 0.1 1.0 1.0 0.5', '2 2 0.8 0.12 1.05 0.8 0.4']],
      ['morse', new PairMorse(), ['2.5'], ['1 1 1.0 3.0 1.12', '2 2 0.8 2.8 1.15']],
    ];
    for (const [name, s, styleArgs, coeffs] of cases) {
      s.settings(styleArgs, ctx);
      s.allocate(2);
      for (const c of coeffs) s.coeff(c.split(/\s+/), ctx);
      expect(() => s.init(ctx), name).toThrow(/all pair coeffs are not set \(pair 1 2\)/);
    }
  });

  it('bad pair_style / pair_coeff arguments throw', () => {
    expect(() => new PairBuck().settings([], ctx)).toThrow(/usage: pair_/);
    expect(() => new PairBuck().settings(['2.5', '3.0'], ctx)).toThrow(/usage: pair_/);
    expect(() => new PairBuckCoulCut().settings(['2.5', '3.0', '4.0'], ctx)).toThrow(/usage: pair_/);
    expect(() => new PairBorn().settings([], ctx)).toThrow(/usage: pair_/);
    expect(() => new PairMorse().settings(['2.5', '1.0'], ctx)).toThrow(/usage: pair_/);
    // buck takes at most one per-pair cutoff (no Coulombic terms)
    const buck = new PairBuck();
    buck.settings(['2.5'], ctx);
    buck.allocate(2);
    expect(() => buck.coeff(['1', '1', '600.0', '0.15', '1.0', '2.0', '3.0'], ctx)).toThrow(/usage: pair_/);
    // rho must be > 0 for buck and born
    const buck2 = new PairBuck();
    buck2.settings(['2.5'], ctx);
    buck2.allocate(2);
    expect(() => buck2.coeff(['*', '*', '600.0', '0.0', '1.0'], ctx)).toThrow(/rho must be > 0/);
    const born = new PairBorn();
    born.settings(['2.5'], ctx);
    born.allocate(2);
    expect(() => born.coeff(['*', '*', '1.0', '-0.1', '1.0', '1.0', '0.5'], ctx)).toThrow(/rho must be > 0/);
    // missing coefficients
    expect(() => buck.coeff(['1', '1', '600.0', '0.15'], ctx)).toThrow(/usage: pair_/);
    expect(() => born.coeff(['1', '1', '1.0', '0.1', '1.0', '1.0'], ctx)).toThrow(/usage: pair_/);
    expect(() => new PairMorse().coeff(['1', '1', '1.0', '3.0'], ctx)).toThrow(/usage: pair_/);
  });

  it('pair_modify tail yes is rejected by morse (docs: not supported)', () => {
    const s = new PairMorse();
    s.settings(['2.5'], ctx);
    s.allocate(2);
    s.coeff(['*', '*', '1.0', '3.0', '1.12'], ctx);
    s.tail = true;
    expect(() => s.init(ctx)).toThrow(/tail/);
  });

  it('buck/coul/cut exposes its Coulomb cutoff via extract', () => {
    const s = make(new PairBuckCoulCut(), ['2.5', '3.0'], ['* * 600.0 0.15 1.0']);
    expect(s.extract('cut_coul')).toBe(3.0);
    const one = make(new PairBuckCoulCut(), ['2.5'], ['* * 600.0 0.15 1.0']);
    expect(one.extract('cut_coul')).toBe(2.5);
  });
});
