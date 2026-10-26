import { describe, expect, it, beforeAll } from 'vitest';
import { PAIR_STYLES, loadAllStyles } from '../src/engine/styles';
import { erfAcc } from '../src/engine/force/pair/coul_cs';
import { PairCoulLongCS } from '../src/engine-gpl/csdiel/coreshell';
import { PairBornCoulLongCS, PairBuckCoulLongCS } from '../src/engine-gpl/csdiel/born_buck_long_cs';
import { PairBornCoulLong } from '../src/engine/force/pair/coul_long2';
import { erfcExact } from '../src/engine/force/erfc';
import type { Pair, StyleContext } from '../src/engine/force/types';

/*
 * Unit checks for pair_style lj/sf (an alias of lj/smooth/linear) and the core-shell /cs styles
 * (born/coul/long/cs, buck/coul/long/cs, coul/long/cs): their write_data lines and argument errors. Native agreement is
 * covered by tests/oracle/w9pair_*.in and w226cs_*.in.
 */

const ctx: StyleContext = { s: null, readFile: () => '', log: () => {} };

const make = <T extends Pair>(s: T, styleArgs: string[], coeffs: string[]): T => {
  s.settings(styleArgs, ctx);
  s.allocate(2);
  for (const c of coeffs) s.coeff(c.split(/\s+/), ctx);
  s.init(ctx);
  return s;
};

// the lazily loaded style registries (src/engine/lazyStyles.ts) are read directly here
beforeAll(() => loadAllStyles());

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

describe('core-shell /cs styles and the erf helper', () => {
  it('erfAcc agrees with the exact erfc', () => {
    for (const x of [0, 0.01, 0.3, 0.5, 0.99, 1, 1.5, 2, 4]) {
      expect(Math.abs(erfAcc(x) + erfcExact(x) - 1), `erf+erfc at ${x}`).toBeLessThan(1e-14);
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
    // born/coul/long/cs and buck/coul/long/cs are GPL ports with native's messages: tests/engineCoreShellGpl.test.ts
    expect(() => new PairCoulLongCS().settings(['10', '8'])).toThrow(/usage: pair_style coul\/long cutoff/);
    expect(() => new PairCoulLongCS().settings([])).toThrow(/usage: pair_style coul\/long cutoff/);
    const c = new PairCoulLongCS();
    c.settings(['10']);
    c.allocate(2);
    expect(() => c.coeff(['1', '1', '1.0'])).toThrow(/usage: pair_coeff I J/);
  });
});
