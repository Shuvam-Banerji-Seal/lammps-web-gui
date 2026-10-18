import { describe, expect, it } from 'vitest';
import { PairBornCoulDsfCS, PairBornCoulWolfCS } from '../src/engine-gpl/csdiel/born_dsf_wolf_cs';
import { PairBornCoulLongCS, PairBuckCoulLongCS } from '../src/engine-gpl/csdiel/born_buck_long_cs';
import { expmsq, myErfcx } from '../src/engine-gpl/csdiel/erfcx';
import { erfc as glibcErfc } from '../src/engine-gpl/libm/erf';
import { PAIRS as GPL_CSDIEL } from '../src/engine-gpl/registry/csdiel';
import { PAIRS as CORE_MISC4 } from '../src/engine/registry/pair_misc4';
import { PAIRS as CORE_MISC18 } from '../src/engine/registry/pair_misc18';
import type { Pair, StyleContext } from '../src/engine/force/types';

/*
 * The four /cs styles ported from LAMMPS into src/engine-gpl/csdiel/ (born/coul/dsf/cs, born/coul/wolf/cs, born/coul/long/cs,
 * buck/coul/long/cs): coefficient parsing, pair_style and pair_coeff argument checks and the error texts of native LAMMPS
 * (src/EXTRA-PAIR/pair_born_coul_dsf.cpp, pair_born_coul_wolf.cpp, src/KSPACE/pair_born_coul_long.cpp, pair_buck_coul_long.cpp
 * at b332f28: "Illegal pair_style command", "Incorrect args for pair coefficients"). Native agreement of the numbers is
 * covered by tests/oracle/w21borncs_*.in, w9pair_born_cs.in, w9pair_buck_cs.in and w19cs_r0.in.
 */

const ctx: StyleContext = { s: null, readFile: () => '', log: () => {} };

/** Settings, allocation and the coefficient lines of one style, without init (so a test can choose init or not). */
const prepare = <T extends Pair>(s: T, styleArgs: string[], coeffs: string[], ntypes = 2): T => {
  s.settings(styleArgs, ctx);
  s.allocate(ntypes);
  for (const c of coeffs) s.coeff(c.split(/\s+/), ctx);
  return s;
};

const nt = 3;

describe('registration of the ported /cs styles', () => {
  it('lists the four names in the GPL registry and no longer in the core registries', () => {
    for (const name of ['born/coul/dsf/cs', 'born/coul/wolf/cs', 'born/coul/long/cs', 'buck/coul/long/cs']) {
      expect(GPL_CSDIEL[name], name).toBeTypeOf('function');
      expect(CORE_MISC4[name], `core pair_misc4 ${name}`).toBeUndefined();
      expect(CORE_MISC18[name], `core pair_misc18 ${name}`).toBeUndefined();
    }
  });

  it('gives each class its native style name', () => {
    expect(new PairBornCoulDsfCS().name).toBe('born/coul/dsf/cs');
    expect(new PairBornCoulWolfCS().name).toBe('born/coul/wolf/cs');
    expect(new PairBornCoulLongCS().name).toBe('born/coul/long/cs');
    expect(new PairBuckCoulLongCS().name).toBe('buck/coul/long/cs');
  });
});

describe('pair_style arguments (native "Illegal pair_style command")', () => {
  it('born/coul/dsf/cs and born/coul/wolf/cs take 2 or 3 arguments', () => {
    for (const make of [() => new PairBornCoulDsfCS(), () => new PairBornCoulWolfCS()]) {
      expect(() => make().settings([], ctx)).toThrow('Illegal pair_style command');
      expect(() => make().settings(['0.3'], ctx)).toThrow('Illegal pair_style command');
      expect(() => make().settings(['0.3', '8.0', '9.0', '10.0'], ctx)).toThrow('Illegal pair_style command');
    }
  });

  it('born/coul/long/cs and buck/coul/long/cs take 1 or 2 arguments', () => {
    for (const make of [() => new PairBornCoulLongCS(), () => new PairBuckCoulLongCS()]) {
      expect(() => make().settings([], ctx)).toThrow('Illegal pair_style command');
      expect(() => make().settings(['8.0', '9.0', '10.0'], ctx)).toThrow('Illegal pair_style command');
    }
  });

  it('sets alpha, the LJ cutoff and the Coulomb cutoff (cutoff2 defaults to the LJ cutoff)', () => {
    const s = prepare(new PairBornCoulDsfCS(), ['0.3', '8.0'], ['* * 0.0 1.0 0.0 0.0 0.0']);
    expect(s.extract('cut_coul')).toBe(8.0);
    const w = prepare(new PairBornCoulWolfCS(), ['0.3', '8.0', '9.5'], ['* * 0.0 1.0 0.0 0.0 0.0']);
    expect(w.extract('cut_coul')).toBe(9.5);
    const l = prepare(new PairBornCoulLongCS(), ['10.0'], ['* * 1.0 0.3 1.0 1.0 0.5']);
    expect(l.extract('cut_coul')).toBe(10.0);
    const b = prepare(new PairBuckCoulLongCS(), ['10.0', '8.0'], ['* * 1.0 0.3 1.0']);
    expect(b.extract('cut_coul')).toBe(8.0);
  });
});

describe('pair_coeff arguments (native "Incorrect args for pair coefficients")', () => {
  it('born styles take 7 or 8 arguments (I J A rho sigma C D [cutoff])', () => {
    expect(() => prepare(new PairBornCoulDsfCS(), ['0.3', '8.0'], ['* * 1.0 0.3 1.0 1.0'])).toThrow('Incorrect args for pair coefficients');
    expect(() => prepare(new PairBornCoulWolfCS(), ['0.3', '8.0'], ['* * 1.0 0.3 1.0 1.0 0.5 9.0 1.0'])).toThrow('Incorrect args for pair coefficients');
    expect(() => prepare(new PairBornCoulLongCS(), ['10.0'], ['* * 1.0 0.3 1.0 1.0'])).toThrow('Incorrect args for pair coefficients');
  });

  it('buck/coul/long/cs takes 5 or 6 arguments (I J A rho C [cutoff])', () => {
    expect(() => prepare(new PairBuckCoulLongCS(), ['10.0'], ['* * 1.0 0.3'])).toThrow('Incorrect args for pair coefficients');
    expect(() => prepare(new PairBuckCoulLongCS(), ['10.0'], ['* * 1.0 0.3 1.0 9.0 1.0'])).toThrow('Incorrect args for pair coefficients');
  });

  it('rho must be greater than zero (native check, same message)', () => {
    expect(() => prepare(new PairBornCoulDsfCS(), ['0.3', '8.0'], ['* * 1.0 0.0 1.0 1.0 0.5'])).toThrow('Incorrect args for pair coefficients');
    expect(() => prepare(new PairBornCoulLongCS(), ['10.0'], ['* * 1.0 -0.3 1.0 1.0 0.5'])).toThrow('Incorrect args for pair coefficients');
    expect(() => prepare(new PairBuckCoulLongCS(), ['10.0'], ['* * 1.0 0.0 1.0'])).toThrow('Incorrect args for pair coefficients');
  });

  it('parses the optional per-pair cutoff; the default is the global LJ cutoff', () => {
    const s = prepare(new PairBornCoulLongCS(), ['10.0', '8.0'], ['* * 1.0 0.3 1.0 1.0 0.5', '1 1 1.0 0.3 1.0 1.0 0.5 6.0']);
    s.init(ctx);
    // cut = MAX(cut_lj, cut_coul) per pair (native init_one)
    expect(s.cut[1 * nt + 1]).toBe(8.0);
    expect(s.cut[2 * nt + 2]).toBe(10.0);
    expect(s.cut[1 * nt + 2]).toBe(10.0);
  });

  it('pair cutoff below the Coulomb cutoff is raised to it (MAX of cut_lj and cut_coul)', () => {
    const s = prepare(new PairBuckCoulLongCS(), ['5.0', '8.0'], ['* * 500.0 0.27 90.0 4.0']);
    s.init(ctx);
    expect(s.cut[1 * nt + 1]).toBe(8.0);
    expect(s.cut[2 * nt + 2]).toBe(8.0);
  });

  it('an unset pair is an error at init (native "All pair coeffs are not set")', () => {
    const s = prepare(new PairBornCoulDsfCS(), ['0.3', '8.0'], ['1 1 6.08 0.317 2.340 24.18 11.51']);
    expect(() => s.init(ctx)).toThrow('All pair coeffs are not set');
  });
});

describe('the /cs single() is refused (native single_enable = 0)', () => {
  it('each style throws a message naming it', () => {
    const cases: [Pair, string][] = [
      [new PairBornCoulDsfCS(), 'born/coul/dsf/cs'],
      [new PairBornCoulWolfCS(), 'born/coul/wolf/cs'],
      [new PairBornCoulLongCS(), 'born/coul/long/cs'],
      [new PairBuckCoulLongCS(), 'buck/coul/long/cs'],
    ];
    for (const [s, name] of cases) {
      expect(() => s.single!(0, 0, 1, 1, 1.0, 0, 1, 1, -1)).toThrow(name);
    }
  });
});

describe('erfc building blocks of born/coul/dsf/cs (math_special.h my_erfcx and expmsq)', () => {
  it('my_erfcx(x) * expmsq(x) is erfc(x) to 1e-12 relative over the cutoff range', () => {
    for (const x of [0.05, 0.3, 0.9, 1.5, 2.5, 4.0, 6.0]) {
      const ref = glibcErfc(x);
      const got = myErfcx(x) * expmsq(x);
      expect(Math.abs(got - ref) / ref, `x=${x}`).toBeLessThan(1e-12);
    }
  });

  it('expmsq(x) is exp(-x^2) to 1e-13 relative', () => {
    for (const x of [0.0, 0.2, 0.7, 1.3, 2.9, 5.0]) {
      const ref = Math.exp(-x * x);
      expect(Math.abs(expmsq(x) - ref) / ref, `x=${x}`).toBeLessThan(1e-13);
    }
  });

  it('expmsq(x) is 0 once x^2 log2(e) reaches 1023 (native guard)', () => {
    expect(expmsq(30)).toBe(0);
  });
});
