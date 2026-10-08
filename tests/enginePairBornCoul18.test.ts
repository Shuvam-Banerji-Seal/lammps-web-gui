import { describe, expect, it } from 'vitest';
import { PairBornCoulDsf, PairBornCoulWolf } from '../src/engine/force/pair/born_coul18';
import { PAIRS as W18_MISC_PAIRS } from '../src/engine/registry/pair_misc18';
import { erfcExact } from '../src/engine/force/erfc';
import type { Pair, StyleContext } from '../src/engine/force/types';

/*
 * Unit checks for born/coul/wolf and born/coul/dsf: the Born-Mayer-Huggins
 * A,C,D term plus the Wolf / damped-shifted-force Coulomb term. The force
 * returned by single() must be the central finite difference of the energy
 * (-dE/dr, the radial force is fforce * r), invalid pair_style/pair_coeff
 * input must raise a StyleError, and a set I,J pair cannot be mixed.
 * Native LAMMPS agreement is covered by tests/oracle/w18borncoul_*.in.
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
      throw new Error(`${s.name}: force != -dE/dr at r=${r}: fforce*r=${m.fforce * r} vs -dE/dr=${-dedr}`);
    }
  }
};

/**
 * The Wolf energy is only potential-shifted ("the Wolf potential is not
 * differentiable at the cutoff", pair_coul.html), so its force is -dE/dr plus
 * a constant damped-shift term. fforce*r + dE/dr must then be the same at
 * every radius inside the cutoff; return that constant.
 */
const wolfShift = (s: Pair, itype: number, jtype: number, radii: number[], qi: number, qj: number, factorCoul = 1) => {
  const single = s.single!;
  const h = 1e-6;
  const deltas: number[] = [];
  for (const r of radii) {
    const em = single.call(s, 0, 0, itype, jtype, (r - h) * (r - h), factorCoul, 0, qi, qj).eng;
    const ep = single.call(s, 0, 0, itype, jtype, (r + h) * (r + h), factorCoul, 0, qi, qj).eng;
    const dedr = (ep - em) / (2 * h);
    const m = single.call(s, 0, 0, itype, jtype, r * r, factorCoul, 0, qi, qj);
    deltas.push(m.fforce * r - -dedr);
  }
  for (const d of deltas) {
    if (!close(d, deltas[0], 1e-6, 1e-7)) throw new Error(`${s.name}: Wolf force shift varies with r: ${deltas.join(', ')}`);
  }
  return deltas[0];
};

const COEFFS = ['* * 1.0 0.1 1.0 1.0 0.5', '1 2 0.8 0.12 1.05 0.8 0.4 2.4'];

describe('born/coul/wolf and born/coul/dsf: force = -dE/dr (central differences)', () => {
  it('born/coul/wolf (born A,C,D term plus Wolf Coulomb)', () => {
    const s = make(new PairBornCoulWolf(), ['0.6', '3.0'], COEFFS);
    // born A,C,D part is differentiable
    finiteDiff(s, 1, 1, [0.7, 1.1, 1.6], 0, 0);
    finiteDiff(s, 2, 2, [0.7, 1.1, 1.6], 0, 0);
    finiteDiff(s, 1, 2, [0.7, 1.1, 1.6], 0, 0);
    // Wolf Coulomb: force = -dE/dr plus the constant damped shift
    const expected = -((erfcExact(1.8) / 3.0) + (2 / Math.sqrt(Math.PI)) * 0.6 * Math.exp(-(1.8 ** 2))) / 3.0;
    const shift11 = wolfShift(s, 1, 1, [0.7, 1.1, 1.6], 0.5, 0.5);
    expect(close(shift11, expected * 0.25, 1e-6, 1e-9)).toBe(true);
    const shift12 = wolfShift(s, 1, 2, [0.7, 1.1, 1.6], 1.5, -2.0);
    expect(close(shift12, expected * -3.0, 1e-6, 1e-9)).toBe(true);
  });

  it('born/coul/dsf (born A,C,D term plus damped shifted force Coulomb)', () => {
    const s = make(new PairBornCoulDsf(), ['0.6', '3.0'], COEFFS);
    finiteDiff(s, 1, 1, [0.7, 1.1, 1.6], 0.5, 0.5);
    finiteDiff(s, 2, 2, [0.7, 1.1, 1.6], -0.5, -0.5);
    finiteDiff(s, 1, 2, [0.7, 1.1, 1.6], 1.5, -2.0);
    finiteDiff(s, 1, 1, [0.7, 1.1, 1.6], 0.5, -0.5);
  });

  it('the force shift makes the dsf Coulomb force vanish at the cutoff', () => {
    const s = make(new PairBornCoulDsf(), ['0.6', '3.0'], COEFFS);
    const near = s.single!(0, 0, 1, 1, (3.0 - 1e-9) ** 2, 1, 0, 1.0, -1.0);
    expect(Math.abs(near.fforce * (3.0 - 1e-9))).toBeLessThan(1e-6);
  });

  it('pair_modify shift subtracts E(rc) from the born part and leaves the force unchanged', () => {
    const plain = make(new PairBornCoulWolf(), ['0.6', '3.0'], COEFFS);
    const shifted = new PairBornCoulWolf();
    shifted.shift = true;
    make(shifted, ['0.6', '3.0'], COEFFS);
    const r = 1.4, rc = 3.0;
    const a = plain.single!(0, 0, 1, 1, r * r, 1, 1, 0.5, -0.5);
    const b = shifted.single!(0, 0, 1, 1, r * r, 1, 1, 0.5, -0.5);
    const bornAtRc = plain.single!(0, 0, 1, 1, (rc - 1e-9) ** 2, 1, 1, 0, 0).eng;
    expect(close(b.eng, a.eng - bornAtRc, 1e-6, 1e-9)).toBe(true);
    expect(close(b.fforce, a.fforce, 1e-12, 1e-14)).toBe(true);
  });

  it('special_bonds subtraction and exclusion follow coul/wolf/dsf', () => {
    const w = make(new PairBornCoulWolf(), ['0.6', '3.0'], COEFFS);
    expect(w.keepExcluded).toBe(true);
    // a weight-0.0 special pair keeps the damped term minus the bare C q_i q_j / r
    const full = w.single!(0, 0, 1, 1, 1.2 ** 2, 1, 1, 1.0, 1.0);
    const zero = w.single!(0, 0, 1, 1, 1.2 ** 2, 0, 1, 1.0, 1.0);
    expect(close(zero.eng, full.eng - 1 / 1.2, 1e-12, 1e-12)).toBe(true);
    const d = make(new PairBornCoulDsf(), ['0.6', '3.0'], COEFFS);
    expect(d.keepExcluded).toBe(true);
  });

  it('the neighbour-list cutoff is the larger of the born and Coulombic cutoffs', () => {
    const s = make(new PairBornCoulWolf(), ['0.6', '3.5', '3.0'], ['* * 1.0 0.1 1.0 1.0 0.5', '1 2 0.8 0.12 1.05 0.8 0.4 2.4']);
    const nt = 3;
    expect(s.cut[1 * nt + 1]).toBeCloseTo(3.5, 12);
    expect(s.cut[2 * nt + 2]).toBeCloseTo(3.5, 12);
    expect(s.cut[1 * nt + 2]).toBeCloseTo(3.0, 12);
  });
});

describe('born/coul/wolf and born/coul/dsf: errors and registration', () => {
  it('bad pair_style arguments throw a StyleError', () => {
    expect(() => new PairBornCoulWolf().settings([], ctx)).toThrow(/usage: pair_style born\/coul\/wolf/);
    expect(() => new PairBornCoulWolf().settings(['0.6'], ctx)).toThrow(/usage: pair_style born\/coul\/wolf/);
    expect(() => new PairBornCoulWolf().settings(['0.6', '3.0', '4.0', '5.0'], ctx)).toThrow(/usage: pair_style born\/coul\/wolf/);
    expect(() => new PairBornCoulDsf().settings([], ctx)).toThrow(/usage: pair_style born\/coul\/dsf/);
    expect(() => new PairBornCoulDsf().settings(['0.6', '3.0', '4.0', '5.0'], ctx)).toThrow(/usage: pair_style born\/coul\/dsf/);
  });

  it('no per-pair Coulomb cutoff is accepted (pair_born.html), so a 9th coeff throws', () => {
    const s = new PairBornCoulWolf();
    s.settings(['0.6', '3.0'], ctx);
    s.allocate(2);
    expect(() => s.coeff(['1', '1', '1.0', '0.1', '1.0', '1.0', '0.5', '2.4', '2.8'], ctx)).toThrow(/usage: pair_coeff/);
  });

  it('an unset I,J pair throws (these styles do not mix)', () => {
    for (const s of [new PairBornCoulWolf(), new PairBornCoulDsf()] as Pair[]) {
      s.settings(['0.6', '3.0'], ctx);
      s.allocate(2);
      s.coeff(['1', '1', '1.0', '0.1', '1.0', '1.0', '0.5'], ctx);
      s.coeff(['2', '2', '0.8', '0.12', '1.05', '0.8', '0.4'], ctx);
      expect(() => s.init(ctx), s.name).toThrow(/all pair coeffs are not set \(pair 1 2\)/);
    }
  });

  it('rho must be > 0', () => {
    const s = new PairBornCoulDsf();
    s.settings(['0.6', '3.0'], ctx);
    s.allocate(2);
    expect(() => s.coeff(['*', '*', '1.0', '0.0', '1.0', '1.0', '0.5'], ctx)).toThrow(/rho must be > 0/);
  });

  it('both styles expose the global Coulomb cutoff via extract', () => {
    expect(make(new PairBornCoulWolf(), ['0.6', '2.5', '3.0'], COEFFS).extract('cut_coul')).toBe(3.0);
    expect(make(new PairBornCoulDsf(), ['0.6', '2.5', '3.0'], COEFFS).extract('cut_coul')).toBe(3.0);
    expect(make(new PairBornCoulWolf(), ['0.6', '3.0'], COEFFS).extract('cut_coul')).toBe(3.0);
  });

  it('the wave-18 registry exposes both factories', () => {
    expect(W18_MISC_PAIRS['born/coul/wolf']!().name).toBe('born/coul/wolf');
    expect(W18_MISC_PAIRS['born/coul/dsf']!().name).toBe('born/coul/dsf');
  });
});
