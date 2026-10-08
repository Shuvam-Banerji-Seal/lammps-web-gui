import { describe, expect, it } from 'vitest';
import { PairZBL } from '../src/engine/force/pair/zbl';
import type { Pair, StyleContext } from '../src/engine/force/types';

/*
 * Unit checks for pair_style zbl: the radial force returned by single() must
 * be the central finite difference of the energy (-dE/dr = fforce * r) across
 * all three branches of the potential (below the inner cutoff, at it, inside
 * the switching region, near the outer cutoff), the switching polynomial must
 * ramp energy and force to zero at the outer cutoff, and incomplete or
 * invalid input must raise a StyleError. Native LAMMPS agreement is covered
 * by tests/oracle/w3zbl_zbl.in.
 */

const ctx: StyleContext = { s: null, readFile: () => '', log: () => {} };

const make = (styleArgs: string[], coeffs: string[]): PairZBL => {
  const s: Pair = new PairZBL();
  s.settings(styleArgs, ctx);
  s.allocate(2);
  for (const c of coeffs) s.coeff(c.split(/\s+/), ctx);
  s.init(ctx);
  return s as PairZBL;
};

/** |a - b| <= abs + rel * max(|a|, |b|) */
const close = (a: number, b: number, rel: number, abs: number) => Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

/** fforce * r must equal -dE/dr (central difference) at each geometry. */
const finiteDiff = (s: PairZBL, itype: number, jtype: number, radii: number[], factorCoul = 1) => {
  const single = s.single!;
  const h = 1e-6;
  for (const r of radii) {
    const em = single.call(s, 0, 0, itype, jtype, (r - h) * (r - h), factorCoul, 1, 0, 0).eng;
    const ep = single.call(s, 0, 0, itype, jtype, (r + h) * (r + h), factorCoul, 1, 0, 0).eng;
    const dedr = (ep - em) / (2 * h);
    const m = single.call(s, 0, 0, itype, jtype, r * r, factorCoul, 1, 0, 0);
    if (!close(m.fforce * r, -dedr, 1e-6, 1e-7)) {
      throw new Error(`zbl: force != -dE/dr at r=${r}: fforce*r=${m.fforce * r} vs -dE/dr=${-dedr}`);
    }
  }
};

const ORACLE_COEFFS = ['1 1 29.0 29.0', '2 2 14.0 14.0', '1 2 29.0 14.0'];
const RADII = [1.5, 2.0, 2.6, 3.1];

describe('zbl: force = -dE/dr (central differences)', () => {
  it('explicit coefficients, all three branches of the switching function', () => {
    const s = make(['2.0', '3.2'], ORACLE_COEFFS);
    finiteDiff(s, 1, 1, RADII);
    finiteDiff(s, 2, 2, RADII);
    finiteDiff(s, 1, 2, RADII);
  });

  it('mixed cross coefficients (Z_a, Z_b from the i==j pairs)', () => {
    const s = make(['2.0', '3.2'], ['1 1 29.0 29.0', '2 2 14.0 14.0']);
    expect(s.p.get('z1', 1, 2)).toBe(29.0);
    expect(s.p.get('z2', 1, 2)).toBe(14.0);
    finiteDiff(s, 1, 2, RADII);
  });

  it('energy and force vanish at the outer cutoff (switching property)', () => {
    const s = make(['2.0', '3.2'], ORACLE_COEFFS);
    for (const [i, j] of [[1, 1], [2, 2], [1, 2]] as const) {
      const m = s.single!(0, 0, i, j, 3.2 * 3.2, 1, 1, 0, 0);
      if (!close(m.eng, 0, 1e-9, 1e-12) || !close(m.fforce, 0, 1e-9, 1e-12)) {
        throw new Error(`zbl: E and F must vanish at r_c, got E=${m.eng} F=${m.fforce} for pair ${i} ${j}`);
      }
    }
  });

  it('special (Coulomb) weight scales energy and force alike', () => {
    const s = make(['2.0', '3.2'], ORACLE_COEFFS);
    const a = s.single!(0, 0, 1, 2, 2.6 * 2.6, 1, 1, 0, 0);
    const b = s.single!(0, 0, 1, 2, 2.6 * 2.6, 0.5, 1, 0, 0);
    if (!close(b.eng, 0.5 * a.eng, 1e-12, 1e-14) || !close(b.fforce, 0.5 * a.fforce, 1e-12, 1e-14)) {
      throw new Error('zbl: factorCoul must scale the pair energy and force');
    }
  });
});

describe('zbl: errors', () => {
  it('missing coefficients for a cross pair throw', () => {
    const s: Pair = new PairZBL();
    s.settings(['2.0', '3.2'], ctx);
    s.allocate(2);
    s.coeff('1 1 29.0 29.0'.split(/\s+/), ctx);
    expect(() => s.init(ctx)).toThrow(/all pair coeffs are not set \(pair 1 2\)/);
  });

  it('Z_i != Z_j on an i==j pair throws', () => {
    const s: Pair = new PairZBL();
    s.settings(['2.0', '3.2'], ctx);
    s.allocate(2);
    s.coeff('1 1 29.0 14.0'.split(/\s+/), ctx);
    expect(() => s.init(ctx)).toThrow(/Z_i must equal Z_j/);
  });

  it('pair_coeff needs exactly I J z_i z_j', () => {
    const s: Pair = new PairZBL();
    s.settings(['2.0', '3.2'], ctx);
    s.allocate(2);
    expect(() => s.coeff('1 1 29.0'.split(/\s+/), ctx)).toThrow(/usage: pair_coeff I J z_i z_j/);
    expect(() => s.coeff('1 1 29.0 29.0 3.0'.split(/\s+/), ctx)).toThrow(/usage: pair_coeff I J z_i z_j/);
    expect(() => s.coeff('1 1 abc 29.0'.split(/\s+/), ctx)).toThrow(/number for z_i/);
  });

  it('pair_style needs inner < outer with inner > 0', () => {
    expect(() => make(['2.0'], [])).toThrow(/usage: pair_style zbl inner outer/);
    expect(() => make(['0.0', '3.2'], ORACLE_COEFFS)).toThrow(/inner cutoff must be > 0/);
    expect(() => make(['3.2', '3.2'], ORACLE_COEFFS)).toThrow(/must be less than outer/);
    expect(() => make(['4.0', '3.2'], ORACLE_COEFFS)).toThrow(/must be less than outer/);
  });

  it('pair_modify shift and tail are not supported', () => {
    const shift = make(['2.0', '3.2'], ORACLE_COEFFS);
    shift.shift = true;
    expect(() => shift.init(ctx)).toThrow(/shift yes is not supported for pair style zbl/);
    const tail = make(['2.0', '3.2'], ORACLE_COEFFS);
    tail.tail = true;
    expect(() => tail.init(ctx)).toThrow(/tail yes is not supported for pair style zbl/);
  });
});
