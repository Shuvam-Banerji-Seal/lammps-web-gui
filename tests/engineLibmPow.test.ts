/**
 * src/engine-gpl/libm/pow.ts reproduces the pow() of the native binary's C library (glibc 2.35, __ieee754_pow_fma on
 * an FMA CPU) bit for bit. The fixture holds native values from a C program linked against that libm: 400 pairs from
 * each of ten classes (x in (0, 10) with the exponents the styles use, x over e^(+-40) with y in (-50, 50), x near 1
 * with |y| up to 5e5, negative x with integer y, random bit patterns, subnormal x, x in (0.01, 100) with |y| < 200, x
 * over e^(+-700) with |y| < 1.5, slightly perturbed common exponents, results near overflow and underflow), and the
 * special cases (zeros, infinities, NaN, x = +-1, odd and even integer y for negative x, tiny and huge y). On Node 18
 * and 20, Math.pow differs from glibc for about 10 percent of ordinary arguments; the port is what makes the results
 * the same on every JavaScript engine.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pow } from '../src/engine-gpl/libm/pow';
import { fma, fmaF } from '../src/engine-gpl/libm/log';

type V = number | string;
const fx = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'libm', 'glibc_pow.json'), 'utf8')) as { pow: [V, V, V][] };
const num = (v: V): number => (v === 'NaN' ? NaN : v === 'Infinity' ? Infinity : v === '-Infinity' ? -Infinity : v === '-0' ? -0 : (v as number));
const same = (a: number, b: number): boolean => Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b));

describe('glibc pow port', () => {
  it('y = 1.5 (the shortcut that takes the double-double x sqrt(x) when it decides the rounding) matches the native libm', () => {
    // glibc_pow15.json: 3000 (x, pow(x, 1.5)) from the same C program on the native libm: x in [2, 100] (eFF's Pauli
    // ratio), log-uniform over e^(+-320), near 1, perfect squares, random bit patterns and next to powers of two (the
    // shortcut leaves powers of two and near-midpoint values to the full computation). A 20,000,000-pair run of the
    // same classes had no difference.
    const p15 = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'libm', 'glibc_pow15.json'), 'utf8')) as { pow15: [number, number][] };
    const bad = p15.pow15.filter(([x, z]) => { pow(3, 0.25); return !Object.is(pow(x, 1.5), z); });
    expect(bad.slice(0, 5)).toEqual([]);
    expect(p15.pow15.length).toBe(3000);
  });
  it('a repeated argument pair gives the same bits (the kept last result)', () => {
    for (const [x, y] of [[2, 1.5], [2, -4], [5.2, 1.5], [0.7, 3.3]]) expect(Object.is(pow(x, y), pow(x, y))).toBe(true);
    expect(pow(-0, -3)).toBe(-Infinity);
    expect(pow(0, -3)).toBe(Infinity);
    expect(pow(-0, -3)).toBe(-Infinity);
  });
  it('matches the native libm for every fixture pair', () => {
    const bad = fx.pow.filter(([x, y, z]) => !same(pow(num(x), num(y)), num(z)));
    expect(bad.slice(0, 5)).toEqual([]);
    expect(fx.pow.length).toBe(4033);
  });
  it('special values', () => {
    expect(pow(2, 0)).toBe(1);
    expect(pow(NaN, 0)).toBe(1);
    expect(pow(1, NaN)).toBe(1);
    expect(pow(-1, Infinity)).toBe(1);
    expect(pow(0.5, Infinity)).toBe(0);
    expect(pow(2, -Infinity)).toBe(0);
    expect(pow(-0, -3)).toBe(-Infinity);
    expect(pow(-0, 3)).toBe(-0);
    expect(pow(-8, 1 / 3)).toBeNaN();
    expect(pow(-2, 3)).toBe(-8);
    expect(pow(2, 1024)).toBe(Infinity);
    expect(pow(2, -1075)).toBe(0);
    expect(pow(4, 0.5)).toBe(2);
  });
  it('fmaF (the fast fused multiply-add of the fast path) agrees with the round-to-odd fma', () => {
    // xorshift doubles; one in three triples is built so that c cancels most of a b or sits at a halfway point
    let s = 0x9e3779b9;
    const rnd = (): number => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
    let bad = 0;
    for (let k = 0; k < 200000; k++) {
      const a = (rnd() + 0.5) * 2 ** Math.floor(rnd() * 60 - 30) * (rnd() < 0.5 ? -1 : 1);
      const b = (rnd() + 0.5) * 2 ** Math.floor(rnd() * 60 - 30) * (rnd() < 0.5 ? -1 : 1);
      const m = k % 3;
      const c = m === 0 ? (rnd() - 0.5) * 2 ** Math.floor(rnd() * 80 - 40)
        : m === 1 ? -(a * b) * (1 + (rnd() - 0.5) * 2 ** -Math.floor(rnd() * 50))
          : Math.round(rnd() * 1024) * 2 ** Math.floor(rnd() * 40 - 20);
      if (!Object.is(fmaF(a, b, c), fma(a, b, c))) bad++;
    }
    expect(bad).toBe(0);
  });
});
