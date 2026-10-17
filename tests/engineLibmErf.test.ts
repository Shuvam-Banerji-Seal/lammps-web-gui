/**
 * src/engine-gpl/libm/erf.ts reproduces the erf() and erfc() of the native binary's C library (glibc 2.35, s_erf.c,
 * whose two exp calls go to the exact exp port) bit for bit. The fixture holds native values from a C program linked
 * against that libm: 500 arguments from each of six classes (x in (-8, 30), x in (-2, 4), random bit patterns, tiny
 * |x| down to 2^-80, the Wolf/Ewald range 0 < x < 3.72, and arguments within 1e-6 of the 0.84375 branch point), and
 * the special values. 2,000,017 arguments were compared when the port was written: no difference.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { erf, erfc } from '../src/engine-gpl/libm/erf';

type V = number | string;
const fx = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'libm', 'glibc_erf.json'), 'utf8')) as { erf: [V, V, V][] };
const num = (v: V): number => (v === 'NaN' ? NaN : v === 'Infinity' ? Infinity : v === '-Infinity' ? -Infinity : v === '-0' ? -0 : (v as number));
const same = (a: number, b: number): boolean => Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b));

describe('glibc erf and erfc ports', () => {
  it('match the native libm for every fixture argument', () => {
    expect(fx.erf.filter(([x, e]) => !same(erf(num(x)), num(e))).slice(0, 5)).toEqual([]);
    expect(fx.erf.filter(([x, , c]) => !same(erfc(num(x)), num(c))).slice(0, 5)).toEqual([]);
    expect(fx.erf.length).toBe(3017);
  });
  it('special values', () => {
    expect(erfc(0)).toBe(1);
    expect(erf(0)).toBe(0);
    expect(erfc(Infinity)).toBe(0);
    expect(erfc(-Infinity)).toBe(2);
    expect(erf(-Infinity)).toBe(-1);
    expect(erfc(NaN)).toBeNaN();
    expect(erfc(30)).toBe(0);
  });
});
