/**
 * src/engine-gpl/libm/cbrt.ts reproduces the cbrt() of the native binary's C library (glibc 2.35 s_cbrt.c, which the
 * installed libm.so.6 compiles without fused multiply-adds) bit for bit. The fixture holds native values written with
 * %.17g by a C program linked against that libm: 3100 arguments (box volumes up to 1e6, random magnitudes 2^-200 to
 * 2^200, negative values, (1, 8) and subnormals). A 2,000,000-argument comparison against the same libm found no
 * difference.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { cbrt } from '../src/engine-gpl/libm/cbrt';

const fx = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'libm', 'glibc_cbrt.json'), 'utf8')) as { cbrt: [number, number][] };

describe('glibc cbrt port', () => {
  it('matches the native libm for every fixture argument', () => {
    expect(fx.cbrt.filter(([x, y]) => !Object.is(cbrt(x), y)).slice(0, 5)).toEqual([]);
    expect(fx.cbrt.length).toBe(3100);
  });
  it('differs from Math.cbrt on some of them (what the port is for)', () => {
    expect(fx.cbrt.filter(([x, y]) => !Object.is(Math.cbrt(x), y)).length).toBeGreaterThan(10);
  });
  it('special values', () => {
    expect(Object.is(cbrt(0), 0)).toBe(true);
    expect(Object.is(cbrt(-0), -0)).toBe(true);
    expect(cbrt(Infinity)).toBe(Infinity);
    expect(cbrt(-Infinity)).toBe(-Infinity);
    expect(cbrt(NaN)).toBeNaN();
    // Measured with the native libm: cbrt(27) = 3.0000000000000004 (glibc's cbrt is not correctly rounded), cbrt(8) = 2,
    // cbrt(1000) = 10
    expect(cbrt(27)).toBe(3.0000000000000004);
    expect(cbrt(8)).toBe(2);
    expect(cbrt(1000)).toBe(10);
  });
});
