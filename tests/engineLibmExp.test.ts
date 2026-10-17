/**
 * src/engine-gpl/libm/exp.ts reproduces the exp() of the native binary's C library (glibc 2.35, __exp_fma on an FMA
 * CPU) bit for bit. The fixture holds native values written with %.17g by a C program linked against that libm: 3500
 * arguments in (-10, 10), (-40, 0), within 1e-6 of 0, the overflow special case (500, 720), the subnormal results
 * (-750, -700), random doubles of either sign up to 2^17 and values in (-700, 700). Math.exp misses some of them.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { exp } from '../src/engine-gpl/libm/exp';

const fx = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'libm', 'glibc_exp.json'), 'utf8')) as { exp: [number, number | string][] };
const want = (y: number | string): number => (y === 'Infinity' ? Infinity : (y as number));

describe('glibc exp port', () => {
  it('matches the native libm for every fixture argument', () => {
    expect(fx.exp.filter(([x, y]) => !Object.is(exp(x), want(y))).slice(0, 5)).toEqual([]);
    expect(fx.exp.length).toBe(3500);
  });
  it('differs from Math.exp on some of them (what the port is for)', () => {
    expect(fx.exp.filter(([x, y]) => !Object.is(Math.exp(x), want(y))).length).toBeGreaterThan(10);
  });
  it('special values', () => {
    expect(exp(0)).toBe(1);
    expect(exp(-Infinity)).toBe(0);
    expect(exp(Infinity)).toBe(Infinity);
    expect(exp(NaN)).toBeNaN();
    expect(exp(800)).toBe(Infinity);
    expect(exp(-800)).toBe(0);
  });
});
