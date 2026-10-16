/**
 * src/engine-gpl/libm/log.ts reproduces the log() of the native binary's C library (glibc 2.35, __ieee754_log_fma on
 * an FMA CPU) bit for bit, and fma() is a correctly rounded fused multiply-add. The fixture holds native values
 * written with %.17g by a C program linked against that libm (3000 log arguments: RanMars rsq values, random normal
 * doubles over the exponent range, the near-1 interval, subnormals, values within 1e-12 of 1, values in (0, 4); 1000
 * hardware fma results including near-total cancellation). V8's Math.log differs from it in the last bit for a few
 * percent of these arguments, which is why the GPL ports call this log.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fma, log } from '../src/engine-gpl/libm/log';

const fx = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'libm', 'glibc_log.json'), 'utf8')) as { log: [number, number][]; fma: [number, number, number, number][] };

describe('glibc log port', () => {
  it('matches the native libm on every fixture argument', () => {
    const bad = fx.log.filter(([x, y]) => !Object.is(log(x), y));
    expect(bad.slice(0, 5)).toEqual([]);
    expect(fx.log.length).toBe(3000);
  });
  it('differs from Math.log on some of them (what the port is for)', () => {
    expect(fx.log.filter(([x, y]) => !Object.is(Math.log(x), y)).length).toBeGreaterThan(10);
  });
  it('handles the special values as glibc does', () => {
    expect(log(1)).toBe(0);
    expect(Object.is(log(1), 0)).toBe(true);
    expect(log(0)).toBe(-Infinity);
    expect(log(-0)).toBe(-Infinity);
    expect(log(Infinity)).toBe(Infinity);
    expect(log(-1)).toBeNaN();
    expect(log(NaN)).toBeNaN();
    expect(log(Number.MIN_VALUE)).toBe(-744.4400719213812);
  });
});

describe('emulated fma', () => {
  it('is correctly rounded (hardware fma values)', () => {
    const bad = fx.fma.filter(([a, b, c, r]) => !Object.is(fma(a, b, c), r));
    expect(bad.slice(0, 5)).toEqual([]);
  });
});
