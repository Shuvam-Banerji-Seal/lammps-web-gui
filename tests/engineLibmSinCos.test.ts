/**
 * src/engine-gpl/libm/sincos.ts reproduces the sincos() of the native binary's C library (glibc 2.35, the IBM accurate
 * routines of s_sin.c, built without FMA contraction) bit for bit. The fixture holds native values written with %.17g
 * by a C program linked against that libm, which computed sin(x) and cos(x) together (gcc merges them into sincos());
 * sin() and cos() alone are the FMA builds, tests/engineLibmSinCosFma.test.ts: 3000 arguments over angles of a few turns, 2 pi u, small
 * values, the reduce_sincos range, the __branred range (up to 1e22), random doubles of either sign over the whole
 * exponent range and values in (-3, 3).
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { cos, sin } from '../src/engine-gpl/libm/sincos';

const fx = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'libm', 'glibc_sincos.json'), 'utf8')) as { sincos: [number, number, number][] };

describe('glibc sin and cos port', () => {
  it('matches the native libm for every fixture argument', () => {
    expect(fx.sincos.filter(([x, s]) => !Object.is(sin(x), s)).slice(0, 5)).toEqual([]);
    expect(fx.sincos.filter(([x, , c]) => !Object.is(cos(x), c)).slice(0, 5)).toEqual([]);
    expect(fx.sincos.length).toBe(3000);
  });
  it('differs from Math.sin / Math.cos on some of them (what the port is for)', () => {
    expect(fx.sincos.filter(([x, s, c]) => !Object.is(Math.sin(x), s) || !Object.is(Math.cos(x), c)).length).toBeGreaterThan(10);
  });
  it('special values', () => {
    expect(Object.is(sin(-0), -0)).toBe(true);
    expect(cos(0)).toBe(1);
    expect(sin(Infinity)).toBeNaN();
    expect(cos(NaN)).toBeNaN();
  });
});
