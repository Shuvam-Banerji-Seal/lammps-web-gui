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
import { cos, sin, sincos, SINCOS } from '../src/engine-gpl/libm/sincos';

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
  it('sincos (one argument reduction, as __sincos) gives sin and cos bit for bit, over every branch', () => {
    // the fixture's native values, then arguments around every branch boundary of s_sin.c and random ones
    for (const [x, s, c] of fx.sincos) {
      sincos(x);
      expect(Object.is(SINCOS[0], s) && Object.is(SINCOS[1], c), `x = ${x}`).toBe(true);
    }
    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const xs: number[] = [0, -0, 1e-300, 2 ** -27, 0.855469, 2.426265, 105414350, -105414350, 1e22, -3e300, Infinity, -Infinity, NaN];
    for (let k = 0; k < 20000; k++) {
      const m = 10 ** (rnd() * 12 - 4) * (rnd() < 0.5 ? -1 : 1);
      xs.push(m, 2 * Math.PI * rnd() * 200, (rnd() - 0.5) * 6);
    }
    for (let k = 0; k < 2000; k++) xs.push((rnd() - 0.5) * 10 ** (rnd() * 300));
    let bad = 0;
    for (const x of xs) {
      sincos(x);
      if (!Object.is(SINCOS[0], sin(x)) || !Object.is(SINCOS[1], cos(x))) bad++;
    }
    expect(bad).toBe(0);
  });
  it('special values', () => {
    expect(Object.is(sin(-0), -0)).toBe(true);
    expect(cos(0)).toBe(1);
    expect(sin(Infinity)).toBeNaN();
    expect(cos(NaN)).toBeNaN();
  });
});
