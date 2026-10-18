/**
 * src/engine-gpl/libm/acos.ts reproduces the acos() of the native binary's C library (glibc 2.35 __ieee754_acos from
 * sysdeps/ieee754/dbl-64/e_asin.c, in the FMA build that the ifunc selects on this CPU) bit for bit. The fixture holds
 * native values written with %.17g by a C program linked against that libm: 3200 arguments, 400 in each branch of the
 * function (|x| < 2^-55, the polynomial below 2^-3, the five table ranges up to 0.96875 and the square-root branch up
 * to 1, half of those within 2^-52 of +-1), with both signs. A 4,950,032-argument comparison against the same libm
 * found no difference; Math.acos differs from it on about 4% of the arguments.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { acos } from '../src/engine-gpl/libm/acos';

const fx = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'libm', 'glibc_acos.json'), 'utf8')) as { acos: [number, number][] };

describe('glibc acos port', () => {
  it('matches the native libm for every fixture argument', () => {
    expect(fx.acos.filter(([x, y]) => !Object.is(acos(x), y)).slice(0, 5)).toEqual([]);
    expect(fx.acos.length).toBe(3200);
  });
  it('differs from Math.acos on some of them (what the port is for)', () => {
    expect(fx.acos.filter(([x, y]) => !Object.is(Math.acos(x), y)).length).toBeGreaterThan(20);
  });
  it('special values', () => {
    expect(acos(1)).toBe(0);
    expect(acos(-1)).toBe(Math.PI);
    expect(acos(0)).toBe(Math.PI / 2);
    expect(acos(-0)).toBe(Math.PI / 2);
    expect(acos(1.0000000000000002)).toBeNaN();
    expect(acos(-2)).toBeNaN();
    expect(acos(NaN)).toBeNaN();
    expect(acos(Infinity)).toBeNaN();
  });
});
