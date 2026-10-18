/**
 * src/engine-gpl/libm/sincos_fma.ts reproduces the standalone sin() and cos() of the native binary's C library (glibc
 * 2.35 on an x86_64 CPU with FMA: the __sin_fma and __cos_fma builds that the ifunc dispatches to). The fixture holds
 * native values as IEEE-754 bit patterns, for arguments where the standalone sin() or cos() differs from sincos() and
 * for random arguments; 4*pi/15 (sin 0x3fe7c7d7a833bec1) is among them.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { cosFma, sinFma } from '../src/engine-gpl/libm/sincos_fma';
import { cos, sin } from '../src/engine-gpl/libm/sincos';

const fx = JSON.parse(readFileSync(join(__dirname, 'fixtures', 'libm', 'glibc_sin_fma.json'), 'utf8')) as {
  sin_cos: [string, string, string][];
};

const bits = (v: number): string => {
  const f = new Float64Array([v]);
  const u = new BigUint64Array(f.buffer);
  return '0x' + u[0].toString(16).padStart(16, '0');
};

describe('glibc standalone sin and cos (FMA build) port', () => {
  it('matches the native libm for every fixture argument', () => {
    const badS = fx.sin_cos.filter(([x, s]) => bits(sinFma(Number(x))) !== s).slice(0, 5);
    const badC = fx.sin_cos.filter(([x, , c]) => bits(cosFma(Number(x))) !== c).slice(0, 5);
    expect(badS).toEqual([]);
    expect(badC).toEqual([]);
    expect(fx.sin_cos.length).toBeGreaterThan(3000);
  });
  it('4*pi/15: the standalone sin() differs from sincos() and matches the native value', () => {
    const x = 0.8377580409572781;
    expect(bits(sinFma(x))).toBe('0x3fe7c7d7a833bec1');
    expect(bits(sin(x))).toBe('0x3fe7c7d7a833bec2');
  });
  it('special values', () => {
    expect(Object.is(sinFma(-0), -0)).toBe(true);
    expect(cosFma(0)).toBe(1);
    expect(sinFma(Infinity)).toBeNaN();
    expect(cosFma(NaN)).toBeNaN();
  });
  it('is the same mathematics as sincos.ts to within one ulp', () => {
    for (const [x] of fx.sin_cos.slice(0, 500)) {
      expect(Math.abs(sinFma(Number(x)) - sin(Number(x)))).toBeLessThanOrEqual(2.3e-16 * Math.max(1, Math.abs(sin(Number(x)))));
      expect(Math.abs(cosFma(Number(x)) - cos(Number(x)))).toBeLessThanOrEqual(2.3e-16 * Math.max(1, Math.abs(cos(Number(x)))));
    }
  });
});
