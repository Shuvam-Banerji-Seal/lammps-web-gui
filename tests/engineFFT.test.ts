import { describe, expect, it } from 'vitest';
import { fft1d, fft3d, isFFTSize, nextFFTSize } from '../src/engine/force/fft';

const dft = (re: number[], im: number[], sign: number) => {
  const n = re.length;
  const or = new Array(n).fill(0), oi = new Array(n).fill(0);
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      const a = (sign * 2 * Math.PI * j * k) / n;
      or[k] += re[j] * Math.cos(a) - im[j] * Math.sin(a);
      oi[k] += re[j] * Math.sin(a) + im[j] * Math.cos(a);
    }
  }
  return [or, oi];
};

describe('mixed-radix FFT', () => {
  it('matches a direct DFT for sizes made of 2, 3 and 5', () => {
    for (const n of [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 45, 48, 60, 64, 75, 90]) {
      const re = Array.from({ length: n }, (_, j) => Math.sin(1.3 * j + 0.2) + 0.1 * j);
      const im = Array.from({ length: n }, (_, j) => Math.cos(0.7 * j * j) - 0.3);
      for (const sign of [-1, 1] as const) {
        const d = new Float64Array(2 * n);
        for (let j = 0; j < n; j++) { d[2 * j] = re[j]; d[2 * j + 1] = im[j]; }
        fft1d(d, 0, 1, n, sign);
        const [or, oi] = dft(re, im, sign);
        for (let k = 0; k < n; k++) {
          expect(d[2 * k]).toBeCloseTo(or[k], 9);
          expect(d[2 * k + 1]).toBeCloseTo(oi[k], 9);
        }
      }
    }
  });
  it('3d forward then inverse returns the input times N', () => {
    const [nx, ny, nz] = [6, 5, 4];
    const n = nx * ny * nz;
    const d = new Float64Array(2 * n);
    for (let k = 0; k < 2 * n; k++) d[k] = Math.sin(0.37 * k) + 0.01 * k;
    const ref = Float64Array.from(d);
    fft3d(d, nx, ny, nz, -1);
    fft3d(d, nx, ny, nz, 1);
    for (let k = 0; k < 2 * n; k++) expect(d[k] / n).toBeCloseTo(ref[k], 10);
  });
  it('size helpers', () => {
    expect(isFFTSize(7)).toBe(false);
    expect(nextFFTSize(7)).toBe(8);
    expect(nextFFTSize(31)).toBe(32);
    expect(nextFFTSize(49)).toBe(50);
  });
});
