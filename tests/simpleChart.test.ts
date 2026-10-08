import { describe, it, expect } from 'vitest';
import { decimate, extent, Point } from '../src/components/charts/SimpleChart';

describe('extent', () => {
  it('finds the range without spreading into Math.min/max', () => {
    expect(extent([3, -1, 7, 0])).toEqual({ lo: -1, hi: 7 });
  });

  it('ignores non-finite values', () => {
    expect(extent([1, NaN, 5, Infinity])).toEqual({ lo: 1, hi: 5 });
  });

  it('falls back to 0..1 when there is nothing usable', () => {
    expect(extent([])).toEqual({ lo: 0, hi: 1 });
    expect(extent([NaN, NaN])).toEqual({ lo: 0, hi: 1 });
  });

  it('handles an array far past the argument-spread limit', () => {
    const big = new Array(300_000);
    for (let i = 0; i < big.length; i++) big[i] = i % 1000;
    expect(() => extent(big)).not.toThrow();
    expect(extent(big)).toEqual({ lo: 0, hi: 999 });
  });
});

describe('decimate', () => {
  const ramp = (n: number): Point[] =>
    Array.from({ length: n }, (_, i) => ({ x: i, y: i }));

  it('leaves a short series untouched', () => {
    const d = ramp(100);
    expect(decimate(d)).toBe(d);
  });

  it('caps the point count for a long series', () => {
    const out = decimate(ramp(50_000));
    expect(out.length).toBeLessThanOrEqual(PLOT_BUDGET);
    expect(out.length).toBeGreaterThan(100);
  });

  it('preserves the global minimum and maximum', () => {
    const d = ramp(20_000);
    const out = decimate(d);
    const ys = out.map(p => p.y);
    expect(Math.min(...ys)).toBe(0);
    expect(Math.max(...ys)).toBe(19_999);
  });

  it('keeps a narrow spike that stride sampling would drop', () => {
    // A single-point peak at an index no uniform stride would land on.
    const d = ramp(20_000).map(p => ({ ...p, y: 1 }));
    d[7_331].y = 999;
    const out = decimate(d);
    expect(out.some(p => p.y === 999)).toBe(true);
  });

  it('keeps points in ascending x order so the path does not zig-zag', () => {
    const out = decimate(ramp(30_000));
    for (let i = 1; i < out.length; i++) {
      expect(out[i].x).toBeGreaterThanOrEqual(out[i - 1].x);
    }
  });

  it('keeps a descending series monotonic in x too', () => {
    const d = Array.from({ length: 30_000 }, (_, i) => ({ x: i, y: 30_000 - i }));
    const out = decimate(d);
    for (let i = 1; i < out.length; i++) {
      expect(out[i].x).toBeGreaterThanOrEqual(out[i - 1].x);
    }
    expect(out[0].y).toBeGreaterThan(out[out.length - 1].y);
  });
});

/** decimate()'s default budget: 2 points per plot pixel at 320px. */
const PLOT_BUDGET = 640;
