import { describe, expect, it } from 'vitest';
import { CellList } from '../src/engine/cellList';

/* The cell list must return every atom within the cut of a point, in ascending index order, with no duplicates. */

/** Deterministic pseudo-random numbers in [0, 1). */
const rng = (seed: number) => {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
};

const randomPositions = (n: number, lo: number, hi: number, seed: number): Float64Array => {
  const r = rng(seed);
  const x = new Float64Array(3 * n);
  for (let k = 0; k < 3 * n; k++) x[k] = lo + (hi - lo) * r();
  return x;
};

/** Brute-force reference: the same all-atom scan the engine used before the cell list. */
const within = (x: Float64Array, n: number, i: number, cut: number, minimage?: (d: number[]) => void): number[] => {
  const out: number[] = [];
  for (let k = 0; k < n; k++) {
    if (k === i) continue;
    const d = [x[3 * i] - x[3 * k], x[3 * i + 1] - x[3 * k + 1], x[3 * i + 2] - x[3 * k + 2]];
    if (minimage) minimage(d);
    if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < cut * cut) out.push(k);
  }
  return out;
};

const assertSortedUnique = (list: Int32Array) => {
  for (let t = 1; t < list.length; t++) expect(list[t]).toBeGreaterThan(list[t - 1]);
};

describe('CellList candidates', () => {
  it('covers every atom within the cut (open box, points outside the box too)', () => {
    const n = 400;
    const x = randomPositions(n, -2, 12, 7);
    for (const cut of [0.5, 1.7, 3.1]) {
      const cells = new CellList(x, n, cut);
      for (let i = 0; i < n; i++) {
        const cand = cells.candidates(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        assertSortedUnique(cand);
        const set = new Set(cand);
        for (const k of within(x, n, i, cut)) expect(set.has(k)).toBe(true);
      }
      // query points outside the bounding box
      for (const q of [[-5, 3, 3], [20, 20, -4], [5, 5, 40]]) {
        const cand = cells.candidates(q[0], q[1], q[2]);
        assertSortedUnique(cand);
        const set = new Set(cand);
        for (let k = 0; k < n; k++) {
          const d2 = (x[3 * k] - q[0]) ** 2 + (x[3 * k + 1] - q[1]) ** 2 + (x[3 * k + 2] - q[2]) ** 2;
          if (d2 < cut * cut) expect(set.has(k)).toBe(true);
        }
      }
    }
  });

  it('covers every minimum image neighbour in a periodic orthogonal box', () => {
    const n = 300;
    const L: [number, number, number] = [7, 9, 5];
    // positions outside [0, L) exercise the wrapping
    const x = randomPositions(n, -10, 20, 11);
    const minimage = (d: number[]) => {
      for (let c = 0; c < 3; c++) {
        if (Math.abs(d[c]) > 0.5 * L[c]) d[c] -= Math.round(d[c] / L[c]) * L[c];
      }
    };
    for (const cut of [1.2, 2.5, 4.4]) {
      const cells = new CellList(x, n, cut, { length: L, periodic: [true, true, true] });
      for (let i = 0; i < n; i++) {
        const cand = cells.candidates(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
        assertSortedUnique(cand);
        const set = new Set(cand);
        for (const k of within(x, n, i, cut, minimage)) expect(set.has(k)).toBe(true);
      }
    }
  });

  it('gives every atom in one cell when the cut is not positive', () => {
    const n = 25;
    const x = randomPositions(n, 0, 3, 3);
    const cells = new CellList(x, n, 0);
    const cand = cells.candidates(1, 1, 1);
    expect(Array.from(cand)).toEqual(Array.from({ length: n }, (_, k) => k));
  });

  it('handles an empty atom set', () => {
    const cells = new CellList(new Float64Array(0), 0, 2);
    expect(cells.candidates(0, 0, 0).length).toBe(0);
  });
});
