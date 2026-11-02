import { describe, expect, it } from 'vitest';
import { buildRows, RowCandidates, type SparseRows } from '../src/engine-gpl/reaxff/charge';

/*
 * The candidate partners of the charge-equilibration matrices (charge.ts RowCandidates), kept between steps as a Verlet
 * list over cut + skin: while no atom has moved more than skin / 2 and the neighbor build is the same, the rows built
 * from them are those of a scan over every j > i (the same pairs, values and order); otherwise they are made again.
 */

/** n owned atoms in a 20 Å cube and their periodic images within 4 Å of its faces as ghosts. */
const config = (seed: number) => {
  let r = seed;
  const rnd = () => {
    r = (r * 1103515245 + 12345) % 2147483648;
    return r / 2147483648;
  };
  const n = 300, L = 20;
  const xs: number[] = [];
  for (let i = 0; i < n; i++) xs.push(L * rnd(), L * rnd(), L * rnd());
  const owner: number[] = [];
  for (let i = 0; i < n; i++) owner.push(i);
  for (let i = 0; i < n; i++) {
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      if (a === 0 && b === 0 && c === 0) continue;
      const gx = xs[3 * i] + a * L, gy = xs[3 * i + 1] + b * L, gz = xs[3 * i + 2] + c * L;
      if (gx > -4 && gx < L + 4 && gy > -4 && gy < L + 4 && gz > -4 && gz < L + 4) {
        xs.push(gx, gy, gz);
        owner.push(i);
      }
    }
  }
  const N = owner.length;
  const orig = Int32Array.from(owner, (o) => o + 1);
  return { n, N, x: Float64Array.from(xs), orig, owner, rnd };
};

const value = (i: number, j: number, r: number) => 1 / (r + 0.01 * ((i + 3 * j) % 7));
const rows = (m: SparseRows, n: number) => ({ start: Array.from(m.start.subarray(0, n + 1)), j: Array.from(m.jlist), v: Array.from(m.val) });

describe('charge.ts RowCandidates: the rows of the QEq matrices from candidates kept between steps', () => {
  it('give the rows of a full scan while atoms move less than skin / 2, and are made again past it', () => {
    const { n, N, x, orig, owner, rnd } = config(7);
    const inGroup = new Uint8Array(N).fill(1);
    const cut = 3.5, skin = 1.0;
    const cands = new RowCandidates();
    cands.update(x, n, N, cut, skin, 1);
    expect(cands.made).toBe(1);
    expect(rows(buildRows(x, n, N, orig, inGroup, cut, value, cands), n)).toEqual(rows(buildRows(x, n, N, orig, inGroup, cut, value), n));
    // owned atoms move by up to 0.28 Å each (ghosts with their owners): the candidates stay
    const moved = Float64Array.from(x);
    for (let i = 0; i < n; i++) {
      const d = [0.32 * (rnd() - 0.5), 0.32 * (rnd() - 0.5), 0.32 * (rnd() - 0.5)];
      for (let a = 0; a < N; a++) if (owner[a] === i) for (let k = 0; k < 3; k++) moved[3 * a + k] += d[k];
    }
    cands.update(moved, n, N, cut, skin, 1);
    expect(cands.made).toBe(1);
    const kept = rows(buildRows(moved, n, N, orig, inGroup, cut, value, cands), n);
    expect(kept).toEqual(rows(buildRows(moved, n, N, orig, inGroup, cut, value), n));
    expect(kept.j.length).toBeGreaterThan(500);
    // one atom moves by 0.6 Å (more than skin / 2): they are made again, and the rows are still the full scan's
    moved[0] += 0.6;
    cands.update(moved, n, N, cut, skin, 1);
    expect(cands.made).toBe(2);
    expect(rows(buildRows(moved, n, N, orig, inGroup, cut, value, cands), n)).toEqual(rows(buildRows(moved, n, N, orig, inGroup, cut, value), n));
    // another neighbor build, another cutoff or another skin: made again
    cands.update(moved, n, N, cut, skin, 2);
    expect(cands.made).toBe(3);
    cands.update(moved, n, N, 3.0, skin, 2);
    expect(cands.made).toBe(4);
    cands.update(moved, n, N, 3.0, 0.5, 2);
    expect(cands.made).toBe(5);
    cands.update(moved, n, N, 3.0, 0.5, 2);
    expect(cands.made).toBe(5);
  });

  it('with skin 0 they are made for every new set of positions', () => {
    const { n, N, x } = config(11);
    const cands = new RowCandidates();
    cands.update(x, n, N, 3.0, 0, 1);
    cands.update(x, n, N, 3.0, 0, 1);
    expect(cands.made).toBe(1);
    const moved = Float64Array.from(x);
    moved[5] += 1e-9;
    cands.update(moved, n, N, 3.0, 0, 1);
    expect(cands.made).toBe(2);
  });
});
