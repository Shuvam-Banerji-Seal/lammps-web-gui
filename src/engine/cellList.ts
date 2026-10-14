/*
 * Cell list over atom positions, built once per compute or force call, so that a neighbour search visits the atoms
 * near a point instead of every atom (the all-atom scan is O(N) per atom, O(N^2) per call).
 *
 * Every cell is at least `cut` wide in each dimension, so an atom within `cut` of a point lies in the 3 x 3 x 3 cells
 * around the point's cell. candidates() returns those atoms in ascending index order. A caller that tests each
 * candidate with its original expression therefore sees the same neighbours, in the same order, as an all-atom loop
 * "for (k = 0; k < nall; k++)", so the floating-point sums it builds are bit-identical.
 *
 * Periodic dimensions (the `period` argument) wrap positions into [0, L) and index the cells modulo the cell count;
 * only an orthogonal box may use them (a triclinic caller passes cut 0, which gives one cell with every atom).
 * Cell widths are 1e-9 relative wider than `cut`, so rounding in the cell index never moves an atom two cells away.
 */

/** Periodic lengths and flags of an orthogonal box (Geometry.lx/ly/lz and Geometry.periodic). */
export interface CellPeriod {
  length: readonly [number, number, number];
  periodic: readonly [boolean, boolean, boolean];
}

/** Cells per atom budget: the cell count is kept at most 2 * nall + 64 by widening the cells. */
const CELLS_PER_ATOM = 2;

export class CellList {
  private readonly n = [1, 1, 1];
  private readonly h = [1, 1, 1];
  private readonly o = [0, 0, 0];
  private readonly per = [false, false, false];
  private readonly len = [0, 0, 0];
  /** Atom indices grouped by cell (cell c holds items[start[c] .. start[c+1]), ascending within a cell). */
  private readonly start: Int32Array;
  private readonly items: Int32Array;
  private readonly out: Int32Array;
  private readonly sx = new Int32Array(3);
  private readonly sy = new Int32Array(3);
  private readonly sz = new Int32Array(3);

  /**
   * Bins the nall positions of xa (3 doubles per atom). `cut` is the largest distance any caller will test; a
   * non-positive or non-finite cut gives a single cell holding every atom.
   */
  constructor(xa: Float64Array, nall: number, cut: number, period?: CellPeriod) {
    const useCut = cut > 0 && Number.isFinite(cut);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let d = 0; d < 3; d++) {
      if (period && period.periodic[d] && period.length[d] > 0) {
        this.per[d] = true;
        this.len[d] = period.length[d];
      }
    }
    if (nall > 0) {
      for (let k = 0; k < nall; k++) {
        for (let d = 0; d < 3; d++) {
          const v = xa[3 * k + d];
          if (v < lo[d]) lo[d] = v;
          if (v > hi[d]) hi[d] = v;
        }
      }
    }
    for (let d = 0; d < 3; d++) {
      if (this.per[d]) {
        this.o[d] = 0;
      } else {
        this.o[d] = nall > 0 ? lo[d] : 0;
        this.len[d] = nall > 0 ? hi[d] - lo[d] : 0;
      }
    }
    // widen the cells until the cell count is within budget (wider cells stay correct, just hold more atoms)
    const budget = CELLS_PER_ATOM * nall + 64;
    let f = 1;
    for (;;) {
      let total = 1;
      for (let d = 0; d < 3; d++) {
        const target = useCut ? cut * (1 + 1e-9) * f : this.len[d] + 1;
        if (this.per[d]) {
          this.n[d] = Math.max(1, Math.floor(this.len[d] / target));
          this.h[d] = this.len[d] / this.n[d];
        } else {
          this.n[d] = Math.floor(this.len[d] / target) + 1;
          this.h[d] = target;
        }
        total *= this.n[d];
      }
      if (!useCut || total <= budget) break;
      f *= 1.2;
    }
    const ncell = this.n[0] * this.n[1] * this.n[2];
    this.start = new Int32Array(ncell + 1);
    this.items = new Int32Array(nall);
    this.out = new Int32Array(nall);
    const cellOf = new Int32Array(nall);
    for (let k = 0; k < nall; k++) {
      const c = this.cellIndex(xa[3 * k], xa[3 * k + 1], xa[3 * k + 2]);
      cellOf[k] = c;
      this.start[c + 1]++;
    }
    for (let c = 0; c < ncell; c++) this.start[c + 1] += this.start[c];
    const fill = this.start.slice(0, ncell);
    for (let k = 0; k < nall; k++) this.items[fill[cellOf[k]]++] = k;
  }

  /** Cell coordinate along dimension d (wrapped for a periodic dimension, clamped into the cell range). */
  private coord(d: number, v: number): number {
    let w = v;
    if (this.per[d]) {
      const L = this.len[d];
      w = v - L * Math.floor(v / L);
    }
    const c = Math.floor((w - this.o[d]) / this.h[d]);
    return c < 0 ? 0 : c >= this.n[d] ? this.n[d] - 1 : c;
  }

  private cellIndex(x: number, y: number, z: number): number {
    return (this.coord(2, z) * this.n[1] + this.coord(1, y)) * this.n[0] + this.coord(0, x);
  }

  /** Cells along dimension d that can hold an atom within the cut of a point in cell c; returns their count. */
  private span(d: number, c: number, s: Int32Array): number {
    const n = this.n[d];
    if (n <= 3) {
      for (let t = 0; t < n; t++) s[t] = t;
      return n;
    }
    if (this.per[d]) {
      s[0] = (c + n - 1) % n; s[1] = c; s[2] = (c + 1) % n;
      return 3;
    }
    let m = 0;
    for (let t = c - 1; t <= c + 1; t++) if (t >= 0 && t < n) s[m++] = t;
    return m;
  }

  /**
   * The atoms in the cells around (x, y, z), ascending by index. The array is reused by the next call: read it before
   * querying again. It contains every atom within the cut of the point (and possibly some further ones).
   */
  candidates(x: number, y: number, z: number): Int32Array {
    const cx = this.coord(0, x), cy = this.coord(1, y), cz = this.coord(2, z);
    const nx = this.n[0], ny = this.n[1];
    const sx = this.sx, sy = this.sy, sz = this.sz;
    const mx = this.span(0, cx, sx), my = this.span(1, cy, sy), mz = this.span(2, cz, sz);
    let m = 0;
    const out = this.out, start = this.start, items = this.items;
    for (let a = 0; a < mz; a++) {
      for (let b = 0; b < my; b++) {
        const row = (sz[a] * ny + sy[b]) * nx;
        for (let c = 0; c < mx; c++) {
          const cell = row + sx[c];
          for (let p = start[cell], e = start[cell + 1]; p < e; p++) out[m++] = items[p];
        }
      }
    }
    const res = out.subarray(0, m);
    res.sort();
    return res;
  }
}
