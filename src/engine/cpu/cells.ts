import type { SimState } from '../types';

/**
 * The cell along a periodic dimension of nc cells (sc = nc / len per unit length) of an atom at offset dx from the box's
 * lower bound. Inside the box as a plain floor, an atom on hi after rounding in the last cell; an atom outside it (it
 * has moved since the last periodic wrap, which the accelerated run loop does on the reneighboring steps only,
 * run/accel.ts) in the cell of its periodic image.
 */
export const periodicCell = (dx: number, len: number, sc: number, nc: number): number => {
  let c = Math.floor(dx * sc);
  if (dx < 0 || dx >= len) {
    c %= nc;
    if (c < 0) c += nc;
    return c === nc ? 0 : c;
  }
  return c >= nc ? nc - 1 : c;
};

/**
 * Atoms counting-sorted by cell: cells of edge >= `rc` (nc per dimension,
 * 1 along z in 2D), `start[c]..start[c+1]` the sorted slots of cell c,
 * `sorted[slot]` the original atom index, `xs`/`ts` positions and types in
 * slot order. Cell index = (cz * ncy + cy) * ncx + cx.
 */
export interface CellSort {
  nc: [number, number, number];
  L: [number, number, number];
  ncell: number;
  start: Int32Array;
  sorted: Int32Array;
  xs: Float64Array;
  ts: Int32Array;
}

/** Cell counts for cutoff rc; null when some used dimension has < 3 cells. */
export const cellGrid = (s: SimState, rc: number): { nc: [number, number, number]; L: [number, number, number] } | null => {
  const two = s.dimension === 2;
  const L: [number, number, number] = [0, 1, 2].map((d) => s.box.hi[d] - s.box.lo[d]) as [number, number, number];
  const nc = L.map((l, d) => (d === 2 && two ? 1 : Math.floor(l / rc))) as [number, number, number];
  if (nc.some((c, d) => !(d === 2 && two) && c < 3)) return null;
  return { nc, L };
};

/** Sorts the atoms of `s` into the cells of `grid`, reusing `into` buffers when large enough. */
export const sortIntoCells = (
  s: SimState,
  grid: { nc: [number, number, number]; L: [number, number, number] },
  into?: CellSort,
): CellSort => {
  const { n, x, type } = s;
  const { nc, L } = grid;
  const two = s.dimension === 2;
  const ncell = nc[0] * nc[1] * nc[2];
  const out: CellSort = into && into.sorted.length >= n && into.start.length >= ncell + 1
    ? { ...into, nc, L, ncell }
    : {
      nc, L, ncell,
      start: new Int32Array(Math.max(ncell + 1, into?.start.length ?? 0)),
      sorted: new Int32Array(n), xs: new Float64Array(3 * n), ts: new Int32Array(n),
    };
  const { start, sorted, xs, ts } = out;
  start.fill(0, 0, ncell + 1);
  const cellOf = new Int32Array(n);
  const [lx, ly, lz] = s.box.lo;
  const sx = nc[0] / L[0], sy = nc[1] / L[1], sz = nc[2] / L[2];
  // a fully periodic box (cpu/parallel.ts)
  for (let i = 0; i < n; i++) {
    const cx = periodicCell(x[3 * i] - lx, L[0], sx, nc[0]);
    const cy = periodicCell(x[3 * i + 1] - ly, L[1], sy, nc[1]);
    const cz = two ? 0 : periodicCell(x[3 * i + 2] - lz, L[2], sz, nc[2]);
    const c = (cz * nc[1] + cy) * nc[0] + cx;
    cellOf[i] = c;
    start[c + 1]++;
  }
  for (let c = 0; c < ncell; c++) start[c + 1] += start[c];
  const fill = start.slice(0, ncell);
  for (let i = 0; i < n; i++) {
    const slot = fill[cellOf[i]]++;
    sorted[slot] = i;
    xs[3 * slot] = x[3 * i]; xs[3 * slot + 1] = x[3 * i + 1]; xs[3 * slot + 2] = x[3 * i + 2];
    ts[slot] = type[i];
  }
  return out;
};
