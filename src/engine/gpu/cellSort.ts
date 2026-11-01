import type { SimState } from '../types';
import { periodicCell } from '../cpu/cells';

/*
 * The cell list the GPU kernels scan (webgpuForces.ts and its kernels): built on the CPU for every force evaluation,
 * cells of edge >= the largest cutoff, the atoms counting-sorted by cell, positions in fp32 from the box origin.
 */

/** The kernels' box length along a non-periodic dimension: their minimum image d - L round(d / L) is then d. */
export const NO_IMAGE = 1e30;

/** Scratch of the sort, grown on demand and kept by the backend. */
export interface CellScratch {
  /** The atom of each slot. */
  sorted: Int32Array;
  cellOfAtom: Int32Array;
  /** Per slot the fp32 position from the box origin, then the type. */
  posData: Float32Array;
  /** The cell starts (ncell + 1), the cell of each slot (n), then the caller's `extra` words. */
  intsData: Uint32Array;
}

export interface CellGrid {
  /** The box lengths of the kernels' minimum image (NO_IMAGE along a non-periodic dimension). */
  L: number[];
  nc: number[];
  ncell: number;
  /** Offset of the per-slot cells in intsData. */
  base: number;
  two: boolean;
}

export const newCellScratch = (): CellScratch => ({
  sorted: new Int32Array(0), cellOfAtom: new Int32Array(0), posData: new Float32Array(0), intsData: new Uint32Array(0),
});

/**
 * The counting sort by cell (the same assignment as cpu/cells.ts) of the force evaluations: sorted[slot] the
 * atom of each slot, posData its fp32 position from the box origin and type, intsData the cell starts (ncell + 1) then
 * the cell of each slot (n), with `extra` more words after them. Null when a periodic dimension has fewer than minCells
 * cells: 3 for the 27-cell scan of compute(); the scan of KERNEL_STYLES takes every cell once along a shorter edge.
 */
export const sortByCell = (buf: CellScratch, state: SimState, maxCutoff: number, extra: number, minCells = 3): CellGrid | null => {
  const { n, x, type } = state;
  const two = state.dimension === 2;
  const per = state.box.periodic;
  const len = [0, 1, 2].map((d) => state.box.hi[d] - state.box.lo[d]);
  // a non-periodic dimension has at least one cell, and the kernels' scan takes each cell once along an edge of fewer
  // than 3 cells (no image to find there)
  const nc = [0, 1, 2].map((d) => (d === 2 && two ? 1 : per[d] ? Math.floor(len[d] / maxCutoff) : Math.max(1, Math.floor(len[d] / maxCutoff))));
  if (nc.some((c, d) => !(d === 2 && two) && per[d] && c < minCells)) return null;
  const ncell = nc[0] * nc[1] * nc[2];
  if (buf.sorted.length < n) {
    buf.sorted = new Int32Array(n);
    buf.cellOfAtom = new Int32Array(n);
    buf.posData = new Float32Array(4 * n);
  }
  if (buf.intsData.length < ncell + 1 + n + extra) buf.intsData = new Uint32Array(ncell + 1 + n + extra);
  const { sorted, cellOfAtom, posData } = buf;
  const ints = buf.intsData;
  const base = ncell + 1;
  ints.fill(0, 0, base);
  const lo = state.box.lo;
  const s = [nc[0] / len[0], nc[1] / len[1], nc[2] / len[2]];
  for (let i = 0; i < n; i++) {
    let c = 0;
    for (let d = 2; d >= 0; d--) {
      let k = 0;
      if (!(d === 2 && two)) {
        // periodic: an atom outside the box (it has moved since the last wrap, run/accel.ts) in its image's cell;
        // non-periodic: one past an edge (a shrink-wrapped box between its resets, an atom about to be lost) in the
        // edge cell, which keeps every partner within the cutoff in the cells the scan takes
        const dx = x[3 * i + d] - lo[d];
        if (per[d]) k = periodicCell(dx, len[d], s[d], nc[d]);
        else {
          k = Math.floor(dx * s[d]);
          if (k >= nc[d]) k = nc[d] - 1; else if (k < 0) k = 0;
        }
      }
      c = c * nc[d] + k;
    }
    cellOfAtom[i] = c;
    ints[c + 1]++;
  }
  for (let c = 0; c < ncell; c++) ints[c + 1] += ints[c];
  const fill = ints.slice(0, ncell);
  for (let i = 0; i < n; i++) {
    const c = cellOfAtom[i];
    const slot = fill[c]++;
    sorted[slot] = i;
    ints[base + slot] = c;
    posData[4 * slot] = x[3 * i] - lo[0];
    posData[4 * slot + 1] = x[3 * i + 1] - lo[1];
    posData[4 * slot + 2] = two ? 0 : x[3 * i + 2] - lo[2];
    posData[4 * slot + 3] = type[i];
  }
  // the kernels' minimum image, d - L round(d / L): the identity along a non-periodic dimension
  const L = len.map((l, d) => (per[d] || (d === 2 && two) ? l : NO_IMAGE));
  return { L, nc, ncell, base, two };
};
