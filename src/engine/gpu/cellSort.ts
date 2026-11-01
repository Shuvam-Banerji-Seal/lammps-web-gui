import type { SimState } from '../types';
import { periodicCell } from '../cpu/cells';

/*
 * The cell list the GPU kernels scan (webgpuForces.ts and its kernels): built on the CPU for every force evaluation,
 * the atoms counting-sorted by cell, positions in fp32 from the box origin. The cells divide the box's fractional
 * coordinates (lamda = h^-1 (x - lo), with h the upper-triangular matrix of the edge vectors a = (lx, 0, 0),
 * b = (xy, ly, 0), c = (xz, yz, lz)), nc_d of them along d with the box's width perpendicular to the other two edges at
 * least nc_d cutoffs: two atoms within the cutoff are then in the same or neighbouring cells. In an orthogonal box the
 * widths are the edge lengths.
 */

/** The kernels' box length along a non-periodic dimension: their minimum image is then the identity there. */
export const NO_IMAGE = 1e30;

/**
 * The kernels' minimum image of a separation d (WGSL, for a kernel whose uniform P has L and tilt = (xy, xz, yz, _)):
 * the fractional coordinates of d by back substitution in h, rounded, and that lattice vector subtracted. The image is
 * the one copy of a partner within the cutoff while every periodic width is at least 2 cutoffs (the cutoff sphere then
 * fits in the cell centred on the atom). With tilt 0 it is d - L round(d / L).
 */
export const WGSL_BOX_IMAGE = /* wgsl */ `
fn boxImage(d : vec3<f32>) -> vec3<f32> {
  let L = P.L;
  let t = P.tilt;
  let fz = d.z / L.z;
  let fy = (d.y - t.z * fz) / L.y;
  let fx = (d.x - t.x * fy - t.y * fz) / L.x;
  let k = round(vec3<f32>(fx, fy, fz));
  return d - vec3<f32>(k.x * L.x + k.y * t.x + k.z * t.y, k.y * L.y + k.z * t.z, k.z * L.z);
}
`;

/**
 * The box's widths perpendicular to the other two edges (volume / area of the opposite face), the edge lengths of an
 * orthogonal box. A periodic width of less than 2 cutoffs puts two copies of a partner within the cutoff.
 */
export const boxWidths = (box: SimState['box']): number[] => {
  const lx = box.hi[0] - box.lo[0], ly = box.hi[1] - box.lo[1], lz = box.hi[2] - box.lo[2];
  if (!box.triclinic) return [lx, ly, lz];
  const [xy, xz, yz] = box.tilt;
  // |b x c| = |(ly lz, -xy lz, xy yz - ly xz)|, |c x a| = lx sqrt(lz^2 + yz^2), |a x b| = lx ly
  const v = lx * ly * lz;
  return [v / Math.hypot(ly * lz, xy * lz, xy * yz - ly * xz), v / (lx * Math.hypot(lz, yz)), lz];
};

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
  /** The tilt factors xy, xz, yz of the kernels' minimum image (0 in an orthogonal box). */
  tilt: number[];
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
 * In a triclinic box an atom's cell along d is that of its fractional coordinate (the offset f_d = lamda_d L_d binned
 * as an orthogonal box's); the positions stay Cartesian.
 */
export const sortByCell = (buf: CellScratch, state: SimState, maxCutoff: number, extra: number, minCells = 3): CellGrid | null => {
  const { n, x, type } = state;
  const two = state.dimension === 2;
  const box = state.box;
  const per = box.periodic;
  const len = [0, 1, 2].map((d) => box.hi[d] - box.lo[d]);
  const wid = boxWidths(box);
  // a non-periodic dimension has at least one cell, and the kernels' scan takes each cell once along an edge of fewer
  // than 3 cells (no image to find there)
  const nc = [0, 1, 2].map((d) => (d === 2 && two ? 1 : per[d] ? Math.floor(wid[d] / maxCutoff) : Math.max(1, Math.floor(wid[d] / maxCutoff))));
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
  const lo = box.lo;
  const tri = box.triclinic;
  const [xy, xz, yz] = box.tilt;
  const s = [nc[0] / len[0], nc[1] / len[1], nc[2] / len[2]];
  const f = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    let c = 0;
    // the offset from the box origin; in a triclinic box its fractional coordinates times the edge lengths (back
    // substitution in h), binned along the edges as an orthogonal box's offsets
    const ox = x[3 * i] - lo[0], oy = x[3 * i + 1] - lo[1], oz = x[3 * i + 2] - lo[2];
    if (tri) {
      f[2] = oz;
      f[1] = oy - yz * (oz / len[2]);
      f[0] = ox - xy * (f[1] / len[1]) - xz * (oz / len[2]);
    } else {
      f[0] = ox; f[1] = oy; f[2] = oz;
    }
    for (let d = 2; d >= 0; d--) {
      let k = 0;
      if (!(d === 2 && two)) {
        // periodic: an atom outside the box (it has moved since the last wrap, run/accel.ts) in its image's cell;
        // non-periodic: one past an edge (a shrink-wrapped box between its resets, an atom about to be lost) in the
        // edge cell, which keeps every partner within the cutoff in the cells the scan takes
        const dx = f[d];
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
  // the kernels' minimum image (WGSL_BOX_IMAGE): the identity along a non-periodic dimension
  const L = len.map((l, d) => (per[d] || (d === 2 && two) ? l : NO_IMAGE));
  return { L, tilt: tri ? [xy, xz, yz] : [0, 0, 0], nc, ncell, base, two };
};
