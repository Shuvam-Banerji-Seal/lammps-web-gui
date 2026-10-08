import type { ForceBackend, ForceResult, PairTable, SimState } from '../types';
import { pairArrays, type PairArrays } from '../pairs';

/*
 * Reference fp64 Lennard-Jones forces (Allen & Tildesley ch. 5).
 *
 * Two paths, chosen per call from the box and the largest cutoff rc:
 *  - cell list: every used periodic dimension has >= 3 cells of edge >= rc,
 *    so the minimum image is unique and a half stencil (own cell with j > i,
 *    plus 13 forward cells; 4 in 2D) visits each interacting pair once;
 *  - all images: a box shorter than 3 rc in some dimension. Pairs are summed
 *    over every periodic image within rc, including an atom's own images when
 *    L < rc. O(N^2), which is fine for the small boxes that need it.
 * In 2D (dimension 2) z is never imaged, matching LAMMPS's 2d convention of a
 * thin periodic z.
 */

const FORWARD_3D: [number, number, number][] = [];
for (let dz = -1; dz <= 1; dz++) {
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      if (dz > 0 || (dz === 0 && (dy > 0 || (dy === 0 && dx > 0)))) FORWARD_3D.push([dx, dy, dz]);
    }
  }
}
const FORWARD_2D = FORWARD_3D.filter(([, , dz]) => dz === 0);

export class CpuForceBackend implements ForceBackend {
  readonly kind = 'cpu' as const;
  readonly label: string = 'CPU · fp64';
  private cache: { table: PairTable; arrays: PairArrays; key: string } | null = null;
  // cell-list scratch, grown on demand and reused across steps
  private start = new Int32Array(0);
  private sorted = new Int32Array(0);
  private cellOfAtom = new Int32Array(0);
  private xs = new Float64Array(0);
  private fs = new Float64Array(0);
  private ts = new Int32Array(0);

  /** Coefficient arrays, rebuilt when the table object or its contents change. */
  private arraysFor(table: PairTable): PairArrays {
    const key = JSON.stringify([table.pairs, table.shift]);
    if (!this.cache || this.cache.table !== table || this.cache.key !== key) {
      this.cache = { table, arrays: pairArrays(table), key };
    }
    return this.cache.arrays;
  }

  compute(state: SimState, table: PairTable): ForceResult {
    const pa = this.arraysFor(table);
    state.f.fill(0);
    if (state.n === 0 || pa.maxCutoff <= 0) return { pe: 0, virial: 0 };
    const dims = state.dimension;
    const L = [0, 1, 2].map((d) => state.box.hi[d] - state.box.lo[d]);
    // v1 boxes are fully periodic (the interpreter rejects anything else)
    const usesCells = [0, 1, 2].every((d) =>
      (d === 2 && dims === 2) || Math.floor(L[d] / pa.maxCutoff) >= 3);
    return usesCells ? this.cellList(state, pa, L) : allImages(state, pa, L);
  }

  private cellList(s: SimState, pa: PairArrays, L: number[]): ForceResult {
    const { n, x, f, type } = s;
    const two = s.dimension === 2;
    const ncx = Math.floor(L[0] / pa.maxCutoff);
    const ncy = Math.floor(L[1] / pa.maxCutoff);
    const ncz = two ? 1 : Math.floor(L[2] / pa.maxCutoff);
    const ncell = ncx * ncy * ncz;
    if (this.start.length < ncell + 1) this.start = new Int32Array(ncell + 1);
    if (this.sorted.length < n) {
      this.sorted = new Int32Array(n);
      this.cellOfAtom = new Int32Array(n);
      this.xs = new Float64Array(3 * n);
      this.fs = new Float64Array(3 * n);
      this.ts = new Int32Array(n);
    }
    const { start, sorted, cellOfAtom, xs, fs, ts } = this;

    // Counting sort of the atoms by cell, then gather positions and types in
    // that order so every cell is a contiguous range (cache-friendly loops).
    start.fill(0, 0, ncell + 1);
    const [lx, ly, lz] = s.box.lo;
    const sx = ncx / L[0], sy = ncy / L[1], sz = ncz / L[2];
    for (let i = 0; i < n; i++) {
      let cx = Math.floor((x[3 * i] - lx) * sx);
      let cy = Math.floor((x[3 * i + 1] - ly) * sy);
      let cz = two ? 0 : Math.floor((x[3 * i + 2] - lz) * sz);
      if (cx >= ncx) cx = ncx - 1; else if (cx < 0) cx = 0;   // x == hi after rounding
      if (cy >= ncy) cy = ncy - 1; else if (cy < 0) cy = 0;
      if (cz >= ncz) cz = ncz - 1; else if (cz < 0) cz = 0;
      const c = (cz * ncy + cy) * ncx + cx;
      cellOfAtom[i] = c;
      start[c + 1]++;
    }
    for (let c = 0; c < ncell; c++) start[c + 1] += start[c];
    for (let i = 0; i < n; i++) {
      // place atoms at the END of their range, walking backwards, so start[]
      // ends up holding each cell's first slot again
      const slot = --start[cellOfAtom[i] + 1];
      sorted[slot] = i;
    }
    for (let k = 0; k < n; k++) {
      const i = sorted[k];
      xs[3 * k] = x[3 * i]; xs[3 * k + 1] = x[3 * i + 1]; xs[3 * k + 2] = x[3 * i + 2];
      ts[k] = type[i];
    }
    fs.fill(0, 0, 3 * n);
    // after the backward fill, start[c + 1] is cell c's first slot; shift down
    for (let c = 0; c < ncell; c++) start[c] = start[c + 1];
    start[ncell] = n;

    const [Lx, Ly, Lz] = L;
    const hx = 0.5 * Lx, hy = 0.5 * Ly, hz = 0.5 * Lz;
    const stride = pa.stride;
    const { cutsq, e12, e6, f12, f6, eshift } = pa;
    let pe = 0;
    let virial = 0;
    const stencil = two ? FORWARD_2D : FORWARD_3D;
    const neighbours = new Int32Array(stencil.length);

    for (let cz = 0; cz < ncz; cz++) {
      for (let cy = 0; cy < ncy; cy++) {
        for (let cx = 0; cx < ncx; cx++) {
          const c = (cz * ncy + cy) * ncx + cx;
          for (let q = 0; q < stencil.length; q++) {
            const [ox, oy, oz] = stencil[q];
            neighbours[q] = ((((cz + oz) % ncz) + ncz) % ncz * ncy
              + (((cy + oy) % ncy) + ncy) % ncy) * ncx
              + (((cx + ox) % ncx) + ncx) % ncx;
          }
          const aEnd = start[c + 1];
          for (let a = start[c]; a < aEnd; a++) {
            const xi = xs[3 * a], yi = xs[3 * a + 1], zi = xs[3 * a + 2];
            const ti = ts[a] * stride;
            let fxi = 0, fyi = 0, fzi = 0;
            // own cell (b > a), then the forward neighbour cells (all b)
            for (let q = -1; q < neighbours.length; q++) {
              const c2 = q < 0 ? c : neighbours[q];
              const bEnd = start[c2 + 1];
              for (let b = q < 0 ? a + 1 : start[c2]; b < bEnd; b++) {
                let dx = xi - xs[3 * b];
                let dy = yi - xs[3 * b + 1];
                let dz = two ? 0 : zi - xs[3 * b + 2];
                if (dx > hx) dx -= Lx; else if (dx < -hx) dx += Lx;
                if (dy > hy) dy -= Ly; else if (dy < -hy) dy += Ly;
                if (dz > hz) dz -= Lz; else if (dz < -hz) dz += Lz;
                const r2 = dx * dx + dy * dy + dz * dz;
                const k = ti + ts[b];
                if (r2 >= cutsq[k]) continue;
                const u = 1 / r2;
                const u3 = u * u * u;
                const fpair = u * u3 * (f12[k] * u3 - f6[k]);
                const fx = fpair * dx, fy = fpair * dy, fz = fpair * dz;
                fxi += fx; fyi += fy; fzi += fz;
                fs[3 * b] -= fx; fs[3 * b + 1] -= fy; fs[3 * b + 2] -= fz;
                pe += u3 * (e12[k] * u3 - e6[k]) - eshift[k];
                virial += fpair * r2;
              }
            }
            fs[3 * a] += fxi; fs[3 * a + 1] += fyi; fs[3 * a + 2] += fzi;
          }
        }
      }
    }
    // scatter back to the atoms' own order
    for (let k = 0; k < n; k++) {
      const i = sorted[k];
      f[3 * i] = fs[3 * k]; f[3 * i + 1] = fs[3 * k + 1]; f[3 * i + 2] = fs[3 * k + 2];
    }
    return { pe, virial };
  }

  dispose(): void {
    this.cache = null;
  }
}

/** Every periodic image within the cutoff; also the reference the cell list is tested against. */
export const allImages = (s: SimState, pa: PairArrays, L: number[]): ForceResult => {
  const { n, x, f, type } = s;
  const two = s.dimension === 2;
  const rc = pa.maxCutoff;
  // with |d| <= L/2 after the minimum image, an image shift k can be within
  // rc only if |k| L - L/2 < rc
  const kmax = [0, 1, 2].map((d) =>
    !s.box.periodic[d] || (d === 2 && two) ? 0 : Math.floor((rc + 0.5 * L[d]) / L[d]));
  const stride = pa.stride;
  const { cutsq, e12, e6, f12, f6, eshift } = pa;
  let pe = 0;
  let virial = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const self = i === j;
      const k = type[i] * stride + type[j];
      let bx = x[3 * i] - x[3 * j];
      let by = x[3 * i + 1] - x[3 * j + 1];
      let bz = two ? 0 : x[3 * i + 2] - x[3 * j + 2];
      if (s.box.periodic[0]) bx -= L[0] * Math.round(bx / L[0]);
      if (s.box.periodic[1]) by -= L[1] * Math.round(by / L[1]);
      if (s.box.periodic[2] && !two) bz -= L[2] * Math.round(bz / L[2]);
      for (let a = -kmax[0]; a <= kmax[0]; a++) {
        for (let b = -kmax[1]; b <= kmax[1]; b++) {
          for (let c = -kmax[2]; c <= kmax[2]; c++) {
            if (self && a === 0 && b === 0 && c === 0) continue;
            const dx = bx + a * L[0];
            const dy = by + b * L[1];
            const dz = bz + c * L[2];
            const r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cutsq[k]) continue;
            const u = 1 / r2;
            const u3 = u * u * u;
            const fpair = u * u3 * (f12[k] * u3 - f6[k]);
            const e = u3 * (e12[k] * u3 - e6[k]) - eshift[k];
            if (self) {
              // an atom and its own image: each such pair is met twice (k, -k)
              // and the two forces cancel; energy and virial count half each
              pe += 0.5 * e;
              virial += 0.5 * fpair * r2;
              continue;
            }
            f[3 * i] += fpair * dx; f[3 * i + 1] += fpair * dy; f[3 * i + 2] += fpair * dz;
            f[3 * j] -= fpair * dx; f[3 * j + 1] -= fpair * dy; f[3 * j + 2] -= fpair * dz;
            pe += e;
            virial += fpair * r2;
          }
        }
      }
    }
  }
  return { pe, virial };
};
