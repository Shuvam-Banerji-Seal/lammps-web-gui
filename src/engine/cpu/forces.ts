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
  readonly label = 'CPU · fp64';
  private cache: { table: PairTable; arrays: PairArrays; key: string } | null = null;
  private head = new Int32Array(0);
  private next = new Int32Array(0);

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
    const nc = [0, 1, 2].map((d) => (d === 2 && two ? 1 : Math.floor(L[d] / pa.maxCutoff)));
    const ncell = nc[0] * nc[1] * nc[2];
    if (this.head.length < ncell) this.head = new Int32Array(ncell);
    if (this.next.length < n) this.next = new Int32Array(n);
    const head = this.head;
    const next = this.next;
    head.fill(-1, 0, ncell);
    const lo = s.box.lo;
    const cellOf = (i: number, d: number) => {
      let c = Math.floor(((x[3 * i + d] - lo[d]) / L[d]) * nc[d]);
      if (c >= nc[d]) c = nc[d] - 1;   // x == hi after rounding
      if (c < 0) c = 0;
      return c;
    };
    for (let i = 0; i < n; i++) {
      const c = (cellOf(i, 2) * nc[1] + cellOf(i, 1)) * nc[0] + cellOf(i, 0);
      next[i] = head[c];
      head[c] = i;
    }
    const [Lx, Ly, Lz] = L;
    const hx = 0.5 * Lx, hy = 0.5 * Ly, hz = 0.5 * Lz;
    const stride = pa.stride;
    const { cutsq, e12, e6, f12, f6, eshift } = pa;
    let pe = 0;
    let virial = 0;
    const stencil = two ? FORWARD_2D : FORWARD_3D;

    const pair = (i: number, j: number) => {
      let dx = x[3 * i] - x[3 * j];
      let dy = x[3 * i + 1] - x[3 * j + 1];
      let dz = two ? 0 : x[3 * i + 2] - x[3 * j + 2];
      if (dx > hx) dx -= Lx; else if (dx < -hx) dx += Lx;
      if (dy > hy) dy -= Ly; else if (dy < -hy) dy += Ly;
      if (dz > hz) dz -= Lz; else if (dz < -hz) dz += Lz;
      const r2 = dx * dx + dy * dy + dz * dz;
      const k = type[i] * stride + type[j];
      if (r2 >= cutsq[k]) return;
      const u = 1 / r2;
      const u3 = u * u * u;
      const fpair = u * u3 * (f12[k] * u3 - f6[k]);
      f[3 * i] += fpair * dx; f[3 * i + 1] += fpair * dy; f[3 * i + 2] += fpair * dz;
      f[3 * j] -= fpair * dx; f[3 * j + 1] -= fpair * dy; f[3 * j + 2] -= fpair * dz;
      pe += u3 * (e12[k] * u3 - e6[k]) - eshift[k];
      virial += fpair * r2;
    };

    for (let cz = 0; cz < nc[2]; cz++) {
      for (let cy = 0; cy < nc[1]; cy++) {
        for (let cx = 0; cx < nc[0]; cx++) {
          const c = (cz * nc[1] + cy) * nc[0] + cx;
          for (let i = head[c]; i >= 0; i = next[i]) {
            for (let j = next[i]; j >= 0; j = next[j]) pair(i, j);
          }
          for (const [ox, oy, oz] of stencil) {
            const c2 = ((((cz + oz) % nc[2]) + nc[2]) % nc[2] * nc[1]
              + (((cy + oy) % nc[1]) + nc[1]) % nc[1]) * nc[0]
              + (((cx + ox) % nc[0]) + nc[0]) % nc[0];
            for (let i = head[c]; i >= 0; i = next[i]) {
              for (let j = head[c2]; j >= 0; j = next[j]) pair(i, j);
            }
          }
        }
      }
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
