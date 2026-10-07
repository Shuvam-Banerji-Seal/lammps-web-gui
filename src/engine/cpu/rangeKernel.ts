/*
 * The force kernel one CPU thread runs: Lennard-Jones forces for the pairs
 * owned by a contiguous range of cells. Like the single-thread backend
 * (cpu/forces.ts) it uses a half stencil — own cell with b > a, plus the 13
 * forward neighbour cells (4 in 2D) — so every pair is evaluated exactly once
 * over all threads (Newton's third law), given >= 3 cells per dimension.
 *
 * A pair's partner b may belong to another thread's range. The thread keeps
 * the forces on its own atoms in `f` and returns the reactions on foreign
 * atoms as a short "ghost" list (slot + force) that the caller adds after
 * all threads finish; threads never write each other's atoms.
 * Pure function of plain typed arrays, so it runs unchanged in a Web Worker.
 *
 * With u = 1/r^2, u3 = u^3: F.r/r^2 = u u3 (f12 u3 - f6),
 * E = u3 (e12 u3 - e6) - eshift (pairs.ts).
 */

export interface RangeCoefficients {
  stride: number;
  cutsq: Float64Array;
  e12: Float64Array;
  e6: Float64Array;
  f12: Float64Array;
  f6: Float64Array;
  eshift: Float64Array;
}

export interface RangeTask {
  /** Positions (3 per slot) and types of ALL atoms in cell-sorted order. */
  xs: Float64Array;
  ts: Int32Array;
  /** start[c]..start[c+1] are the slots of cell c (length ncell + 1). */
  start: Int32Array;
  nc: [number, number, number];
  L: [number, number, number];
  two: boolean;
  /** This thread's cells: [cell0, cell1). */
  cell0: number;
  cell1: number;
  coef: RangeCoefficients;
}

export interface RangeResult {
  /** First slot of the range; f holds 3 components per own slot from there. */
  slot0: number;
  f: Float64Array;
  /** Reaction forces on atoms outside the range: slots and 3 components each. */
  ghostSlots: Int32Array;
  ghostF: Float64Array;
  pe: number;
  virial: number;
}

const FORWARD_3D: number[][] = [];
for (let dz = -1; dz <= 1; dz++) {
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      if (dz > 0 || (dz === 0 && (dy > 0 || (dy === 0 && dx > 0)))) FORWARD_3D.push([dx, dy, dz]);
    }
  }
}
const FORWARD_2D = FORWARD_3D.filter((o) => o[2] === 0);

// scratch reused across calls (one per thread / worker)
let ghostAcc = new Float64Array(0);
let ghostStamp = new Int32Array(0);
let ghostList = new Int32Array(0);
let epoch = 0;

export const computeCellRange = (t: RangeTask): RangeResult => {
  const { xs, ts, start, nc, L, two, cell0, cell1 } = t;
  const { stride, cutsq, e12, e6, f12, f6, eshift } = t.coef;
  const [ncx, ncy, ncz] = nc;
  const [Lx, Ly, Lz] = L;
  const hx = 0.5 * Lx, hy = 0.5 * Ly, hz = 0.5 * Lz;
  const n = ts.length;
  const slot0 = start[cell0];
  const slot1 = start[cell1];
  const f = new Float64Array(3 * (slot1 - slot0));
  if (ghostStamp.length < n) {
    ghostAcc = new Float64Array(3 * n);
    ghostStamp = new Int32Array(n);
    ghostList = new Int32Array(n);
    epoch = 0;
  }
  epoch++;
  let nGhost = 0;
  const stencil = two ? FORWARD_2D : FORWARD_3D;
  const neighbours = new Int32Array(stencil.length + 1);
  let pe = 0;
  let virial = 0;

  for (let c = cell0; c < cell1; c++) {
    const cx = c % ncx;
    const cy = Math.floor(c / ncx) % ncy;
    const cz = Math.floor(c / (ncx * ncy));
    neighbours[0] = c;
    for (let q = 0; q < stencil.length; q++) {
      const o = stencil[q];
      neighbours[q + 1] = ((((cz + o[2]) % ncz) + ncz) % ncz * ncy
        + (((cy + o[1]) % ncy) + ncy) % ncy) * ncx
        + (((cx + o[0]) % ncx) + ncx) % ncx;
    }
    const aEnd = start[c + 1];
    for (let a = start[c]; a < aEnd; a++) {
      const xi = xs[3 * a], yi = xs[3 * a + 1], zi = xs[3 * a + 2];
      const ti = ts[a] * stride;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let q = 0; q < neighbours.length; q++) {
        const c2 = neighbours[q];
        const bEnd = start[c2 + 1];
        for (let b = q === 0 ? a + 1 : start[c2]; b < bEnd; b++) {
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
          if (b >= slot0 && b < slot1) {
            const o = 3 * (b - slot0);
            f[o] -= fx; f[o + 1] -= fy; f[o + 2] -= fz;
          } else {
            if (ghostStamp[b] !== epoch) {
              ghostStamp[b] = epoch;
              ghostList[nGhost++] = b;
              ghostAcc[3 * b] = 0; ghostAcc[3 * b + 1] = 0; ghostAcc[3 * b + 2] = 0;
            }
            ghostAcc[3 * b] -= fx; ghostAcc[3 * b + 1] -= fy; ghostAcc[3 * b + 2] -= fz;
          }
          pe += u3 * (e12[k] * u3 - e6[k]) - eshift[k];
          virial += fpair * r2;
        }
      }
      const o = 3 * (a - slot0);
      f[o] += fxi; f[o + 1] += fyi; f[o + 2] += fzi;
    }
  }
  const ghostSlots = ghostList.slice(0, nGhost);
  const ghostF = new Float64Array(3 * nGhost);
  for (let g = 0; g < nGhost; g++) {
    const b = ghostSlots[g];
    ghostF[3 * g] = ghostAcc[3 * b]; ghostF[3 * g + 1] = ghostAcc[3 * b + 1]; ghostF[3 * g + 2] = ghostAcc[3 * b + 2];
  }
  return { slot0, f, ghostSlots, ghostF, pe, virial };
};
