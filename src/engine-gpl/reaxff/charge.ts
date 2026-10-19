/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/fix_qeq_reaxff.cpp (init_taper, calculate_H, compute_H, CG, sparse_matvec,
 * calculate_Q, init_matvec), src/REAXFF/fix_qtpie_reaxff.cpp (the same routines of the QTPIE
 * solver, which repeat those of fix qeq/reaxff), at commit b332f28. Authors credited in the C++ file
 * headers: Hasan Metin Aktulga (Purdue University, now Lawrence Berkeley National Laboratory); Ray
 * Shan (Sandia); Efstratios M Kritikos (Caltech) and Navraj S Lalli (Imperial College London) for the
 * QTPIE solver. Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * The shared charge-equilibration machinery of fix qeq/reaxff, fix qtpie/reaxff (and its qeq/rel
 * variant) and fix acks2/reaxff: the taper polynomial of the fix, the shielded H matrix over the
 * owned and ghost atoms with the half-list flag rule of LAMMPS compute_H, the Jacobi-preconditioned
 * CG, and the two-solve (s and t) fictitious-charge step with its history extrapolation.
 */
import type { System } from '../../engine/system';
import { pow as libmPow } from '../libm/pow';

export const EV_TO_KCAL_PER_MOL = 14.4;
const EPSILON = 0.0001;
/** nprev of the fix: the number of stored solutions of the extrapolation. */
export const NPREV = 4;

/** The Taper polynomial of the fix (init_taper of fix_qeq_reaxff.cpp and fix_qtpie_reaxff.cpp). */
export const taperFix = (swa: number, swb: number): Float64Array => {
  const Tap = new Float64Array(8);
  const d7 = libmPow(swb - swa, 7);
  const swa2 = swa * swa, swa3 = swa * swa * swa;
  const swb2 = swb * swb, swb3 = swb * swb * swb;
  Tap[7] = 20.0 / d7;
  Tap[6] = (-70.0 * (swa + swb)) / d7;
  Tap[5] = (84.0 * (swa2 + 3.0 * swa * swb + swb2)) / d7;
  Tap[4] = (-35.0 * (swa3 + 9.0 * swa2 * swb + 9.0 * swa * swb2 + swb3)) / d7;
  Tap[3] = (140.0 * (swa3 * swb + 3.0 * swa2 * swb2 + swa * swb3)) / d7;
  Tap[2] = (-210.0 * (swa3 * swb2 + swa2 * swb3)) / d7;
  Tap[1] = (140.0 * swa3 * swb3) / d7;
  Tap[0] = (-35.0 * swa3 * swb2 * swb2 + 21.0 * swa2 * swb3 * swb2 - 7.0 * swa * swb3 * swb3 + swb3 * swb3 * swb) / d7;
  return Tap;
};

/** calculate_H: the taper times 14.4 over the shielded distance. */
export const calcHValue = (Tap: Float64Array, r: number, gamma: number): number => {
  let taper = Tap[7] * r + Tap[6];
  taper = taper * r + Tap[5];
  taper = taper * r + Tap[4];
  taper = taper * r + Tap[3];
  taper = taper * r + Tap[2];
  taper = taper * r + Tap[1];
  taper = taper * r + Tap[0];
  let denom = r * r * r + gamma;
  denom = libmPow(denom, 1.0 / 3.0);
  return (taper * EV_TO_KCAL_PER_MOL) / denom;
};

/** A row-compressed symmetric sparse matrix: row i holds jlist/val over [start[i], start[i+1]). */
export interface SparseRows { start: Int32Array; jlist: Int32Array; val: Float64Array }

/**
 * The half-list flag of compute_H and compute_X: a pair of owned i with a partner j within the cutoff
 * (squared cut2) is taken when j is owned, or when j is a ghost image of an atom with a larger id, or
 * the same atom with a positive displacement in z, then y, then x (exactly one of two mirrored images).
 */
export const halfFlag = (dx: number, dy: number, dz: number, rsq: number, cut2: number,
  i: number, j: number, n: number, orig: Int32Array): boolean => {
  if (!(rsq <= cut2)) return false;
  if (j < n) return true;
  if (orig[i] < orig[j]) return true;
  if (orig[i] === orig[j]) {
    if (dz > EPSILON) return true;
    if (Math.abs(dz) < EPSILON) {
      if (dy > EPSILON) return true;
      if (Math.abs(dy) < EPSILON && dx > EPSILON) return true;
    }
  }
  return false;
};

/**
 * Builds the rows of a half-list matrix over the owned atoms of the group (the loop of compute_H).
 * `value(i, j, r, rsq)` returns the matrix element of the pair (called only for pairs that pass the flag);
 * a NaN return leaves the pair out. Ghost rows are empty.
 *
 * The partners j > i within the cutoff come from a cell list over all N atoms (cell edges at least 1.0001 cut, so a
 * pair within the cutoff is never more than one cell apart after rounding) and are visited in ascending j, so the rows
 * and the sequence of value() calls are those of a scan over every j > i, without its N^2 / 2 distances.
 */
export const buildRows = (
  x: Float64Array, n: number, N: number, orig: Int32Array, inGroup: Uint8Array, cut: number,
  value: (i: number, j: number, r: number, rsq: number) => number,
): SparseRows => {
  const start = new Int32Array(N + 1);
  const jl: number[] = [], vl: number[] = [];
  const cut2 = cut * cut;
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  for (let a = 0; a < N; a++) {
    const ax = x[3 * a], ay = x[3 * a + 1], az = x[3 * a + 2];
    if (ax < x0) x0 = ax; if (ax > x1) x1 = ax;
    if (ay < y0) y0 = ay; if (ay > y1) y1 = ay;
    if (az < z0) z0 = az; if (az > z1) z1 = az;
  }
  const edge = cut * 1.0001;
  const ncx = Math.max(1, Math.floor((x1 - x0) / edge)), ncy = Math.max(1, Math.floor((y1 - y0) / edge));
  const ncz = Math.max(1, Math.floor((z1 - z0) / edge));
  const ncell = ncx * ncy * ncz;
  const hx = ncx > 1 ? (x1 - x0) / ncx : Infinity, hy = ncy > 1 ? (y1 - y0) / ncy : Infinity;
  const hz = ncz > 1 ? (z1 - z0) / ncz : Infinity;
  const cellOf = new Int32Array(N);
  const cellStart = new Int32Array(ncell + 1);
  for (let a = 0; a < N; a++) {
    const cx = Math.min(ncx - 1, Math.floor((x[3 * a] - x0) / hx) | 0);
    const cy = Math.min(ncy - 1, Math.floor((x[3 * a + 1] - y0) / hy) | 0);
    const cz = Math.min(ncz - 1, Math.floor((x[3 * a + 2] - z0) / hz) | 0);
    const c = (cz * ncy + cy) * ncx + cx;
    cellOf[a] = c;
    cellStart[c + 1]++;
  }
  for (let c = 0; c < ncell; c++) cellStart[c + 1] += cellStart[c];
  // atoms of each cell in ascending index (filled in index order)
  const cellAtoms = new Int32Array(N);
  const fill = cellStart.slice(0, ncell);
  for (let a = 0; a < N; a++) cellAtoms[fill[cellOf[a]]++] = a;
  const cand = new Int32Array(N);
  for (let i = 0; i < n; i++) {
    start[i] = jl.length;
    if (!inGroup[i]) continue;
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const c = cellOf[i];
    const cx = c % ncx, cy = ((c - cx) / ncx) % ncy, cz = (c - cx - cy * ncx) / (ncx * ncy);
    let m = 0;
    for (let kz = Math.max(0, cz - 1); kz <= Math.min(ncz - 1, cz + 1); kz++) {
      for (let ky = Math.max(0, cy - 1); ky <= Math.min(ncy - 1, cy + 1); ky++) {
        for (let kx = Math.max(0, cx - 1); kx <= Math.min(ncx - 1, cx + 1); kx++) {
          const cc = (kz * ncy + ky) * ncx + kx;
          for (let q = cellStart[cc]; q < cellStart[cc + 1]; q++) {
            const j = cellAtoms[q];
            if (j <= i) continue;
            const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
            if (dx * dx + dy * dy + dz * dz <= cut2) cand[m++] = j;
          }
        }
      }
    }
    const js = cand.subarray(0, m).sort();
    for (let k = 0; k < m; k++) {
      const j = js[k];
      const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      if (!halfFlag(dx, dy, dz, rsq, cut2, i, j, n, orig)) continue;
      const v = value(i, j, Math.sqrt(rsq), rsq);
      if (Number.isNaN(v)) continue;
      jl.push(j);
      vl.push(v);
    }
  }
  for (let i = n; i <= N; i++) start[i] = jl.length;
  return { start, jlist: Int32Array.from(jl), val: Float64Array.from(vl) };
};

/** The per-atom storage of the fictitious charges of one solve (s and t histories, nprev each). */
export type HistoryMap = Map<number, Float64Array>;

/** The history of an atom id (most recent first: s 0..3, t 4..7), created zeroed on first use. */
export const historyOf = (hist: HistoryMap, id: number, len = 2 * NPREV): Float64Array => {
  let h = hist.get(id);
  if (!h) {
    h = new Float64Array(len);
    hist.set(id, h);
  }
  return h;
};

/** Everything the matrix-vector product and the CG need about the owned/ghost layout. */
export interface CgWork {
  grp: number[];
  n: number;
  N: number;
  owner: Int32Array;
  H: SparseRows;
  eta: Float64Array;
  typeAll: Int32Array;
  Hdia_inv: Float64Array;
}

/** sparse_matvec: b = H x over the owned group atoms, the ghost contributions folded to the owners. */
export const sparseMatvec = (w: CgWork, x: Float64Array, b: Float64Array): void => {
  const { grp, n, N, owner, H, eta, typeAll } = w;
  b.fill(0, 0, N);
  for (const i of grp) b[i] = eta[typeAll[i]] * x[i];
  for (const i of grp) {
    for (let p = H.start[i]; p < H.start[i + 1]; p++) {
      const j = H.jlist[p];
      b[i] += H.val[p] * x[j];
      b[j] += H.val[p] * x[i];
    }
  }
  for (let g = n; g < N; g++) b[owner[g]] += b[g];
};

/** Copies the owner value to each ghost (the forward communication of a solution vector). */
export const forwardGhosts = (v: Float64Array, n: number, N: number, owner: Int32Array): void => {
  for (let g = n; g < N; g++) v[g] = v[owner[g]];
};

/** Result of one CG solve: the iteration count (LAMMPS CG return value) and the warning flag. */
export interface CgOptions { tolerance: number; imax: number; maxwarn: boolean; warn: (text: string) => void; step: number; name: string }

/** CG (fix_qeq_reaxff.cpp, fix_qtpie_reaxff.cpp): Jacobi-preconditioned conjugate gradient; returns the iteration count. */
export const cgSolve = (w: CgWork, b: Float64Array, x: Float64Array, o: CgOptions): number => {
  const { grp, N, Hdia_inv, owner, n } = w;
  const r = new Float64Array(N), d = new Float64Array(N), p = new Float64Array(N), q = new Float64Array(N);
  forwardGhosts(x, n, N, owner);
  sparseMatvec(w, x, q);
  for (const j of grp) r[j] = b[j] - q[j];
  for (const j of grp) d[j] = r[j] * Hdia_inv[j];
  let bn = 0;
  for (const j of grp) bn += b[j] * b[j];
  const bNorm = Math.sqrt(bn);
  let sigNew = 0;
  for (const j of grp) sigNew += r[j] * d[j];
  let i = 1;
  for (; i < o.imax && Math.sqrt(sigNew) / bNorm > o.tolerance; ++i) {
    forwardGhosts(d, n, N, owner);
    sparseMatvec(w, d, q);
    let tmp = 0;
    for (const j of grp) tmp += d[j] * q[j];
    const alpha = sigNew / tmp;
    for (const j of grp) x[j] += alpha * d[j];
    for (const j of grp) r[j] += -alpha * q[j];
    for (const j of grp) p[j] = r[j] * Hdia_inv[j];
    const sigOld = sigNew;
    sigNew = 0;
    for (const j of grp) sigNew += r[j] * p[j];
    const beta = sigNew / sigOld;
    for (const j of grp) d[j] = p[j] + beta * d[j];
  }
  if (i >= o.imax && o.maxwarn) {
    o.warn(`Fix ${o.name} CG convergence failed after ${i} iterations at step ${o.step}`);
  }
  return i;
};

/** The inputs of one fictitious-charge solve (the pre_force of fix qeq/reaxff and fix qtpie/reaxff). */
export interface FictitiousInput {
  sys: System;
  groupBit: number;
  swb: number;
  ntypes: number;
  /** shld[i * (ntypes + 1) + j] = (gamma_i gamma_j)^(-1.5). */
  shld: Float64Array;
  Tap: Float64Array;
  eta: Float64Array;
  /** The effective electronegativity of each owned atom (chi for qeq/reaxff, chi_eff for qtpie). */
  chiEff: Float64Array;
  hist: HistoryMap;
  opts: CgOptions;
}

/**
 * The step of pre_force: compute_H, the CG solves for s (b = -chi_eff) and t (b = -1) with the
 * extrapolated starting guesses, and calculate_Q (q = s - u t, u = sum s / sum t over the group).
 * Returns the iteration counts of the two solves.
 */
export const solveFictitious = (f: FictitiousInput): { matvecsS: number; matvecsT: number } => {
  const s = f.sys.state;
  const nb = f.sys.nb;
  const n = nb.nlocal;
  const N = nb.nall;
  const bit = f.groupBit;
  const x = nb.xall;
  const owner = nb.owner;
  const typeAll = nb.typeall;
  // atom ids of owned and ghost atoms (tags)
  const orig = new Int32Array(N);
  for (let a = 0; a < N; a++) orig[a] = s.id[owner[a]];
  const grp: number[] = [];
  for (let i = 0; i < n; i++) if (s.mask[i] & bit) grp.push(i);
  const inGroup = new Uint8Array(N);
  for (const i of grp) inGroup[i] = 1;

  const nt1 = f.ntypes + 1;
  const H = buildRows(x, n, N, orig, inGroup, f.swb, (i, j, r) => {
    return calcHValue(f.Tap, r, f.shld[typeAll[i] * nt1 + typeAll[j]]);
  });

  // init_matvec: preconditioner, sources and extrapolated starting guesses
  const Hdia_inv = new Float64Array(N);
  const bS = new Float64Array(N), bT = new Float64Array(N);
  const sv = new Float64Array(N), tv = new Float64Array(N);
  for (const i of grp) {
    Hdia_inv[i] = 1.0 / f.eta[typeAll[i]];
    bS[i] = -f.chiEff[i];
    bT[i] = -1.0;
    const h = historyOf(f.hist, orig[i]);
    tv[i] = h[4 + 2] + 3 * (h[4] - h[4 + 1]);
    sv[i] = 4 * (h[0] + h[2]) - (6 * h[1] + h[3]);
  }
  forwardGhosts(sv, n, N, owner);
  forwardGhosts(tv, n, N, owner);

  const work: CgWork = { Hdia_inv, grp, owner, n, N, H, eta: f.eta, typeAll };
  const matvecsS = cgSolve(work, bS, sv, f.opts);
  const matvecsT = cgSolve(work, bT, tv, f.opts);

  // calculate_Q
  let sSum = 0, tSum = 0;
  for (const i of grp) { sSum += sv[i]; tSum += tv[i]; }
  const u = sSum / tSum;
  for (const i of grp) {
    s.q[i] = sv[i] - u * tv[i];
    const h = historyOf(f.hist, orig[i]);
    for (let k = NPREV - 1; k > 0; k--) {
      h[k] = h[k - 1];
      h[4 + k] = h[4 + k - 1];
    }
    h[0] = sv[i];
    h[4] = tv[i];
  }
  // charges of the ghosts follow their owners (forward comm of pack_flag 4)
  nb.refreshCharges(s);
  f.sys.bump();
  return { matvecsS, matvecsT };
};
