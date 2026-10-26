/*
 * The pair search of the common half neighbor list (no special bonds, exclusions, include group or radii; a type pair
 * without interaction has cutsq 0), shared by Neighbor.buildList (neighbor.ts) and the shared-memory threads
 * (pairThreads.ts, pairThreadsCore.ts), which run it on ranges of bins. The order of the entries is buildList's: for
 * each bin and each of its rows (the bin itself first), each owned atom of the bin with the owned atoms of the row
 * (later ones in its own bin) and then the ghosts of the row, then each ghost of the bin with the owned atoms of the
 * row. An owned pair is kept by the lower index, a ghost when it lies above the owned atom.
 */

/** The binned atoms of one list build (see Neighbor.buildList). */
export interface HalfBins {
  /** prefix sums of the owned and of the ghost atoms per bin (nbins + 1) */
  bL: Int32Array;
  bG: Int32Array;
  /** the stencil rows, as bin offsets [rLo, rHi] (nrows) */
  rLo: Int32Array;
  rHi: Int32Array;
  nrows: number;
  /** owned and ghost atoms in bin order: index, type, position */
  lk: Int32Array;
  lt: Int32Array;
  lxs: Float64Array;
  gk: Int32Array;
  gt: Int32Array;
  gxs: Float64Array;
  /** squared neighbor cutoff per type pair ((ntypes + 1)^2), nt = ntypes + 1 */
  cutsq: Float64Array;
  nt: number;
}

/** The pairs bin b tests (an upper bound of its entries). */
export const binCandidates = (B: HalfBins, b: number): number => {
  const { bL, bG, rLo, rHi } = B;
  const l = bL[b + 1] - bL[b], g = bG[b + 1] - bG[b];
  if (l === 0 && g === 0) return 0;
  let n = (l * (l - 1)) / 2 + l * g;
  for (let r = 0; r < B.nrows; r++) {
    const ba = b + rLo[r], bb = b + rHi[r] + 1;
    const lr = bL[bb] - bL[ba], gr = bG[bb] - bG[ba];
    n += l * (lr + gr) + g * lr;
  }
  return n;
};

/**
 * Appends the entries of bins [b0, b1) to (pa, pj) from entry np, stopping before a bin whose candidates do not fit in
 * cap. out[0] is the bin it stopped at (b1 when every bin was done), out[1] the number of entries.
 */
export const halfBinsPairs = (
  B: HalfBins, b0: number, b1: number, pa: Int32Array, pj: Int32Array, np0: number, cap: number, out: Int32Array,
): void => {
  const { bL, bG, rLo, rHi, nrows, lk, lt, lxs, gk, gt, gxs, cutsq, nt } = B;
  let np = np0;
  let b = b0;
  for (; b < b1; b++) {
    const l0 = bL[b], l1 = bL[b + 1], g0 = bG[b], g1 = bG[b + 1];
    if (l1 === l0 && g1 === g0) continue;
    if (np + binCandidates(B, b) > cap) break;
    for (let r = -1; r < nrows; r++) {
      const own = r < 0;
      let lA = l0, lB = l1, gA = g0, gB = g1;
      if (!own) {
        const ba = b + rLo[r], bb = b + rHi[r];
        lA = bL[ba]; lB = bL[bb + 1];
        gA = bG[ba]; gB = bG[bb + 1];
        if (lB === lA && gB === gA) continue;
      }
      // an entry is written before its distance test and kept by advancing np
      for (let p = l0; p < l1; p++) {
        const kp = lk[p], tp = lt[p], row = tp * nt;
        const px = lxs[3 * p], py = lxs[3 * p + 1], pz = lxs[3 * p + 2];
        for (let q = own ? p + 1 : lA; q < lB; q++) {
          const kq = lk[q], tq = lt[q];
          const dx = lxs[3 * q] - px, dy = lxs[3 * q + 1] - py, dz = lxs[3 * q + 2] - pz;
          const r2 = dx * dx + dy * dy + dz * dz;
          const low = kp < kq;
          pa[np] = low ? kp : kq; pj[np] = low ? kq : kp;
          np += +(r2 <= (low ? cutsq[row + tq] : cutsq[tq * nt + tp]));
        }
        for (let q = gA; q < gB; q++) {
          const dx = gxs[3 * q] - px, dy = gxs[3 * q + 1] - py, dz = gxs[3 * q + 2] - pz;
          const r2 = dx * dx + dy * dy + dz * dz;
          pa[np] = kp; pj[np] = gk[q];
          np += +(r2 <= cutsq[row + gt[q]] && !(dz < 0 || (dz === 0 && (dy < 0 || (dy === 0 && dx < 0)))));
        }
      }
      if (own) continue;
      for (let p = g0; p < g1; p++) {
        const kp = gk[p], tp = gt[p];
        const px = gxs[3 * p], py = gxs[3 * p + 1], pz = gxs[3 * p + 2];
        for (let q = lA; q < lB; q++) {
          const dx = lxs[3 * q] - px, dy = lxs[3 * q + 1] - py, dz = lxs[3 * q + 2] - pz;
          const r2 = dx * dx + dy * dy + dz * dz;
          pa[np] = lk[q]; pj[np] = kp;
          // the ghost lies at offset p - q from the owned atom q: kept when that is lex-positive
          np += +(r2 <= cutsq[lt[q] * nt + tp] && !(dz > 0 || (dz === 0 && (dy > 0 || (dy === 0 && dx > 0)))));
        }
      }
    }
  }
  out[0] = b;
  out[1] = np;
};

/** Entries o .. o + n - 1 of the arrays (a: the owned atom, j: its neighbor). */
export interface ListSegment { a: Int32Array; j: Int32Array; o: number; n: number }

/** Threads that run halfBinsPairs on ranges of bins (pairThreads.ts SharedPairThreads.halfList). */
export interface NeighborThreads {
  /**
   * The entries of all bins, as segments in buildList's order (read before the next search, which reuses the arrays);
   * null when the threads cannot take this build (the engine thread then runs the kernel alone).
   */
  halfList(B: HalfBins, nbins: number, nlocal: number): { segs: ListSegment[]; np: number } | null;
}
