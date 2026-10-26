/*
 * The pair search of the common half neighbor lists (no exclusions, include group or radii; a type pair without
 * interaction has cutsq 0), with or without special bonds, shared by Neighbor.buildList (neighbor.ts) and the shared-memory threads
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

/**
 * An upper bound of the pairs any one bin tests: with at most F atoms in a bin, each of a bin's atoms tests at most F
 * atoms in its own bin and F in each bin of the half-stencil rows.
 */
export const binBound = (B: HalfBins, nbins: number): number => {
  const { bL, bG, rLo, rHi } = B;
  let F = 0;
  for (let b = 0; b < nbins; b++) {
    const c = bL[b + 1] - bL[b] + bG[b + 1] - bG[b];
    if (c > F) F = c;
  }
  let nhalf = 1;
  for (let r = 0; r < B.nrows; r++) nhalf += rHi[r] - rLo[r] + 1;
  return F * F * nhalf;
};

/**
 * Appends the entries of bins [b0, b1) to (pa, pj) from entry np, stopping before a bin when binMax more entries (an
 * upper bound of one bin's candidates, binBound) would not fit in cap. out[0] is the bin it stopped at (b1 when every
 * bin was done), out[1] the number of entries.
 */
export const halfBinsPairs = (
  B: HalfBins, b0: number, b1: number, pa: Int32Array, pj: Int32Array, np0: number, cap: number, binMax: number, out: Int32Array,
): void => {
  const { bL, bG, rLo, rHi, nrows, lk, lt, lxs, gk, gt, gxs, cutsq, nt } = B;
  let np = np0;
  let b = b0;
  for (; b < b1; b++) {
    const l0 = bL[b], l1 = bL[b + 1], g0 = bG[b], g1 = bG[b + 1];
    if (l1 === l0 && g1 === g0) continue;
    if (np + binMax > cap) break;
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

/** The special bonds of a list build (Neighbor.buildList encode, for a list with special bonds and no exclusions). */
export interface SpecialBins {
  /** the partners (atom IDs) of owned atom a at offset[a] .. offset[a + 1] - 1, with their order (1 = 1-2, 2 = 1-3, 3 = 1-4) */
  offset: Int32Array;
  partner: Int32Array;
  order: Int8Array;
  /** atom IDs of the owned atoms, and the owned atom each owned or ghost atom is (an image of) */
  id: Int32Array;
  owner: Int32Array;
  /** special_bonds lj and coul weights by order (index 0 unused); keep pairs whose weights are both 0 (kspace) */
  lj: Float64Array;
  coul: Float64Array;
  keep: boolean;
  /**
   * Half the box length of each periodic dimension (Infinity for a non-periodic one): a partner seen through an image
   * more than half a box away is an ordinary neighbor (see buildList).
   */
  hx: number;
  hy: number;
  hz: number;
  /** neighbor.ts SBBITS: the order is stored in the bits above it */
  sbbits: number;
}

/** encodeSpecial's result for a dropped pair: outside the int32 range of the codes (an order of 2 or 3 sets the sign bit). */
const DROP = 2 ** 31;

/** buildList's entry for owned atom a and neighbor j at separation (dx, dy, dz): j, j with its order, or DROP. */
const encodeSpecial = (S: SpecialBins, a: number, j: number, dx: number, dy: number, dz: number): number => {
  const jid = S.id[S.owner[j]];
  const k1 = S.offset[a + 1];
  for (let k = S.offset[a]; k < k1; k++) {
    if (S.partner[k] !== jid) continue;
    if (Math.abs(dx) > S.hx || Math.abs(dy) > S.hy || Math.abs(dz) > S.hz) return j;
    const o = S.order[k];
    const lj = S.lj[o], cl = S.coul[o];
    if (lj === 1 && cl === 1) return j;
    if (lj === 0 && cl === 0 && !S.keep) return DROP;
    return j | (o << S.sbbits);
  }
  return j;
};

/** halfBinsPairs for a list with special bonds: the same pairs, each encoded (or dropped) as buildList does. */
export const halfBinsPairsSpecial = (
  B: HalfBins, S: SpecialBins, b0: number, b1: number, pa: Int32Array, pj: Int32Array, np0: number, cap: number, binMax: number, out: Int32Array,
): void => {
  const { bL, bG, rLo, rHi, nrows, lk, lt, lxs, gk, gt, gxs, cutsq, nt } = B;
  let np = np0;
  let b = b0;
  for (; b < b1; b++) {
    const l0 = bL[b], l1 = bL[b + 1], g0 = bG[b], g1 = bG[b + 1];
    if (l1 === l0 && g1 === g0) continue;
    if (np + binMax > cap) break;
    for (let r = -1; r < nrows; r++) {
      const own = r < 0;
      let lA = l0, lB = l1, gA = g0, gB = g1;
      if (!own) {
        const ba = b + rLo[r], bb = b + rHi[r];
        lA = bL[ba]; lB = bL[bb + 1];
        gA = bG[ba]; gB = bG[bb + 1];
        if (lB === lA && gB === gA) continue;
      }
      for (let p = l0; p < l1; p++) {
        const kp = lk[p], tp = lt[p], row = tp * nt;
        const px = lxs[3 * p], py = lxs[3 * p + 1], pz = lxs[3 * p + 2];
        for (let q = own ? p + 1 : lA; q < lB; q++) {
          const kq = lk[q], tq = lt[q];
          const dx = lxs[3 * q] - px, dy = lxs[3 * q + 1] - py, dz = lxs[3 * q + 2] - pz;
          const low = kp < kq;
          if (!(dx * dx + dy * dy + dz * dz <= (low ? cutsq[row + tq] : cutsq[tq * nt + tp]))) continue;
          const a = low ? kp : kq;
          const e = encodeSpecial(S, a, low ? kq : kp, dx, dy, dz);
          if (e !== DROP) { pa[np] = a; pj[np] = e; np++; }
        }
        for (let q = gA; q < gB; q++) {
          const dx = gxs[3 * q] - px, dy = gxs[3 * q + 1] - py, dz = gxs[3 * q + 2] - pz;
          if (!(dx * dx + dy * dy + dz * dz <= cutsq[row + gt[q]]) || dz < 0 || (dz === 0 && (dy < 0 || (dy === 0 && dx < 0)))) continue;
          const e = encodeSpecial(S, kp, gk[q], dx, dy, dz);
          if (e !== DROP) { pa[np] = kp; pj[np] = e; np++; }
        }
      }
      if (own) continue;
      for (let p = g0; p < g1; p++) {
        const kp = gk[p], tp = gt[p];
        const px = gxs[3 * p], py = gxs[3 * p + 1], pz = gxs[3 * p + 2];
        for (let q = lA; q < lB; q++) {
          const dx = lxs[3 * q] - px, dy = lxs[3 * q + 1] - py, dz = lxs[3 * q + 2] - pz;
          if (!(dx * dx + dy * dy + dz * dz <= cutsq[lt[q] * nt + tp]) || dz > 0 || (dz === 0 && (dy > 0 || (dy === 0 && dx > 0)))) continue;
          const e = encodeSpecial(S, lk[q], kp, dx, dy, dz);
          if (e !== DROP) { pa[np] = lk[q]; pj[np] = e; np++; }
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
  halfList(B: HalfBins, S: SpecialBins | null, nbins: number, nlocal: number, binMax: number): { segs: ListSegment[]; np: number } | null;
}
