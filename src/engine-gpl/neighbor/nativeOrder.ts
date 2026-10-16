/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/comm_brick.cpp (CommBrick::setup and
 * CommBrick::borders: the ghost atoms of one processor with periodic boundaries, their order and their periodic
 * images), src/nbin.cpp (NBin::coord2bin), src/nbin_standard.cpp (NBinStandard::setup_bins and bin_atoms: the bin
 * geometry and the bin chains), src/nstencil.cpp (NStencil::bin_distance and the sx, sy, sz bounds) and
 * src/nstencil_bin.cpp (NStencilBin::create: the half stencil with the central bin first), and src/npair_bin.cpp
 * (NPairBin::build, half list, newton on, orthogonal box: the order in which the pairs of each atom are stored).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Original authors: the LAMMPS development team
 * (Sandia National Laboratories), as named in those files.
 *
 * Purpose: pair styles that draw one random number per pair (pair dpd, dpd/ext, dpd/tstat, dpd/ext/tstat) must
 * draw them in native LAMMPS's order. This module reproduces native's local atom order (SimState.order), its
 * ghost atoms (borders(): per dimension, per swap, the atoms of each slab in ascending index order, shifted by
 * one box length), its bins (ascending index inside each bin chain) and its half/bin/newton enumeration. The
 * engine's own neighbor list supplies the membership test, so special-bond and exclusion filters and the
 * engine's cutoff decisions are kept; only the order of the pairs comes from native.
 *
 * Scope: orthogonal boxes (triclinic boxes use the engine's list order and are reported by the caller). Periodic
 * dimensions get ghosts; non-periodic dimensions have none, as in native with one processor.
 */
import type { Neighbor, NeighList } from '../../engine/neighbor';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { nativeOrder } from '../../engine/atoms';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';

/** comm_brick.cpp: "static constexpr double BIG = 1.0e20;" (slab limits that take in every atom). */
const BIG = 1.0e20;

/** Integer key of an owner and its periodic image (images are small; owners below 2^22). */
const imgKey = (owner: number, ix: number, iy: number, iz: number): number =>
  ((owner * 64 + (ix + 32)) * 64 + (iy + 32)) * 64 + (iz + 32);

/** Integer key of a bin (bin indices are far below 4096 in magnitude for any box that fits in memory). */
const OFF = 4096, BW = 8192;
const binKey = (ix: number, iy: number, iz: number): number => ((iz + OFF) * BW + (iy + OFF)) * BW + (ix + OFF);

/**
 * Owned atoms in native local order (x = positions in that order, length 3n) and their ghosts, built as
 * CommBrick::borders() does for one processor. Ghost k (k >= n) is the image of atom root[k] (native index)
 * shifted by img[3k..3k+2] box lengths; x holds the coordinates as native computes them (the shift is added
 * to the sending atom's coordinate in each swap).
 */
export interface NativeGhosts {
  nall: number;
  x: Float64Array;
  root: Int32Array;
  img: Int32Array;
}

/** comm_brick.cpp CommBrick::setup (cutghost = comm cutoff, maxneed = 1 + cut/prd for one processor) and borders(). */
export const nativeGhosts = (xn: Float64Array, n: number, g: Geometry, cut: number, two: boolean): NativeGhosts => {
  const xs: number[] = Array.from(xn.subarray(0, 3 * n));
  const roots: number[] = [];
  const imgs: number[] = [];
  for (let i = 0; i < n; i++) { roots.push(i); imgs.push(0, 0, 0); }
  const prd = [g.lx, g.ly, g.lz];
  for (let dim = 0; dim < (two ? 2 : 3); dim++) {
    if (!g.periodic[dim]) continue;
    const sublo = g.lo[dim], subhi = g.hi[dim], len = prd[dim];
    const maxneed = Math.trunc(cut * 1 / len) + 1;
    const mid = 0.5 * (sublo + subhi);
    let nfirst = 0, nlast = 0;
    for (let ineed = 0; ineed < 2 * maxneed; ineed++) {
      const even = ineed % 2 === 0;
      if (even) { nfirst = nlast; nlast = roots.length; }
      let slo: number, shi: number, sgn: number;
      if (even) {
        slo = ineed < 2 ? -BIG : mid;
        shi = sublo + cut;
        sgn = 1;
      } else {
        slo = subhi - cut;
        shi = ineed < 2 ? BIG : mid;
        sgn = -1;
      }
      const sendlist: number[] = [];
      for (let i = nfirst; i < nlast; i++) {
        const c = xs[3 * i + dim];
        if (c >= slo && c <= shi) sendlist.push(i);
      }
      // pack_border: the sent coordinate gets the periodic shift (pbc * prd) in this dimension only
      for (const i of sendlist) {
        const k = roots.length;
        for (let c = 0; c < 3; c++) xs.push(xs[3 * i + c]);
        xs[3 * k + dim] = xs[3 * i + dim] + sgn * len;
        roots.push(roots[i]);
        imgs.push(imgs[3 * i], imgs[3 * i + 1], imgs[3 * i + 2]);
        imgs[3 * k + dim] += sgn;
      }
    }
  }
  const nall = roots.length;
  return {
    nall,
    x: Float64Array.from(xs),
    root: Int32Array.from(roots),
    img: Int32Array.from(imgs),
  };
};

/**
 * Half/bin/newton pair enumeration in native order (npair_bin.cpp build with HALF && NEWTON, orthonormal;
 * nbin_standard.cpp bin chains; nstencil_bin.cpp half stencil). Returns the pairs as (i, jEnc) entries, two
 * Int32 per pair, with i the engine index of the owned atom, jEnc the engine index of the partner | (order << SBBITS)
 * (the engine's encoding), in native's order: atoms in native local order, and for each atom the partners of its
 * own bin after it in the chain (ghosts only when above and to the right), then the partners of the upper stencil.
 * A candidate is kept only when the engine's half list holds the same pair (its cutoff, special bonds, exclusions).
 */

/**
 * Open-addressing hash table from integer keys to int32 values. The keys are numbers below 2^53 compared exactly, so
 * the table has the same key equality as the Map it replaces; absent keys read as -1 (all values here are >= 0).
 */
class IntTable {
  private cap = 0;
  private mask = 0;
  private keys = new Float64Array(0);
  private vals = new Int32Array(0);
  private used = new Uint8Array(0);
  private count = 0;

  constructor(expected: number) {
    let cap = 16;
    while (cap < 2 * expected) cap *= 2;
    this.alloc(cap);
  }

  private alloc(cap: number): void {
    this.cap = cap;
    this.mask = cap - 1;
    this.keys = new Float64Array(cap);
    this.vals = new Int32Array(cap);
    this.used = new Uint8Array(cap);
  }

  private slot(key: number): number {
    const hi = Math.floor(key / 4294967296);
    const lo = key - hi * 4294967296;
    let h = Math.imul(lo | 0, 0x9e3779b1) ^ Math.imul((hi | 0) + 0x27d4eb2f, 0x85ebca6b);
    h ^= h >>> 15;
    h = Math.imul(h, 0xc2b2ae35);
    h ^= h >>> 13;
    let s = h & this.mask;
    while (this.used[s] === 1 && this.keys[s] !== key) s = (s + 1) & this.mask;
    return s;
  }

  private insert(s: number, key: number, val: number): void {
    this.used[s] = 1;
    this.keys[s] = key;
    this.vals[s] = val;
    this.count++;
    if (this.count * 2 > this.cap) this.grow();
  }

  private grow(): void {
    const ok = this.keys, ov = this.vals, ou = this.used, oc = this.cap;
    this.alloc(oc * 2);
    for (let i = 0; i < oc; i++) {
      if (ou[i] !== 1) continue;
      const s = this.slot(ok[i]);
      this.used[s] = 1;
      this.keys[s] = ok[i];
      this.vals[s] = ov[i];
    }
  }

  get(key: number): number {
    const s = this.slot(key);
    return this.used[s] === 1 ? this.vals[s] : -1;
  }

  /** Map.set: the last value wins. */
  set(key: number, val: number): void {
    const s = this.slot(key);
    if (this.used[s] === 1) this.vals[s] = val;
    else this.insert(s, key, val);
  }
}

/** A growable Int32 buffer for the pair output (pairs are appended as two int32 each). */
interface Growable {
  buf: Int32Array;
  len: number;
}

const pushPair = (out: Growable, a: number, b: number): void => {
  if (out.len + 2 > out.buf.length) {
    const grown = new Int32Array(out.buf.length * 2 + 2);
    grown.set(out.buf.subarray(0, out.len));
    out.buf = grown;
  }
  out.buf[out.len++] = a;
  out.buf[out.len++] = b;
};

/** The native layout of one step (one processor, orthogonal box): native order, ghosts, bins, bin chains. */
export interface NativeLayout {
  /** native local order: native atom k is engine atom order[k] */
  order: Int32Array;
  n: number;
  two: boolean;
  /** native positions of the owned atoms and of the ghosts (3 per atom) */
  X: Float64Array;
  /** the bin coordinates of each native atom (3 per atom) */
  binOf: Int32Array;
  /** bin key (binKey) -> bin number, -1 when no atom is in that bin */
  bins: IntTable;
  /** the bin number of each native atom */
  atomBin: Int32Array;
  /**
   * the bin chains as CSR: the native indices of bin b, in ascending order, are
   * members[chainStart[b] .. chainStart[b + 1])
   */
  chainStart: Int32Array;
  members: Int32Array;
  /** position of each native atom in its bin chain */
  posInBin: Int32Array;
  binsize: number[];
  bininv: number[];
  /** nstencil.cpp sx, sy, sz: the bin reach of the stencil */
  sx: number;
  sy: number;
  sz: number;
  cutmaxsq: number;
  /** nstencil_bin.cpp bin_distance: squared minimum distance from the central bin to bin (i, j, k) */
  binDist: (i: number, j: number, k: number) => number;
  /** engine index of native atom k (owned or ghost), or -1 when the engine holds no such image */
  gEng: Int32Array;
  /** the engine's neighbor-list index of each (owned engine atom, engine atom) membership key */
  engByKey: IntTable;
  NA: number;
}

const nativeLayout = (s: SimState, nb: Neighbor, g: Geometry): NativeLayout => {
  const order = nativeOrder(s);
  const n = s.n;
  const two = s.dimension === 2;
  // the engine's neighbor list keys (owned engine i, engine atom k) by their periodic image
  const NA = nb.nall;
  const gi = nb.gimage;
  const engByKey = new IntTable(NA);
  for (let k = 0; k < NA; k++) engByKey.set(imgKey(nb.owner[k], gi[3 * k], gi[3 * k + 1], gi[3 * k + 2]), k);
  // native positions and ghosts (native local order: atom k of the native list is engine atom order[k])
  const xn = new Float64Array(3 * n);
  for (let k = 0; k < n; k++) {
    const e = order[k];
    xn[3 * k] = s.x[3 * e]; xn[3 * k + 1] = s.x[3 * e + 1]; xn[3 * k + 2] = s.x[3 * e + 2];
  }
  const cutghost = nb.cutghost;
  const gh = nativeGhosts(xn, n, g, cutghost, two);
  const nall = gh.nall;
  const X = gh.x;

  // bins (nbin.cpp coord2bin with bboxlo = boxlo, bboxhi = boxhi; nbin_standard.cpp setup_bins)
  const lo = g.lo, hi = g.hi;
  const bbox = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  let binsizeOpt = nb.binsize > 0 ? nb.binsize : 0.5 * nb.cutneighmax;
  if (binsizeOpt === 0) binsizeOpt = bbox[0];
  const binsizeinv = 1.0 / binsizeOpt;
  const nbin = [0, 0, 0];
  const binsize = [0, 0, 0];
  const bininv = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    nbin[d] = Math.trunc(bbox[d] * binsizeinv);
    if (nbin[d] === 0) nbin[d] = 1;
    if (d === 2 && two) nbin[d] = 1;
    binsize[d] = bbox[d] / nbin[d];
    bininv[d] = 1.0 / binsize[d];
  }
  const coord = (x: number, d: number): number => {
    if (x >= hi[d]) return Math.trunc((x - hi[d]) * bininv[d]) + nbin[d];
    if (x >= lo[d]) return Math.min(Math.trunc((x - lo[d]) * bininv[d]), nbin[d] - 1);
    return Math.trunc((x - lo[d]) * bininv[d]) - 1;
  };
  const binOf = new Int32Array(nall * 3);
  const atomBin = new Int32Array(nall);
  const bins = new IntTable(nall);
  const binCount = new Int32Array(nall);
  let nbins = 0;
  for (let k = 0; k < nall; k++) {
    const bx = coord(X[3 * k], 0), by = coord(X[3 * k + 1], 1), bz = coord(X[3 * k + 2], 2);
    binOf[3 * k] = bx; binOf[3 * k + 1] = by; binOf[3 * k + 2] = bz;
    const key = binKey(bx, by, bz);
    let b = bins.get(key);
    if (b < 0) { b = nbins++; bins.set(key, b); }
    atomBin[k] = b;
    binCount[b]++;
  }
  const chainStart = new Int32Array(nbins + 1);
  for (let b = 0; b < nbins; b++) chainStart[b + 1] = chainStart[b] + binCount[b];
  const fill = chainStart.slice(0, nbins);
  const members = new Int32Array(nall);
  const posInBin = new Int32Array(nall);
  for (let k = 0; k < nall; k++) {
    const b = atomBin[k];
    posInBin[k] = fill[b] - chainStart[b];
    members[fill[b]++] = k;
  }

  // stencil reach (nstencil.cpp sx, sy, sz) and bin_distance (nstencil_bin.cpp)
  const cutmax = nb.cutneighmax;
  const reach = (d: number): number => {
    const r = Math.trunc(cutmax * bininv[d]);
    return r * binsize[d] < cutmax ? r + 1 : r;
  };
  const binDist = (i: number, j: number, k: number): number => {
    const dx = i > 0 ? (i - 1) * binsize[0] : i === 0 ? 0 : (i + 1) * binsize[0];
    const dy = j > 0 ? (j - 1) * binsize[1] : j === 0 ? 0 : (j + 1) * binsize[1];
    const dz = k > 0 ? (k - 1) * binsize[2] : k === 0 ? 0 : (k + 1) * binsize[2];
    return dx * dx + dy * dy + dz * dz;
  };
  // engine index of each native atom (owned: order[k]; ghost: the engine image, or -1 when there is none)
  const gEng = new Int32Array(nall);
  for (let k = 0; k < nall; k++) {
    if (k < n) { gEng[k] = order[k]; continue; }
    const e = order[gh.root[k]];
    gEng[k] = engByKey.get(imgKey(e, gh.img[3 * k], gh.img[3 * k + 1], gh.img[3 * k + 2]));
  }
  return {
    order, n, two, X, binOf, bins, atomBin, chainStart, members, posInBin, binsize, bininv,
    sx: reach(0), sy: reach(1), sz: two ? 0 : reach(2), cutmaxsq: cutmax * cutmax,
    binDist, gEng, engByKey, NA,
  };
};

/**
 * Membership rows: row r holds the (column, value) entries of engine atom r in first-insertion order. The original
 * Map kept the first value of each key ("if (!memb.has(key)) memb.set(key, m)"), so scattering a row into a dense
 * array (first entry wins) reproduces each lookup exactly.
 */
interface Rows {
  start: Int32Array;
  col: Int32Array;
  val: Int32Array;
}

/** Stable counting sort of the (row, column, value) insertions into rows (insertion order kept inside each row). */
const buildRows = (nrows: number, cnt: number, rowOf: Int32Array, colOf: Int32Array, valOf: Int32Array): Rows => {
  const start = new Int32Array(nrows + 1);
  for (let e = 0; e < cnt; e++) start[rowOf[e] + 1]++;
  for (let r = 0; r < nrows; r++) start[r + 1] += start[r];
  const fill = start.slice(0, nrows);
  const col = new Int32Array(cnt);
  const val = new Int32Array(cnt);
  for (let e = 0; e < cnt; e++) {
    const p = fill[rowOf[e]]++;
    col[p] = colOf[e];
    val[p] = valOf[e];
  }
  return { start, col, val };
};

/** dense[column] = value of the first entry of row r in that column (-1 where the row has none). */
const scatterRow = (rows: Rows, r: number, dense: Int32Array): void => {
  const end = rows.start[r + 1];
  for (let p = rows.start[r]; p < end; p++) {
    if (dense[rows.col[p]] < 0) dense[rows.col[p]] = rows.val[p];
  }
};

/** Undoes scatterRow for row r, leaving dense all -1. */
const clearRow = (rows: Rows, r: number, dense: Int32Array): void => {
  const end = rows.start[r + 1];
  for (let p = rows.start[r]; p < end; p++) dense[rows.col[p]] = -1;
};

/**
 * The pair (ei, engine partner j) with its pair code, appended when the engine's half list holds the pair (dense holds
 * the membership row of ei, see scatterRow).
 */
const takePair = (out: Growable, dense: Int32Array, gEng: Int32Array, ei: number, j: number): void => {
  const ek = gEng[j];
  if (ek < 0) return;
  const m = dense[ek];
  if (m < 0) return;
  pushPair(out, ei, ek | (m << SBBITS));
};

export const nativePairs = (s: SimState, nb: Neighbor, g: Geometry, half: NeighList, newton = true): Int32Array => {
  const L = nativeLayout(s, nb, g);
  const { order, n, two, X, binOf, bins, atomBin, chainStart, members, posInBin, sx, sy, sz, cutmaxsq, binDist, gEng, engByKey, NA } = L;
  const gi = nb.gimage;
  let total = 0;
  for (let i = 0; i < nb.nlocal; i++) total += half.numneigh[i];
  // membership insertions, in the original order: (i, k) with the half-list value, then the mirror (owner(k), kr)
  const rowOf = new Int32Array(2 * total), colOf = new Int32Array(2 * total), valOf = new Int32Array(2 * total);
  let cnt = 0;
  for (let i = 0; i < nb.nlocal; i++) {
    const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
    for (let q = k0; q < k1; q++) {
      const jj = half.neighbors[q];
      const k = jj & NEIGHMASK;
      const m = jj >>> SBBITS;
      rowOf[cnt] = i; colOf[cnt] = k; valOf[cnt] = m; cnt++;
      // the mirror of the pair (i, k) with image shift d is the pair (owner(k), the image of i with shift -d)
      const kr = engByKey.get(imgKey(i, -gi[3 * k], -gi[3 * k + 1], -gi[3 * k + 2]));
      if (kr >= 0) { rowOf[cnt] = nb.owner[k]; colOf[cnt] = kr; valOf[cnt] = m; cnt++; }
    }
  }
  const rows = buildRows(nb.nlocal, cnt, rowOf, colOf, valOf);
  // half newton stencil (nstencil.cpp sx, sy, sz; nstencil_bin.cpp create)
  // newton off: npair_bin.cpp with the full stencil (neighbor.cpp: "request a full stencil if building full neighbor list
  // or newton is off"), the central bin in its place, every bin in k, j, i order
  const stencil: number[] = [];
  if (!newton) {
    const szf = two ? 0 : sz;
    for (let k = -szf; k <= szf; k++) {
      for (let j = -sy; j <= sy; j++) {
        for (let i = -sx; i <= sx; i++) {
          if (binDist(i, j, k) < cutmaxsq) stencil.push(i, j, k);
        }
      }
    }
  } else if (!two) {
    for (let k = 0; k <= sz; k++) {
      for (let j = -sy; j <= sy; j++) {
        for (let i = -sx; i <= sx; i++) {
          if (k <= 0 && j <= 0 && (j !== 0 || i <= 0)) continue;
          if (binDist(i, j, k) < cutmaxsq) stencil.push(i, j, k);
        }
      }
    }
  } else {
    for (let j = 0; j <= sy; j++) {
      for (let i = -sx; i <= sx; i++) {
        if (j <= 0 && (j !== 0 || i <= 0)) continue;
        if (binDist(i, j, 0) < cutmaxsq) stencil.push(i, j, 0);
      }
    }
  }
  // npair_bin.cpp build(): own bin after i (ghosts above and to the right only), then the stencil bins
  const dense = new Int32Array(NA).fill(-1);
  const out: Growable = { buf: new Int32Array(Math.max(16, 2 * total)), len: 0 };
  for (let i = 0; i < n; i++) {
    const ei = order[i];
    scatterRow(rows, ei, dense);
    const xi = X[3 * i], yi = X[3 * i + 1], zi = X[3 * i + 2];
    const bx = binOf[3 * i], by = binOf[3 * i + 1], bz = binOf[3 * i + 2];
    if (!newton) {
      // npair_half_bin_newtoff: "only store pair if i < j" (own/own once), every ghost; the chains start at the bin head
      for (let st = 0; st < stencil.length; st += 3) {
        const b = bins.get(binKey(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]));
        if (b < 0) continue;
        for (let q = chainStart[b]; q < chainStart[b + 1]; q++) {
          const j = members[q];
          if (j > i) takePair(out, dense, gEng, ei, j);
        }
      }
    } else {
      const ob = atomBin[i];
      const ownEnd = chainStart[ob + 1];
      for (let q = chainStart[ob] + posInBin[i] + 1; q < ownEnd; q++) {
        const j = members[q];
        if (j >= n) {
          const zj = X[3 * j + 2], yj = X[3 * j + 1], xj = X[3 * j];
          if (zj < zi) continue;
          if (zj === zi) {
            if (yj < yi) continue;
            if (yj === yi && xj < xi) continue;
          }
        }
        takePair(out, dense, gEng, ei, j);
      }
      for (let st = 0; st < stencil.length; st += 3) {
        const b = bins.get(binKey(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]));
        if (b < 0) continue;
        for (let q = chainStart[b]; q < chainStart[b + 1]; q++) takePair(out, dense, gEng, ei, members[q]);
      }
    }
    clearRow(rows, ei, dense);
  }
  return out.buf.slice(0, out.len);
};

/**
 * npair_bin.cpp build() for a FULL list (HALF = 0, orthogonal box, one processor): for each owned atom, the engine index
 * of the partner that native's full list stores LAST (-1 when there is none). The full stencil of nstencil_bin.cpp create
 * (HALF = 0) is visited k, j, i ascending over the bins with bin_distance < cutneighmax^2, and each bin chain in order;
 * a candidate is kept when the engine's full list holds the pair. Native's compute_single_pair of pair spin/dipole/long
 * keeps the term of this partner only (see pair_spin_dipole_long.ts).
 */
export const nativeFullLast = (s: SimState, nb: Neighbor, g: Geometry, full: NeighList): Int32Array => {
  const L = nativeLayout(s, nb, g);
  const { order, n, binOf, bins, chainStart, members, sx, sy, sz, cutmaxsq, binDist, gEng, NA } = L;
  let total = 0;
  for (let i = 0; i < nb.nlocal; i++) total += full.numneigh[i];
  const rowOf = new Int32Array(total), colOf = new Int32Array(total), valOf = new Int32Array(total).fill(1);
  let cnt = 0;
  for (let i = 0; i < nb.nlocal; i++) {
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    for (let q = k0; q < k1; q++) { rowOf[cnt] = i; colOf[cnt] = full.neighbors[q] & NEIGHMASK; cnt++; }
  }
  const rows = buildRows(nb.nlocal, cnt, rowOf, colOf, valOf);
  const stencil: number[] = [];
  for (let k = -sz; k <= sz; k++) {
    for (let j = -sy; j <= sy; j++) {
      for (let i = -sx; i <= sx; i++) {
        if (binDist(i, j, k) < cutmaxsq) stencil.push(i, j, k);
      }
    }
  }
  const dense = new Int32Array(NA).fill(-1);
  const last = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const ei = order[i];
    scatterRow(rows, ei, dense);
    const bx = binOf[3 * i], by = binOf[3 * i + 1], bz = binOf[3 * i + 2];
    let found = -1;
    for (let st = 0; st < stencil.length; st += 3) {
      const b = bins.get(binKey(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]));
      if (b < 0) continue;
      for (let q = chainStart[b]; q < chainStart[b + 1]; q++) {
        const j = members[q];
        if (j === i) continue;
        const ek = gEng[j];
        if (ek >= 0 && dense[ek] >= 0) found = ek;
      }
    }
    clearRow(rows, ei, dense);
    last[ei] = found;
  }
  return last;
};

/**
 * The pair order of one step: rebuilt when the engine rebuilds its lists or the native order changes (sorting),
 * kept between rebuilds (native keeps its pair lists and only tests the cutoff each step).
 */
export class NativePairOrder {
  /** newton: the native pair enumeration of newton on (default) or newton off. */
  constructor(private readonly newton = true) {}
  private nbuild = -1;
  private order: Int32Array | null = null;
  private half: NeighList | null = null;
  private pairsCache: Int32Array = new Int32Array(0);

  /** Pairs as (i, jEnc) entries in native order, for the current neighbor lists. */
  pairs(s: SimState, nb: Neighbor, g: Geometry, half: NeighList | null): Int32Array {
    if (!half) throw new Error('native pair order needs a half neighbor list');
    if (this.nbuild === nb.nbuild && this.order === s.order && this.half === half) return this.pairsCache;
    this.pairsCache = g.triclinic ? engineOrder(half, nb.nlocal) : nativePairs(s, nb, g, half, this.newton);
    this.nbuild = nb.nbuild;
    this.order = s.order;
    this.half = half;
    return this.pairsCache;
  }
}

/** The engine's own list order (triclinic boxes, where the native enumeration is not reproduced). */
const engineOrder = (half: NeighList, nlocal: number): Int32Array => {
  const out: number[] = [];
  for (let i = 0; i < nlocal; i++) {
    const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
    for (let q = k0; q < k1; q++) out.push(i, half.neighbors[q]);
  }
  return Int32Array.from(out);
};
