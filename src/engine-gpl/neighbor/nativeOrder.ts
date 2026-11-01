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
import { ANGSTROM_PER_UNIT } from '../../engine/lepton/units';
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
  /** Ghost k: the native index of the atom (owned or an earlier ghost) whose coordinates were sent; -1 for owned atoms. */
  from: Int32Array;
}

/** comm_brick.cpp CommBrick::setup (cutghost = comm cutoff, maxneed = 1 + cut/prd for one processor) and borders(). */
export const nativeGhosts = (xn: Float64Array, n: number, g: Geometry, cut: number, two: boolean): NativeGhosts => {
  const xs: number[] = Array.from(xn.subarray(0, 3 * n));
  const roots: number[] = [];
  const imgs: number[] = [];
  const froms: number[] = [];
  for (let i = 0; i < n; i++) { roots.push(i); imgs.push(0, 0, 0); froms.push(-1); }
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
        froms.push(i);
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
    from: Int32Array.from(froms),
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
  /** the bin number of bin (bx, by, bz), -1 when no atom is in it: bins.get(binKey(bx, by, bz)) from a dense grid */
  binAt: (bx: number, by: number, bz: number) => number;
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
  /** the native index of the owned atom each native atom is (k for an owned atom) or is an image of (NativeGhosts.root) */
  root: Int32Array;
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
  // the occupied bins as a dense grid over their bounding range (each stencil lookup then indexes it instead of
  // hashing), unless that range is very large for the number of atoms
  let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let k = 0; k < nall; k++) {
    const bx = binOf[3 * k], by = binOf[3 * k + 1], bz = binOf[3 * k + 2];
    if (bx < mnx) mnx = bx;
    if (bx > mxx) mxx = bx;
    if (by < mny) mny = by;
    if (by > mxy) mxy = by;
    if (bz < mnz) mnz = bz;
    if (bz > mxz) mxz = bz;
  }
  const gx = mxx - mnx + 1, gy = mxy - mny + 1, gz = mxz - mnz + 1;
  const cells = nall > 0 ? gx * gy * gz : 0;
  let binAt: (bx: number, by: number, bz: number) => number;
  if (nall > 0 && cells <= Math.max(1 << 22, 64 * nall)) {
    const grid = new Int32Array(cells).fill(-1);
    for (let k = 0; k < nall; k++) grid[((binOf[3 * k + 2] - mnz) * gy + (binOf[3 * k + 1] - mny)) * gx + (binOf[3 * k] - mnx)] = atomBin[k];
    binAt = (bx, by, bz) => {
      const x = bx - mnx, y = by - mny, z = bz - mnz;
      if (x < 0 || y < 0 || z < 0 || x >= gx || y >= gy || z >= gz) return -1;
      return grid[(z * gy + y) * gx + x];
    };
  } else {
    binAt = (bx, by, bz) => bins.get(binKey(bx, by, bz));
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
    order, n, two, X, binOf, bins, binAt, atomBin, chainStart, members, posInBin, binsize, bininv,
    sx: reach(0), sy: reach(1), sz: two ? 0 : reach(2), cutmaxsq: cutmax * cutmax,
    binDist, gEng, root: gh.root, engByKey, NA,
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

/**
 * How native builds the half list whose pairs are enumerated: by bins (npair_bin.cpp, the default) or all pairs
 * (neighbor nsq: npair_nsq.cpp), and directly or derived from a perpetual full list (halffull: neighbor.cpp
 * morph_halffull and npair_halffull.cpp), whose own rows are built by bins or nsq.
 */
export interface NativeListBuild {
  nsq: boolean;
  halffull: boolean;
}

export const nativePairs = (s: SimState, nb: Neighbor, g: Geometry, half: NeighList, newton = true,
  build: NativeListBuild = { nsq: false, halffull: false }): Int32Array => {
  const L = nativeLayout(s, nb, g);
  const { order, n, two, X, binOf, bins, binAt, atomBin, chainStart, members, posInBin, sx, sy, sz, cutmaxsq, binDist, gEng, root, engByKey, NA } = L;
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
  // or newton is off"), the central bin in its place, every bin in k, j, i order; a half list derived from a full one
  // walks the full list's rows, which use that stencil too
  const stencil: number[] = [];
  if (!newton || build.halffull) {
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
  const nall = gEng.length;
  // the tag of native atom k: its owned atom's
  const tagOf = (k: number): number => s.id[order[root[k]]];
  for (let i = 0; i < n; i++) {
    const ei = order[i];
    scatterRow(rows, ei, dense);
    const xi = X[3 * i], yi = X[3 * i + 1], zi = X[3 * i + 2];
    const bx = binOf[3 * i], by = binOf[3 * i + 1], bz = binOf[3 * i + 2];
    // npair_halffull.cpp build, newton on (NPairHalffull<1, 0, 0>): the parent full list's row of i, "if (j < nlocal) {
    // if (i > j) continue; }" and a ghost kept "above" i ("if (x[j][2] < ztmp) continue; if (x[j][2] == ztmp) { if
    // (x[j][1] < ytmp) continue; if (x[j][1] == ytmp && x[j][0] < xtmp) continue; }"); newton off keeps "j > i", which
    // the half newton off enumeration below already gives
    const keepHalffull = (j: number): boolean => {
      if (j < n) return j > i;
      const zj = X[3 * j + 2], yj = X[3 * j + 1], xj = X[3 * j];
      if (zj < zi) return false;
      if (zj === zi) {
        if (yj < yi) return false;
        if (yj === yi && xj < xi) return false;
      }
      return true;
    };
    if (build.nsq) {
      if (build.halffull && newton) {
        // npair_nsq.cpp, full list (HALF = 0): "for (j = 0; j < nall; j++)" with "if (i == j) continue;"
        for (let j = 0; j < nall; j++) if (j !== i && keepHalffull(j)) takePair(out, dense, gEng, ei, j);
      } else {
        // npair_nsq.cpp, half list: "jstart = i + 1"; newton on, a ghost by the tag parity rule ("if (itag > jtag) { if
        // ((itag + jtag) % 2 == 0) continue; } else if (itag < jtag) { if ((itag + jtag) % 2 == 1) continue; }") and an
        // image of i itself by the coordinate rule
        const itag = tagOf(i);
        for (let j = i + 1; j < nall; j++) {
          if (newton && j >= n) {
            const jtag = tagOf(j);
            if (itag > jtag) {
              if ((itag + jtag) % 2 === 0) continue;
            } else if (itag < jtag) {
              if ((itag + jtag) % 2 === 1) continue;
            } else if (!keepHalffull(j)) continue;
          }
          takePair(out, dense, gEng, ei, j);
        }
      }
    } else if (build.halffull && newton) {
      // npair_bin.cpp, full list: every stencil bin (the central one in its place), "if (i == j) continue;"
      for (let st = 0; st < stencil.length; st += 3) {
        const b = binAt(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]);
        if (b < 0) continue;
        for (let q = chainStart[b]; q < chainStart[b + 1]; q++) {
          const j = members[q];
          if (j !== i && keepHalffull(j)) takePair(out, dense, gEng, ei, j);
        }
      }
    } else if (!newton) {
      // npair_half_bin_newtoff: "only store pair if i < j" (own/own once), every ghost; the chains start at the bin head
      for (let st = 0; st < stencil.length; st += 3) {
        const b = binAt(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]);
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
        const b = binAt(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]);
        if (b < 0) continue;
        for (let q = chainStart[b]; q < chainStart[b + 1]; q++) takePair(out, dense, gEng, ei, members[q]);
      }
    }
    clearRow(rows, ei, dense);
  }
  return out.buf.slice(0, out.len);
};

/** The size-list settings of NativePairOrder (pair styles that request NeighConst::REQ_SIZE). */
export interface SizeListSpec {
  /** the type pairs the style owns (a sub-style of pair hybrid gets the skip list of its own type pairs) */
  allow: (itype: number, jtype: number) => boolean;
}

/**
 * npair_bin.cpp build() for a SIZE list (SIZE = 1, orthogonal box, one processor): the candidates of the bin stencil
 * in native's order, as nativePairs walks them (half stencil with newton on; with newton off the full stencil, each
 * owned pair once and every ghost pair), kept when "radsum = radius[i] + radius[j]; cut = radsum + skin; ... if (rsq
 * <= cutsq)" by the radii and positions of the build, after the neigh_modify exclusions; no type cutoff, as native.
 * The membership is native's, not the engine's list (the radii of styles such as smd/ulsph change between builds, so
 * native's list holds pairs beyond the cutoff of the last init). Pairs whose ghost the engine does not hold are left out.
 */
export const nativeSizePairs = (s: SimState, nb: Neighbor, g: Geometry, newton: boolean, spec: SizeListSpec): Int32Array => {
  const L = nativeLayout(s, nb, g);
  const { order, n, two, X, binOf, bins, binAt, atomBin, chainStart, members, posInBin, sx, sy, sz, cutmaxsq, binDist, gEng } = L;
  const radius = s.radius;
  if (!radius) throw new Error('native size list needs per-atom radii');
  const owner = nb.owner;
  const skin = nb.skin;
  const exclude = nb.exclusion(s);
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
  const out: Growable = { buf: new Int32Array(64), len: 0 };
  for (let i = 0; i < n; i++) {
    const ei = order[i];
    const xi = X[3 * i], yi = X[3 * i + 1], zi = X[3 * i + 2];
    const ri = radius[ei], ti = s.type[ei];
    const consider = (j: number): void => {
      const ek = gEng[j];
      if (ek < 0) return;
      const oj = owner[ek];
      if (!spec.allow(ti, s.type[oj])) return;
      if (exclude && exclude(ei, oj)) return;
      const dx = xi - X[3 * j], dy = yi - X[3 * j + 1], dz = zi - X[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const cut = ri + radius[oj] + skin;
      if (rsq <= cut * cut) pushPair(out, ei, ek);
    };
    const bx = binOf[3 * i], by = binOf[3 * i + 1], bz = binOf[3 * i + 2];
    if (!newton) {
      for (let st = 0; st < stencil.length; st += 3) {
        const b = binAt(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]);
        if (b < 0) continue;
        for (let q = chainStart[b]; q < chainStart[b + 1]; q++) {
          const j = members[q];
          if (j > i) consider(j);
        }
      }
    } else {
      const ob = atomBin[i];
      for (let q = chainStart[ob] + posInBin[i] + 1; q < chainStart[ob + 1]; q++) {
        const j = members[q];
        if (j >= n) {
          const zj = X[3 * j + 2], yj = X[3 * j + 1], xj = X[3 * j];
          if (zj < zi) continue;
          if (zj === zi) {
            if (yj < yi) continue;
            if (yj === yi && xj < xi) continue;
          }
        }
        consider(j);
      }
      for (let st = 0; st < stencil.length; st += 3) {
        const b = binAt(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]);
        if (b < 0) continue;
        for (let q = chainStart[b]; q < chainStart[b + 1]; q++) consider(members[q]);
      }
    }
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
  const { order, n, binOf, binAt, chainStart, members, sx, sy, sz, cutmaxsq, binDist, gEng, NA } = L;
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
      const b = binAt(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]);
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
 * npair_bin.cpp build() for a FULL list (HALF = 0, orthogonal box, one processor): the entries of the engine's full list
 * with each owned atom's row in native's order. The full stencil of nstencil_bin.cpp create (HALF = 0) is visited k, j,
 * i ascending over the bins with bin_distance < cutneighmax^2, and each bin chain in order. Rows keep the engine's
 * firstneigh and numneigh; an entry the native layout does not reach keeps its engine order after the native ones.
 */
export const nativeFullOrder = (s: SimState, nb: Neighbor, g: Geometry, full: NeighList): Int32Array => {
  const L = nativeLayout(s, nb, g);
  const { order, n, binOf, binAt, chainStart, members, sx, sy, sz, cutmaxsq, binDist, gEng, NA } = L;
  const out = full.neighbors.slice();
  const stencil: number[] = [];
  for (let k = -sz; k <= sz; k++) {
    for (let j = -sy; j <= sy; j++) {
      for (let i = -sx; i <= sx; i++) {
        if (binDist(i, j, k) < cutmaxsq) stencil.push(i, j, k);
      }
    }
  }
  // slot[ek]: the list slot of engine atom ek in the current row, -1 when it is not (or no longer) in the row
  const slot = new Int32Array(NA).fill(-1);
  for (let i = 0; i < n; i++) {
    const ei = order[i];
    const k0 = full.firstneigh[ei], k1 = k0 + full.numneigh[ei];
    for (let q = k0; q < k1; q++) slot[full.neighbors[q] & NEIGHMASK] = q;
    let w = k0;
    const bx = binOf[3 * i], by = binOf[3 * i + 1], bz = binOf[3 * i + 2];
    for (let st = 0; st < stencil.length; st += 3) {
      const b = binAt(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]);
      if (b < 0) continue;
      for (let q = chainStart[b]; q < chainStart[b + 1]; q++) {
        const j = members[q];
        if (j === i) continue;
        const ek = gEng[j];
        if (ek < 0 || slot[ek] < 0) continue;
        out[w++] = full.neighbors[slot[ek]];
        slot[ek] = -1;
      }
    }
    for (let q = k0; q < k1; q++) {
      const ek = full.neighbors[q] & NEIGHMASK;
      if (slot[ek] >= 0) { out[w++] = full.neighbors[q]; slot[ek] = -1; }
    }
  }
  return out;
};

/**
 * The native full-list order (nativeFullOrder) of one step: rebuilt when the engine rebuilds its lists or the native
 * order changes, kept between rebuilds. Triclinic boxes keep the engine's order.
 */
export class NativeFullOrder {
  private nbuild = -1;
  private order: Int32Array | null = null;
  private full: NeighList | null = null;
  private cache: Int32Array = new Int32Array(0);

  /** The full list's entries with each owned atom's row in native order (indexed by the list's firstneigh/numneigh). */
  neighbors(s: SimState, nb: Neighbor, g: Geometry, full: NeighList): Int32Array {
    if (g.triclinic) return full.neighbors;
    if (this.nbuild === nb.nbuild && this.order === s.order && this.full === full) return this.cache;
    this.cache = nativeFullOrder(s, nb, g, full);
    this.nbuild = nb.nbuild;
    this.order = s.order;
    this.full = full;
    return this.cache;
  }
}

/**
 * The pair order of one step: rebuilt when the engine rebuilds its lists or the native order changes (sorting),
 * kept between rebuilds (native keeps its pair lists and only tests the cutoff each step).
 */
export class NativePairOrder {
  /**
   * newton: the native pair enumeration of newton on (default) or newton off. size: a size list (NeighConst::REQ_SIZE)
   * whose membership is native's (nativeSizePairs) rather than the engine list's.
   */
  constructor(private readonly newton = true, private readonly size: SizeListSpec | null = null) {}
  private nbuild = -1;
  private order: Int32Array | null = null;
  private half: NeighList | null = null;
  private pairsCache: Int32Array = new Int32Array(0);

  private build = -1;

  /**
   * Pairs as (i, jEnc) entries in native order, for the current neighbor lists: by bins or nsq (Neighbor.nsqStyle), and
   * from a half list native derives from a fix's perpetual full list when there is one (Neighbor.nativeFullParent).
   */
  pairs(s: SimState, nb: Neighbor, g: Geometry, half: NeighList | null): Int32Array {
    if (!half) throw new Error('native pair order needs a half neighbor list');
    const build: NativeListBuild = { nsq: nb.nsqStyle, halffull: nb.nativeFullParent };
    const key = (build.nsq ? 1 : 0) | (build.halffull ? 2 : 0);
    if (this.nbuild === nb.nbuild && this.order === s.order && this.half === half && this.build === key) return this.pairsCache;
    // force->angstrom of the unit style (update.cpp set_units): the 0.01 Angstrom delta of the triclinic tie rule
    const angstrom = 1 / (ANGSTROM_PER_UNIT[s.units.style] ?? 1);
    this.pairsCache = g.triclinic
      ? (build.nsq && !build.halffull && this.newton && !this.size ? nsqTriPairs(s, nb, half, 0.01 * angstrom) : engineOrder(half, nb.nlocal))
      : this.size ? nativeSizePairs(s, nb, g, this.newton, this.size) : nativePairs(s, nb, g, half, this.newton, build);
    this.nbuild = nb.nbuild;
    this.order = s.order;
    this.half = half;
    this.build = key;
    return this.pairsCache;
  }
}

/**
 * npair_nsq.cpp build, half list, newton on, triclinic box (NPairNsq<1, 1, 1, 0>), for one processor: each pair of the
 * engine's half list goes to the row native gives it. Two owned atoms: the row of the one first in native's local order
 * ("jstart = i + 1"). An owned atom and an image of another: the row of i where "if (itag > jtag) { if ((itag + jtag) % 2
 * == 0) continue; } else if (itag < jtag) { if ((itag + jtag) % 2 == 1) continue; }" keeps it, else the mirror pair in the
 * row of the other atom; an image of i itself by the triclinic rule ("if (fabs(x[j][2] - ztmp) > delta) { if (x[j][2] <
 * ztmp) continue; } else if (fabs(x[j][1] - ytmp) > delta) { if (x[j][1] < ytmp) continue; } else { if (x[j][0] < xtmp)
 * continue; }", delta = 0.01 force->angstrom). Rows follow native's local order and an owned partner native's index;
 * the ghosts of a row keep the engine's order (native's triclinic ghosts are not built here), which leaves the rows' atoms,
 * the pairs' orientation and numforce as native's.
 */
const nsqTriPairs = (s: SimState, nb: Neighbor, half: NeighList, delta: number): Int32Array => {
  const order = nativeOrder(s);
  const nlocal = nb.nlocal;
  const rank = new Int32Array(nlocal);
  for (let k = 0; k < order.length; k++) rank[order[k]] = k;
  const NA = nb.nall, gi = nb.gimage, xa = nb.xall, owner = nb.owner, tag = s.id;
  const engByKey = new IntTable(NA);
  for (let k = 0; k < NA; k++) engByKey.set(imgKey(owner[k], gi[3 * k], gi[3 * k + 1], gi[3 * k + 2]), k);
  // the pairs of each row: owned partners (sorted by native index below), then ghosts in the engine's order
  const ownRows: number[][] = Array.from({ length: nlocal }, () => []);
  const ghostRows: number[][] = Array.from({ length: nlocal }, () => []);
  const keepGhost = (i: number, k: number): boolean => {
    const itag = tag[i], jtag = tag[owner[k]];
    if (itag > jtag) return (itag + jtag) % 2 !== 0;
    if (itag < jtag) return (itag + jtag) % 2 !== 1;
    const dz = xa[3 * k + 2] - xa[3 * i + 2], dy = xa[3 * k + 1] - xa[3 * i + 1];
    if (Math.abs(dz) > delta) return dz >= 0;
    if (Math.abs(dy) > delta) return dy >= 0;
    return xa[3 * k] >= xa[3 * i];
  };
  for (let i = 0; i < nlocal; i++) {
    const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
    for (let q = k0; q < k1; q++) {
      const jEnc = half.neighbors[q];
      const k = jEnc & NEIGHMASK;
      const code = jEnc & ~NEIGHMASK;
      if (k < nlocal) {
        if (rank[i] < rank[k]) ownRows[i].push(k | code); else ownRows[k].push(i | code);
      } else if (keepGhost(i, k)) {
        ghostRows[i].push(jEnc);
      } else {
        // the mirror: the image of i seen from the owner of k, with the opposite shift
        const kr = engByKey.get(imgKey(i, -gi[3 * k], -gi[3 * k + 1], -gi[3 * k + 2]));
        if (kr >= 0) ghostRows[owner[k]].push(kr | code); else ghostRows[i].push(jEnc);
      }
    }
  }
  const out: Growable = { buf: new Int32Array(16), len: 0 };
  for (let r = 0; r < order.length; r++) {
    const i = order[r];
    ownRows[i].sort((a, b) => rank[a & NEIGHMASK] - rank[b & NEIGHMASK]);
    for (const jEnc of ownRows[i]) pushPair(out, i, jEnc);
    for (const jEnc of ghostRows[i]) pushPair(out, i, jEnc);
  }
  return out.buf.slice(0, out.len);
};

/** The engine's own list order (triclinic boxes, where the native enumeration is not reproduced). */
const engineOrder = (half: NeighList, nlocal: number): Int32Array => {
  const out: number[] = [];
  for (let i = 0; i < nlocal; i++) {
    const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
    for (let q = k0; q < k1; q++) out.push(i, half.neighbors[q]);
  }
  return Int32Array.from(out);
};

/**
 * comm_brick.cpp CommBrick::reverse_comm with one processor, where every swap is to self ("for (int iswap = nswap-1;
 * iswap >= 0; iswap--)" ... "avec->unpack_reverse(sendnum[iswap],sendlist[iswap],buf_send)": the values of the ghosts
 * of swap iswap are added to the atoms they were sent from), and pair.cpp Pair::virial_fdotr_compute ("sum over force
 * on all particles including ghosts": owned atoms, then ghosts, in local order). Inside one swap every target receives
 * one value, so folding the ghosts from the last to the first gives native's sums bit for bit.
 *
 * Engine arrays are indexed by the engine's atoms (owned, then its ghosts). A native atom maps to the engine atom with
 * the same owner and periodic image; a native ghost that the engine does not hold carries zero, and an engine ghost
 * with no native counterpart is folded the engine's way. The mapping is kept between neighbor builds, as native keeps
 * its ghost lists (borders) between reneighborings. Orthogonal boxes, newton on.
 */
export class NativeReverseComm {
  private nbuild = -1;
  private order: Int32Array | null = null;
  private n = 0;
  /** engine index of each native atom (-1: the engine holds no such image) */
  private eng: Int32Array = new Int32Array(0);
  /** native index of the atom each native ghost was sent from */
  private from: Int32Array = new Int32Array(0);
  /** engine ghosts that native does not hold */
  private extra: number[] = [];
  private buf = new Float64Array(0);

  private refresh(s: SimState, nb: Neighbor, g: Geometry): void {
    if (this.nbuild === nb.nbuild && this.order === s.order) return;
    const order = nativeOrder(s);
    const n = s.n;
    const gi = nb.gimage;
    const engByKey = new Map<number, number>();
    for (let k = 0; k < nb.nall; k++) engByKey.set(imgKey(nb.owner[k], gi[3 * k], gi[3 * k + 1], gi[3 * k + 2]), k);
    const xn = new Float64Array(3 * n);
    for (let k = 0; k < n; k++) {
      const e = order[k];
      xn[3 * k] = s.x[3 * e]; xn[3 * k + 1] = s.x[3 * e + 1]; xn[3 * k + 2] = s.x[3 * e + 2];
    }
    const gh = nativeGhosts(xn, n, g, nb.cutghost, s.dimension === 2);
    const eng = new Int32Array(gh.nall);
    const held = new Uint8Array(nb.nall);
    for (let k = 0; k < gh.nall; k++) {
      const e = order[gh.root[k]];
      const hit = k < n ? e : engByKey.get(imgKey(e, gh.img[3 * k], gh.img[3 * k + 1], gh.img[3 * k + 2]));
      eng[k] = hit === undefined ? -1 : hit;
      if (hit !== undefined) held[hit] = 1;
    }
    this.extra = [];
    for (let e = nb.nlocal; e < nb.nall; e++) if (!held[e]) this.extra.push(e);
    this.eng = eng;
    this.from = gh.from;
    this.n = n;
    if (this.buf.length < 3 * gh.nall) this.buf = new Float64Array(3 * gh.nall);
    this.nbuild = nb.nbuild;
    this.order = s.order;
  }

  /** The values of a (3 per engine atom) in native order, folded as reverse_comm; the owned atoms' sums are in buf[0, 3n). */
  private gatherFold(a: Float64Array): void {
    const { eng, from, n, buf } = this;
    const nall = eng.length;
    for (let k = 0; k < nall; k++) {
      const e = eng[k];
      if (e < 0) { buf[3 * k] = 0; buf[3 * k + 1] = 0; buf[3 * k + 2] = 0; continue; }
      buf[3 * k] = a[3 * e]; buf[3 * k + 1] = a[3 * e + 1]; buf[3 * k + 2] = a[3 * e + 2];
    }
    for (let k = nall - 1; k >= n; k--) {
      const t = from[k];
      buf[3 * t] += buf[3 * k]; buf[3 * t + 1] += buf[3 * k + 1]; buf[3 * t + 2] += buf[3 * k + 2];
    }
  }

  /**
   * Pair::virial_fdotr_compute over the pair forces f (3 per engine atom, owned and ghosts) at positions x, in native
   * order, added to virial (xx, yy, zz, xy, xz, yz); then f is folded in place as reverse_comm: the owned entries hold
   * the folded sums and the ghost entries native holds are zero (the engine's own fold then adds nothing).
   */
  fdotrFold(s: SimState, nb: Neighbor, g: Geometry, x: Float64Array, f: Float64Array, virial: Float64Array): void {
    this.refresh(s, nb, g);
    const { eng, n, buf } = this;
    const nall = eng.length;
    let v0 = virial[0], v1 = virial[1], v2 = virial[2], v3 = virial[3], v4 = virial[4], v5 = virial[5];
    for (let k = 0; k < nall; k++) {
      const e = eng[k];
      if (e < 0) continue;
      const fx = f[3 * e], fy = f[3 * e + 1], fz = f[3 * e + 2];
      const xx = x[3 * e], xy = x[3 * e + 1], xz = x[3 * e + 2];
      v0 += fx * xx; v1 += fy * xy; v2 += fz * xz; v3 += fy * xx; v4 += fz * xx; v5 += fz * xy;
    }
    for (const e of this.extra) {
      const fx = f[3 * e], fy = f[3 * e + 1], fz = f[3 * e + 2];
      const xx = x[3 * e], xy = x[3 * e + 1], xz = x[3 * e + 2];
      v0 += fx * xx; v1 += fy * xy; v2 += fz * xz; v3 += fy * xx; v4 += fz * xx; v5 += fz * xy;
    }
    virial[0] = v0; virial[1] = v1; virial[2] = v2; virial[3] = v3; virial[4] = v4; virial[5] = v5;
    this.gatherFold(f);
    for (let k = 0; k < n; k++) {
      const e = eng[k];
      f[3 * e] = buf[3 * k]; f[3 * e + 1] = buf[3 * k + 1]; f[3 * e + 2] = buf[3 * k + 2];
    }
    for (let k = n; k < nall; k++) {
      const e = eng[k];
      if (e >= 0) { f[3 * e] = 0; f[3 * e + 1] = 0; f[3 * e + 2] = 0; }
    }
  }

  /** Folds a (3 per engine atom) as reverse_comm and adds the owned sums to out (3 per owned atom). */
  foldInto(s: SimState, nb: Neighbor, g: Geometry, a: Float64Array, out: Float64Array): void {
    this.refresh(s, nb, g);
    const { eng, n, buf } = this;
    this.gatherFold(a);
    for (let k = 0; k < n; k++) {
      const e = eng[k];
      out[3 * e] += buf[3 * k]; out[3 * e + 1] += buf[3 * k + 1]; out[3 * e + 2] += buf[3 * k + 2];
    }
    const o = nb.owner;
    for (const e of this.extra) {
      const i = o[e];
      out[3 * i] += a[3 * e]; out[3 * i + 1] += a[3 * e + 1]; out[3 * i + 2] += a[3 * e + 2];
    }
  }
}
