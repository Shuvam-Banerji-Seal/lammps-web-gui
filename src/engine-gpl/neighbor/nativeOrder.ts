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
  /** the bin chains: native indices in ascending order inside each bin */
  chains: Map<number, number[]>;
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
  engineIndex: (k: number) => number;
  /** the engine's neighbor-list index of each (owned engine atom, engine atom) membership key */
  engByKey: Map<number, number>;
  NA: number;
}

const nativeLayout = (s: SimState, nb: Neighbor, g: Geometry): NativeLayout => {
  const order = nativeOrder(s);
  const n = s.n;
  const two = s.dimension === 2;
  // the engine's neighbor list keys (owned engine i, engine atom k) by their periodic image
  const NA = nb.nall;
  const gi = nb.gimage;
  const engByKey = new Map<number, number>();
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
  const chains = new Map<number, number[]>();
  const posInBin = new Int32Array(nall);
  for (let k = 0; k < nall; k++) {
    const bx = coord(X[3 * k], 0), by = coord(X[3 * k + 1], 1), bz = coord(X[3 * k + 2], 2);
    binOf[3 * k] = bx; binOf[3 * k + 1] = by; binOf[3 * k + 2] = bz;
    const key = binKey(bx, by, bz);
    let chain = chains.get(key);
    if (!chain) { chain = []; chains.set(key, chain); }
    posInBin[k] = chain.length;
    chain.push(k);
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
  /** Engine index of native atom k (owned or ghost), or -1 when the engine holds no such image. */
  const engineIndex = (k: number): number => {
    const r = gh.root[k];
    const e = order[r];
    if (k < n) return e;
    const hit = engByKey.get(imgKey(e, gh.img[3 * k], gh.img[3 * k + 1], gh.img[3 * k + 2]));
    return hit === undefined ? -1 : hit;
  };
  return {
    order, n, two, X, binOf, chains, posInBin, binsize, bininv,
    sx: reach(0), sy: reach(1), sz: two ? 0 : reach(2), cutmaxsq: cutmax * cutmax,
    binDist, engineIndex, engByKey, NA,
  };
};

export const nativePairs = (s: SimState, nb: Neighbor, g: Geometry, half: NeighList, newton = true): Int32Array => {
  const L = nativeLayout(s, nb, g);
  const { order, n, two, X, binOf, chains, posInBin, sx, sy, sz, cutmaxsq, binDist, engineIndex, engByKey, NA } = L;
  const gi = nb.gimage;
  const memb = new Map<number, number>();
  for (let i = 0; i < nb.nlocal; i++) {
    const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
    for (let q = k0; q < k1; q++) {
      const jj = half.neighbors[q];
      const k = jj & NEIGHMASK;
      const m = jj >>> SBBITS;
      if (!memb.has(i * NA + k)) memb.set(i * NA + k, m);
      // the mirror of the pair (i, k) with image shift d is the pair (owner(k), the image of i with shift -d)
      const kr = engByKey.get(imgKey(i, -gi[3 * k], -gi[3 * k + 1], -gi[3 * k + 2]));
      if (kr !== undefined && !memb.has(nb.owner[k] * NA + kr)) memb.set(nb.owner[k] * NA + kr, m);
    }
  }
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
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const ei = order[i];
    const xi = X[3 * i], yi = X[3 * i + 1], zi = X[3 * i + 2];
    const bx = binOf[3 * i], by = binOf[3 * i + 1], bz = binOf[3 * i + 2];
    const own = chains.get(binKey(bx, by, bz))!;
    const take = (j: number): void => {
      const ek = engineIndex(j);
      if (ek < 0) return;
      const m = memb.get(ei * NA + ek);
      if (m === undefined) return;
      out.push(ei, ek | (m << SBBITS));
    };
    if (!newton) {
      // npair_half_bin_newtoff: "only store pair if i < j" (own/own once), every ghost; the chains start at the bin head
      for (let st = 0; st < stencil.length; st += 3) {
        const chain = chains.get(binKey(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]));
        if (!chain) continue;
        for (const j of chain) if (j > i) take(j);
      }
      continue;
    }
    for (let q = posInBin[i] + 1; q < own.length; q++) {
      const j = own[q];
      if (j >= n) {
        const zj = X[3 * j + 2], yj = X[3 * j + 1], xj = X[3 * j];
        if (zj < zi) continue;
        if (zj === zi) {
          if (yj < yi) continue;
          if (yj === yi && xj < xi) continue;
        }
      }
      take(j);
    }
    for (let st = 0; st < stencil.length; st += 3) {
      const chain = chains.get(binKey(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]));
      if (!chain) continue;
      for (const j of chain) take(j);
    }
  }
  return Int32Array.from(out);
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
  const { order, n, binOf, chains, sx, sy, sz, cutmaxsq, binDist, engineIndex, NA } = L;
  const memb = new Set<number>();
  for (let i = 0; i < nb.nlocal; i++) {
    const k0 = full.firstneigh[i], k1 = k0 + full.numneigh[i];
    for (let q = k0; q < k1; q++) memb.add(i * NA + (full.neighbors[q] & NEIGHMASK));
  }
  const stencil: number[] = [];
  for (let k = -sz; k <= sz; k++) {
    for (let j = -sy; j <= sy; j++) {
      for (let i = -sx; i <= sx; i++) {
        if (binDist(i, j, k) < cutmaxsq) stencil.push(i, j, k);
      }
    }
  }
  const last = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const ei = order[i];
    const bx = binOf[3 * i], by = binOf[3 * i + 1], bz = binOf[3 * i + 2];
    let found = -1;
    for (let st = 0; st < stencil.length; st += 3) {
      const chain = chains.get(binKey(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]));
      if (!chain) continue;
      for (const j of chain) {
        if (j === i) continue;
        const ek = engineIndex(j);
        if (ek >= 0 && memb.has(ei * NA + ek)) found = ek;
      }
    }
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
