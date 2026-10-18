/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/npair_bin.cpp (NPairBin::build for a
 * full list, the atom-only variant: the partners of each atom in the order of the bins of the stencil and of each bin
 * chain), src/nstencil_bin.cpp (NStencilBin::create for a full stencil: bins k, j, i ascending with bin_distance below
 * cutneighmax^2), src/nbin.cpp (NBin::coord2bin), src/nbin_standard.cpp (NBinStandard::setup_bins and bin_atoms: bins
 * of the box, chains in ascending index) and src/nstencil.cpp (sx, sy, sz). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Original authors: the LAMMPS development team (Sandia National Laboratories).
 *
 * Purpose: fix orient/fcc and orient/bcc keep the first 12 (or 8) neighbours of each atom in native's order when more
 * than that lie within the cutoff and ties in distance decide which are kept. The native neighbour list is built at
 * the neighbour builds (neighbor.ts nbuild), from the positions of that build; this module takes the order of the
 * candidates from the bins of the last build, and the membership from the engine's current list (within the cutoff).
 * Orthogonal boxes, one processor, as native runs. The membership is native's list, the order is native's.
 */
import type { Neighbor, NeighList } from '../../engine/neighbor';
import { NEIGHMASK } from '../../engine/neighbor';
import { nativeOrder } from '../../engine/atoms';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { nativeGhosts } from '../neighbor/nativeOrder';

/** nbin_standard.cpp bin keys: bins below 4096 in magnitude (as nativeOrder.ts). */
const OFF = 4096, BW = 8192;
const binKey = (ix: number, iy: number, iz: number): number => ((iz + OFF) * BW + (iy + OFF)) * BW + (ix + OFF);

/** The candidates of one build: for each engine atom, its candidate partners (engine indices) in native order. */
interface Candidates {
  start: Int32Array;
  end: Int32Array;
  idx: Int32Array;
}

export interface NativeFullRows {
  ptr: Int32Array;
  idx: Int32Array;
}

export class NativeFullOrder {
  private build = -1;
  private orderRef: Int32Array | null = null;
  private cand: Candidates | null = null;
  private inList = new Int32Array(0);
  private done = new Int32Array(0);
  private stamp = 0;

  /** The partners of each owned engine atom in native's full-list order, restricted to the engine's list row. */
  rows(s: SimState, nb: Neighbor, g: Geometry, list: NeighList): NativeFullRows {
    if (this.build !== nb.nbuild || this.orderRef !== s.order || !this.cand) {
      this.cand = candidatesOf(s, nb, g);
      this.build = nb.nbuild;
      this.orderRef = s.order;
    }
    const cand = this.cand;
    const nlocal = nb.nlocal, nall = nb.nall;
    if (this.inList.length !== nall) {
      this.inList = new Int32Array(nall);
      this.done = new Int32Array(nall);
      this.stamp = 0;
    }
    let total = 0;
    for (let i = 0; i < nlocal; i++) total += list.numneigh[i];
    const ptr = new Int32Array(nlocal + 1);
    const idx = new Int32Array(total);
    let out = 0;
    for (let i = 0; i < nlocal; i++) {
      ptr[i] = out;
      const st = ++this.stamp;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) this.inList[list.neighbors[kk] & NEIGHMASK] = st;
      for (let c = cand.start[i]; c < cand.end[i]; c++) {
        const e = cand.idx[c];
        if (this.inList[e] === st && this.done[e] !== st) {
          this.done[e] = st;
          idx[out++] = e;
        }
      }
      // members of the engine list the native candidates do not hold (none for a build that covers them) keep list order
      for (let kk = k0; kk < k1; kk++) {
        const e = list.neighbors[kk] & NEIGHMASK;
        if (this.done[e] !== st) {
          this.done[e] = st;
          idx[out++] = e;
        }
      }
    }
    ptr[nlocal] = out;
    return { ptr, idx: idx.subarray(0, out) };
  }
}

/** Engine-index key of a ghost image: its owner and its periodic image (nativeLayout's imgKey). */
const imageKey = (owner: number, ix: number, iy: number, iz: number): string => `${owner},${ix},${iy},${iz}`;

/** The native candidate partners of each owned atom at the current positions (the positions of the last build). */
const candidatesOf = (s: SimState, nb: Neighbor, g: Geometry): Candidates => {
  const order = nativeOrder(s);
  const n = s.n;
  const two = s.dimension === 2;
  const xn = new Float64Array(3 * n);
  for (let k = 0; k < n; k++) {
    const e = order[k];
    xn[3 * k] = s.x[3 * e]; xn[3 * k + 1] = s.x[3 * e + 1]; xn[3 * k + 2] = s.x[3 * e + 2];
  }
  const gh = nativeGhosts(xn, n, g, nb.cutghost, two);
  const nall = gh.nall;
  const X = gh.x;

  // the engine index of each native atom: owned atoms by order, ghosts by their owner and image
  const engByKey = new Map<string, number>();
  for (let k = 0; k < nb.nall; k++) {
    const gi = nb.gimage;
    engByKey.set(imageKey(nb.owner[k], gi[3 * k], gi[3 * k + 1], gi[3 * k + 2]), k);
  }
  const gEng = new Int32Array(nall);
  for (let k = 0; k < nall; k++) {
    if (k < n) { gEng[k] = order[k]; continue; }
    const e = order[gh.root[k]];
    const found = engByKey.get(imageKey(e, gh.img[3 * k], gh.img[3 * k + 1], gh.img[3 * k + 2]));
    gEng[k] = found ?? -1;
  }

  // bins (nbin.cpp coord2bin, nbin_standard.cpp setup_bins) over the box
  const lo = g.lo, hi = g.hi;
  const bbox = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  let binsizeOpt = nb.binsize > 0 ? nb.binsize : 0.5 * nb.cutneighmax;
  if (binsizeOpt === 0) binsizeOpt = bbox[0];
  const binsizeinv = 1.0 / binsizeOpt;
  const nbin = [0, 0, 0], binsize = [0, 0, 0], bininv = [0, 0, 0];
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
  const binOf = new Int32Array(3 * nall);
  const binIndex = new Map<number, number>();
  const atomBin = new Int32Array(nall);
  const counts: number[] = [];
  for (let k = 0; k < nall; k++) {
    const bx = coord(X[3 * k], 0), by = coord(X[3 * k + 1], 1), bz = coord(X[3 * k + 2], 2);
    binOf[3 * k] = bx; binOf[3 * k + 1] = by; binOf[3 * k + 2] = bz;
    const key = binKey(bx, by, bz);
    let b = binIndex.get(key);
    if (b === undefined) { b = counts.length; binIndex.set(key, b); counts.push(0); }
    atomBin[k] = b;
    counts[b]++;
  }
  // the chains in ascending native index (nbin_standard.cpp bin_atoms builds them from the top down)
  const nbins = counts.length;
  const chainStart = new Int32Array(nbins + 1);
  for (let b = 0; b < nbins; b++) chainStart[b + 1] = chainStart[b] + counts[b];
  const fill = chainStart.slice(0, nbins);
  const members = new Int32Array(nall);
  for (let k = 0; k < nall; k++) members[fill[atomBin[k]]++] = k;

  // the full stencil (nstencil.cpp sx, sy, sz; nstencil_bin.cpp bin_distance below cutneighmax^2, k, j, i ascending)
  const cutmax = nb.cutneighmax;
  const reach = (d: number): number => {
    const r = Math.trunc(cutmax * bininv[d]);
    return r * binsize[d] < cutmax ? r + 1 : r;
  };
  const sx = reach(0), sy = reach(1), sz = two ? 0 : reach(2);
  const cutmaxsq = cutmax * cutmax;
  const binDist = (i: number, j: number, k: number): number => {
    const dx = i > 0 ? (i - 1) * binsize[0] : i === 0 ? 0 : (i + 1) * binsize[0];
    const dy = j > 0 ? (j - 1) * binsize[1] : j === 0 ? 0 : (j + 1) * binsize[1];
    const dz = k > 0 ? (k - 1) * binsize[2] : k === 0 ? 0 : (k + 1) * binsize[2];
    return dx * dx + dy * dy + dz * dz;
  };
  const stencil: number[] = [];
  for (let k = -sz; k <= sz; k++) {
    for (let j = -sy; j <= sy; j++) {
      for (let i = -sx; i <= sx; i++) {
        if (binDist(i, j, k) < cutmaxsq) stencil.push(i, j, k);
      }
    }
  }

  // per owned native atom, its candidates in order: the chains of the stencil bins, within the neighbour cutoff
  const start = new Int32Array(n), end = new Int32Array(n);
  const flat: number[] = [];
  for (let k = 0; k < n; k++) {
    const e = order[k];
    start[e] = flat.length;
    const bx = binOf[3 * k], by = binOf[3 * k + 1], bz = binOf[3 * k + 2];
    for (let st = 0; st < stencil.length; st += 3) {
      const b = binIndex.get(binKey(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]));
      if (b === undefined) continue;
      for (let q = chainStart[b]; q < chainStart[b + 1]; q++) {
        const j = members[q];
        if (j === k) continue;
        const dx = X[3 * k] - X[3 * j], dy = X[3 * k + 1] - X[3 * j + 1], dz = X[3 * k + 2] - X[3 * j + 2];
        if (dx * dx + dy * dy + dz * dz > cutmaxsq) continue;
        const ej = gEng[j];
        if (ej >= 0) flat.push(ej);
      }
    }
    end[e] = flat.length;
  }
  return { start, end, idx: Int32Array.from(flat) };
};
