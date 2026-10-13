/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/npair_bin_ghost.cpp (NPairBinGhost::build, half list with newton off: "only store pair if i < j",
 * owned atoms check every bin of the stencil, ghosts are stored when j is above i), src/nstencil_ghost_bin.cpp
 * (NStencilGhostBin::create: the full stencil in k, j, i order with bin_distance < cutneighmax), and
 * src/nbin_standard.cpp / src/nbin.cpp (the bin geometry and the ascending bin chains), at commit b332f28.
 * Used by src/REAXFF/pair_reaxff.cpp (write_reax_lists iterates the ghost neighbor list in this order) and by
 * src/REAXFF/fix_reaxff_bonds.cpp (the order of the bond list of an atom follows it). Copyright (2003) Sandia
 * Corporation (GNU General Public License); the LAMMPS development team is named in those files.
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * What it gives: for every owned atom i (engine index), the rank of each partner j (engine index, owned or
 * ghost) in the far neighbor list that native LAMMPS builds for i at the last neighbor build, so that the
 * bond list of an atom can be written in native order. The ranks are computed from the positions of the build
 * step (native keeps its neighbor list between builds). Orthogonal boxes only (triclinic boxes are refused by
 * the caller).
 *
 * Triclinic boxes (the TATB example): native converts the owned atoms to lamda coordinates (domain.cpp x2lamda),
 * creates the ghosts in lamda space with the cutoffs of CommBrick::setup (cut * the inverse box lengths, prd = 1),
 * converts back (lamda2x) and bins in the bounding box of the tilted cell (Domain::set_global_box boxlo_bound and
 * boxhi_bound), with the same stencil as the orthogonal box.
 */
import type { Geometry } from '../../engine/domain';
import type { Neighbor } from '../../engine/neighbor';
import { nativeOrder } from '../../engine/atoms';
import type { SimState } from '../../engine/types';

/** Integer key of an owner and its periodic image (the same key as neighbor/nativeOrder.ts). */
const imgKey = (owner: number, ix: number, iy: number, iz: number): number =>
  ((owner * 64 + (ix + 32)) * 64 + (iy + 32)) * 64 + (iz + 32);

const OFF = 4096, BW = 8192;
const binKey = (ix: number, iy: number, iz: number): number => ((iz + OFF) * BW + (iy + OFF)) * BW + (ix + OFF);

export interface NativeFarOrder {
  /** owned engine index -> native local index. */
  nativeIdx: Int32Array;
  /** key (i * nall + j, engine indices) -> rank of j in the native far list of i (only j above i). */
  rank: Map<number, number>;
  nall: number;
}

/** CommBrick::borders for one processor: the ghosts of owned atoms in coordinates c (3 per atom), per dimension
 *  and swap, in the order of the native swaps; root = native index of the owner, img = image offsets. */
const borders = (c0: Float64Array, n: number, g: Geometry, lo: number[], hi: number[], prd: number[], cut: number[]) => {
  const xs: number[] = Array.from(c0.subarray(0, 3 * n));
  const roots: number[] = [];
  const imgs: number[] = [];
  for (let i = 0; i < n; i++) { roots.push(i); imgs.push(0, 0, 0); }
  for (let dim = 0; dim < 3; dim++) {
    if (!g.periodic[dim]) continue;
    const sublo = lo[dim], subhi = hi[dim], len = prd[dim], cd = cut[dim];
    const maxneed = Math.trunc(cd / len) + 1;
    const mid = 0.5 * (sublo + subhi);
    let nfirst = 0, nlast = 0;
    for (let ineed = 0; ineed < 2 * maxneed; ineed++) {
      const even = ineed % 2 === 0;
      if (even) { nfirst = nlast; nlast = roots.length; }
      let slo: number, shi: number, sgn: number;
      if (even) {
        slo = ineed < 2 ? -BIG : mid;
        shi = sublo + cd;
        sgn = 1;
      } else {
        slo = subhi - cd;
        shi = ineed < 2 ? BIG : mid;
        sgn = -1;
      }
      const sendlist: number[] = [];
      for (let i = nfirst; i < nlast; i++) {
        const c = xs[3 * i + dim];
        if (c >= slo && c <= shi) sendlist.push(i);
      }
      for (const i of sendlist) {
        const k = roots.length;
        for (let q = 0; q < 3; q++) xs.push(xs[3 * i + q]);
        xs[3 * k + dim] = xs[3 * i + dim] + sgn * len;
        roots.push(roots[i]);
        imgs.push(imgs[3 * i], imgs[3 * i + 1], imgs[3 * i + 2]);
        imgs[3 * k + dim] += sgn;
      }
    }
  }
  return { nall: roots.length, x: Float64Array.from(xs), root: Int32Array.from(roots), img: Int32Array.from(imgs) };
};

/** comm_brick.cpp: "static constexpr double BIG = 1.0e20;" (slab limits that take in every atom). */
const BIG = 1.0e20;

/**
 * The native far-list ranks of one neighbor build. x holds the owned and ghost positions of the build step
 * (engine indices, 3 per atom).
 */
export const nativeFarOrder = (s: SimState, nb: Neighbor, g: Geometry, x: Float64Array): NativeFarOrder => {
  const n = nb.nlocal, NA = nb.nall;
  const order = nativeOrder(s);
  const nativeIdx = new Int32Array(n);
  for (let k = 0; k < n; k++) nativeIdx[order[k]] = k;
  const xn = new Float64Array(3 * n);
  for (let k = 0; k < n; k++) {
    const e = order[k];
    xn[3 * k] = x[3 * e]; xn[3 * k + 1] = x[3 * e + 1]; xn[3 * k + 2] = x[3 * e + 2];
  }
  const cutg = nb.cutghost;
  const tri = g.triclinic;
  const boxlo = g.lo, boxhi = g.hi;
  // box coordinates -> ghosts: orthogonal boxes shift the coordinates, triclinic boxes shift the lamda coordinates
  let gh: ReturnType<typeof borders>;
  let bblo: number[], bbhi: number[];
  if (!tri) {
    gh = borders(xn, n, g, [g.lo[0], g.lo[1], g.lo[2]], [g.hi[0], g.hi[1], g.hi[2]], [g.lx, g.ly, g.lz], [cutg, cutg, cutg]);
    bblo = [boxlo[0], boxlo[1], boxlo[2]];
    bbhi = [boxhi[0], boxhi[1], boxhi[2]];
  } else {
    // domain.cpp x2lamda (h_inv of Domain::set_global_box: h_inv[0] = 1/lx, h_inv[5] = -xy/(lx ly), h_inv[4] =
    // (xy yz - ly xz)/(lx ly lz), h_inv[1] = 1/ly, h_inv[3] = -yz/(ly lz), h_inv[2] = 1/lz)
    const lx = g.lx, ly = g.ly, lz = g.lz, xy = g.xy, xz = g.xz, yz = g.yz;
    const i00 = 1 / lx, i01 = -xy / (lx * ly), i02 = (xy * yz - ly * xz) / (lx * ly * lz);
    const i11 = 1 / ly, i12 = -yz / (ly * lz), i22 = 1 / lz;
    const lam = new Float64Array(3 * n);
    const tmp = [0, 0, 0];
    for (let k = 0; k < n; k++) {
      g.toLamda(xn[3 * k], xn[3 * k + 1], xn[3 * k + 2], tmp);
      lam[3 * k] = tmp[0]; lam[3 * k + 1] = tmp[1]; lam[3 * k + 2] = tmp[2];
    }
    // CommBrick::setup: cutghost in lamda units from the inverse box matrix columns
    const len0 = Math.sqrt(i00 * i00 + i01 * i01 + i02 * i02);
    const len1 = Math.sqrt(i11 * i11 + i12 * i12);
    const len2 = i22;
    gh = borders(lam, n, g, [0, 0, 0], [1, 1, 1], [1, 1, 1], [cutg * len0, cutg * len1, cutg * len2]);
    // lamda2x (domain.cpp) for every ghost
    const xb = new Float64Array(3 * gh.nall);
    for (let k = 0; k < gh.nall; k++) {
      g.fromLamda(gh.x[3 * k], gh.x[3 * k + 1], gh.x[3 * k + 2], tmp);
      xb[3 * k] = tmp[0]; xb[3 * k + 1] = tmp[1]; xb[3 * k + 2] = tmp[2];
    }
    for (let k = 0; k < n; k++) { xb[3 * k] = xn[3 * k]; xb[3 * k + 1] = xn[3 * k + 1]; xb[3 * k + 2] = xn[3 * k + 2]; }
    gh = { ...gh, x: xb };
    // domain.cpp set_global_box: the bounding box of the tilted cell
    let l0 = Math.min(boxlo[0], boxlo[0] + xy);
    l0 = Math.min(l0, l0 + xz);
    const l1 = Math.min(boxlo[1], boxlo[1] + yz);
    let h0 = Math.max(boxhi[0], boxhi[0] + xy);
    h0 = Math.max(h0, h0 + xz);
    const h1 = Math.max(boxhi[1], boxhi[1] + yz);
    bblo = [l0, l1, boxlo[2]];
    bbhi = [h0, h1, boxhi[2]];
  }
  const xyzOwned = gh.x;
  const engByKey = new Map<number, number>();
  for (let k = n; k < NA; k++) engByKey.set(imgKey(nb.owner[k], nb.gimage[3 * k], nb.gimage[3 * k + 1], nb.gimage[3 * k + 2]), k);
  /** engine index of native atom k (owned or ghost), -1 when the engine holds no such image. */
  const engineIndex = (k: number): number => {
    if (k < n) return order[k];
    const hit = engByKey.get(imgKey(order[gh.root[k]], gh.img[3 * k], gh.img[3 * k + 1], gh.img[3 * k + 2]));
    return hit === undefined ? -1 : hit;
  };

  // bins (nbin.cpp coord2bin with bboxlo/bboxhi of the box; nbin_standard.cpp setup_bins)
  const bbox = [bbhi[0] - bblo[0], bbhi[1] - bblo[1], bbhi[2] - bblo[2]];
  let binsizeOpt = nb.binsize > 0 ? nb.binsize : 0.5 * nb.cutneighmax;
  if (binsizeOpt === 0) binsizeOpt = bbox[0];
  const binsizeinv = 1.0 / binsizeOpt;
  const nbin = [0, 0, 0], binsize = [0, 0, 0], bininv = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    nbin[d] = Math.trunc(bbox[d] * binsizeinv);
    if (nbin[d] === 0) nbin[d] = 1;
    binsize[d] = bbox[d] / nbin[d];
    bininv[d] = 1.0 / binsize[d];
  }
  const coord = (v: number, d: number): number => {
    if (v >= bbhi[d]) return Math.trunc((v - bbhi[d]) * bininv[d]) + nbin[d];
    if (v >= bblo[d]) return Math.min(Math.trunc((v - bblo[d]) * bininv[d]), nbin[d] - 1);
    return Math.trunc((v - bblo[d]) * bininv[d]) - 1;
  };
  const nall = gh.nall;
  const binOf = new Int32Array(nall * 3);
  const chains = new Map<number, number[]>();
  for (let k = 0; k < nall; k++) {
    const bx = coord(xyzOwned[3 * k], 0), by = coord(xyzOwned[3 * k + 1], 1), bz = coord(xyzOwned[3 * k + 2], 2);
    binOf[3 * k] = bx; binOf[3 * k + 1] = by; binOf[3 * k + 2] = bz;
    const key = binKey(bx, by, bz);
    let chain = chains.get(key);
    if (!chain) { chain = []; chains.set(key, chain); }
    chain.push(k);
  }

  // full stencil (nstencil_ghost_bin.cpp create): k outer, then j, then i, bin_distance < cutneighmax^2
  const cutmax = nb.cutneighmax;
  const reach = (d: number): number => {
    const r = Math.trunc(cutmax * bininv[d]);
    return r * binsize[d] < cutmax ? r + 1 : r;
  };
  const sx = reach(0), sy = reach(1), sz = reach(2);
  const binDist = (i: number, j: number, k: number): number => {
    const dx = i > 0 ? (i - 1) * binsize[0] : i === 0 ? 0 : (i + 1) * binsize[0];
    const dy = j > 0 ? (j - 1) * binsize[1] : j === 0 ? 0 : (j + 1) * binsize[1];
    const dz = k > 0 ? (k - 1) * binsize[2] : k === 0 ? 0 : (k + 1) * binsize[2];
    return dx * dx + dy * dy + dz * dz;
  };
  const cutmaxsq = cutmax * cutmax;
  const stencil: number[] = [];
  for (let k = -sz; k <= sz; k++) {
    for (let j = -sy; j <= sy; j++) {
      for (let i = -sx; i <= sx; i++) {
        if (binDist(i, j, k) < cutmaxsq) stencil.push(i, j, k);
      }
    }
  }

  const rank = new Map<number, number>();
  for (let k = 0; k < n; k++) {
    const ei = order[k];
    const bx = binOf[3 * k], by = binOf[3 * k + 1], bz = binOf[3 * k + 2];
    let r = 0;
    for (let st = 0; st < stencil.length; st += 3) {
      const chain = chains.get(binKey(bx + stencil[st], by + stencil[st + 1], bz + stencil[st + 2]));
      if (!chain) continue;
      for (const j of chain) {
        if (j <= k) continue;
        const ek = engineIndex(j);
        if (ek < 0) continue;
        rank.set(ei * NA + ek, r++);
      }
    }
  }
  return { nativeIdx, rank, nall: NA };
};
