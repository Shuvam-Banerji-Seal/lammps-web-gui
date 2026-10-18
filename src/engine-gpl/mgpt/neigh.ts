/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source, at commit b332f28: src/nbin_standard.cpp (NBinStandard::setup_bins, the bins that cover the ghost atoms, and
 * bin_atoms, the bin chains), src/npair.cpp (NPair::coord2bin with the bin coordinates), src/nstencil.cpp (the stencil
 * reach sx, sy, sz and bin_distance), src/nstencil_ghost_bin.cpp (NStencilGhostBin::create, the full stencil with its
 * bin offsets) and src/npair_bin_ghost.cpp (NPairBinGhost<0>::build, the full neighbor list of owned and ghost atoms
 * that pair_style mgpt requests with NeighConst::REQ_FULL | NeighConst::REQ_GHOST), and src/comm_brick.cpp
 * (CommBrick::forward_comm: the ghost coordinates of each step). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Original authors: the LAMMPS development team (Sandia National Laboratories), as named in those files.
 *
 * One processor, orthogonal box, atom style without special neighbors: native's local atom order, its ghost atoms
 * (../neighbor/nativeOrder.ts nativeGhosts) and its full neighbor list with the neighbors of ghost atoms, in native's
 * order. The list is built at each reneighboring; the coordinates follow the atoms at every step.
 */
import type { Neighbor } from '../../engine/neighbor';
import { nativeOrder } from '../../engine/atoms';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { nativeGhosts } from '../neighbor/nativeOrder';

/** nbin_standard.cpp: "static constexpr double SMALL = 1.0e-6;" */
const SMALL = 1.0e-6;

/** The native atoms of one neighbor build and their full ghost neighbor list (CSR). */
export interface MgptLayout {
  /** native local atom k is engine atom order[k] */
  order: Int32Array;
  n: number;
  nall: number;
  /** ghost k (k >= n): the native atom its coordinates are sent from, and the shift of that swap (dimension, sign) */
  from: Int32Array;
  dim: Int8Array;
  sgn: Int8Array;
  /** periodic image of each native atom relative to its owner, and the owner's engine index */
  img: Int32Array;
  root: Int32Array;
  /** neighbors of native atom i: list[first[i] .. first[i+1]) */
  first: Int32Array;
  list: Int32Array;
  /** positions of the native atoms (3 per atom), refreshed each step by positions() */
  X: Float64Array;
  nbuild: number;
  stamp: Int32Array | null;
}

/**
 * Native atoms, ghosts and the full/bin/ghost neighbor list of one build. cutoff is the pair cutoff of every type pair
 * (cutneighsq = cutneighghostsq = (cutoff + skin)^2 for pair mgpt).
 */
export const buildLayout = (s: SimState, nb: Neighbor, g: Geometry, cutoff: number): MgptLayout => {
  const order = nativeOrder(s);
  const n = s.n;
  const two = s.dimension === 2;
  const xn = new Float64Array(3 * n);
  for (let k = 0; k < n; k++) {
    const e = order[k];
    xn[3 * k] = s.x[3 * e]; xn[3 * k + 1] = s.x[3 * e + 1]; xn[3 * k + 2] = s.x[3 * e + 2];
  }
  const cutghost = nb.cutghost;
  const gh = nativeGhosts(xn, n, g, cutghost, two);
  const nall = gh.nall;
  const X = gh.x;
  const dim = new Int8Array(nall), sgn = new Int8Array(nall);
  for (let k = n; k < nall; k++) {
    const f = gh.from[k];
    for (let d = 0; d < 3; d++) {
      const di = gh.img[3 * k + d] - gh.img[3 * f + d];
      if (di !== 0) { dim[k] = d; sgn[k] = di; }
    }
  }

  // nbin_standard.cpp setup_bins (bboxlo/hi = boxlo/hi of an orthogonal box; bsubbox = subdomain +- comm->cutghost)
  const lo = g.lo, hi = g.hi;
  const bbox = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const cutneighmax = nb.cutneighmax;
  let binsizeOptimal = nb.binsize > 0 ? nb.binsize : 0.5 * cutneighmax;
  if (binsizeOptimal === 0.0) binsizeOptimal = bbox[0];
  const binsizeinv = 1.0 / binsizeOptimal;
  const nbin = [0, 0, 0], binsize = [0, 0, 0], bininv = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    nbin[d] = d === 2 && two ? 1 : Math.trunc(bbox[d] * binsizeinv);
    if (nbin[d] === 0) nbin[d] = 1;
    binsize[d] = bbox[d] / nbin[d];
    bininv[d] = 1.0 / binsize[d];
  }
  const mlo = [0, 0, 0], mn = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    if (d === 2 && two) { mlo[d] = 0; mn[d] = 1; continue; }
    let coord = lo[d] - cutghost - SMALL * bbox[d];
    let mbinlo = Math.trunc((coord - lo[d]) * bininv[d]);
    if (coord < lo[d]) mbinlo = mbinlo - 1;
    coord = hi[d] + cutghost + SMALL * bbox[d];
    const mbinhi = Math.trunc((coord - lo[d]) * bininv[d]);
    // extend bins by 1 to ensure stencil extent is included
    mlo[d] = mbinlo - 1;
    mn[d] = mbinhi + 1 - mlo[d] + 1;
  }
  const [mbinx, mbiny, mbinz] = mn;
  const mbins = mbinx * mbiny * mbinz;

  // npair.cpp coord2bin(x, ix, iy, iz): bin coordinates relative to mbinlo
  const bc = [0, 0, 0];
  const coord2bin = (k: number): number => {
    for (let d = 0; d < 3; d++) {
      const x = X[3 * k + d];
      let c: number;
      if (x >= hi[d]) c = Math.trunc((x - hi[d]) * bininv[d]) + nbin[d];
      else if (x >= lo[d]) c = Math.min(Math.trunc((x - lo[d]) * bininv[d]), nbin[d] - 1);
      else c = Math.trunc((x - lo[d]) * bininv[d]) - 1;
      bc[d] = c - mlo[d];
    }
    return bc[2] * mbiny * mbinx + bc[1] * mbinx + bc[0];
  };

  // nbin_standard.cpp bin_atoms: "bin in reverse order so linked list will be in forward order"
  const binhead = new Int32Array(mbins).fill(-1);
  const bins = new Int32Array(nall);
  const atom2bin = new Int32Array(nall);
  for (let i = nall - 1; i >= 0; i--) {
    const ibin = coord2bin(i);
    atom2bin[i] = ibin;
    bins[i] = binhead[ibin];
    binhead[ibin] = i;
  }

  // nstencil.cpp sx, sy, sz and nstencil_ghost_bin.cpp create
  const reach = (d: number): number => {
    const r = Math.trunc(cutneighmax * bininv[d]);
    return r * binsize[d] < cutneighmax ? r + 1 : r;
  };
  const sx = reach(0), sy = reach(1), sz = two ? 0 : reach(2);
  const binDist = (i: number, j: number, k: number): number => {
    const dx = i > 0 ? (i - 1) * binsize[0] : i === 0 ? 0.0 : (i + 1) * binsize[0];
    const dy = j > 0 ? (j - 1) * binsize[1] : j === 0 ? 0.0 : (j + 1) * binsize[1];
    const dz = k > 0 ? (k - 1) * binsize[2] : k === 0 ? 0.0 : (k + 1) * binsize[2];
    return dx * dx + dy * dy + dz * dz;
  };
  const cutneighmaxsq = cutneighmax * cutneighmax;
  const sxyz: number[] = [];
  const stencil: number[] = [];
  for (let k = -sz; k <= sz; k++) {
    for (let j = -sy; j <= sy; j++) {
      for (let i = -sx; i <= sx; i++) {
        if (binDist(i, j, k) < cutneighmaxsq) {
          sxyz.push(i, j, k);
          stencil.push(k * mbiny * mbinx + j * mbinx + i);
        }
      }
    }
  }
  const nstencil = stencil.length;

  // npair_bin_ghost.cpp build, full list (HALF = 0): owned atoms take every stencil bin, ghost atoms the bins inside
  // the binned region; cutneighsq = cutneighghostsq = (cutoff + skin)^2
  const cut = Math.sqrt(cutoff * cutoff) + nb.skin;
  const cutsq = cut * cut;
  const first = new Int32Array(nall + 1);
  let list = new Int32Array(Math.max(1024, nall * 64));
  let m = 0;
  const push = (j: number): void => {
    if (m === list.length) {
      const grown = new Int32Array(2 * list.length);
      grown.set(list);
      list = grown;
    }
    list[m++] = j;
  };
  for (let i = 0; i < nall; i++) {
    first[i] = m;
    const xtmp = X[3 * i], ytmp = X[3 * i + 1], ztmp = X[3 * i + 2];
    if (i < n) {
      const ibin = atom2bin[i];
      for (let k = 0; k < nstencil; k++) {
        const b = ibin + stencil[k];
        if (b < 0 || b >= mbins) continue;
        for (let j = binhead[b]; j >= 0; j = bins[j]) {
          if (i === j) continue;
          const delx = xtmp - X[3 * j], dely = ytmp - X[3 * j + 1], delz = ztmp - X[3 * j + 2];
          const rsq = delx * delx + dely * dely + delz * delz;
          if (rsq <= cutsq) push(j);
        }
      }
    } else {
      const ibin = coord2bin(i);
      const xbin = bc[0], ybin = bc[1], zbin = bc[2];
      for (let k = 0; k < nstencil; k++) {
        const xbin2 = xbin + sxyz[3 * k], ybin2 = ybin + sxyz[3 * k + 1], zbin2 = zbin + sxyz[3 * k + 2];
        if (xbin2 < 0 || xbin2 >= mbinx || ybin2 < 0 || ybin2 >= mbiny || zbin2 < 0 || zbin2 >= mbinz) continue;
        for (let j = binhead[ibin + stencil[k]]; j >= 0; j = bins[j]) {
          if (i === j) continue;
          const delx = xtmp - X[3 * j], dely = ytmp - X[3 * j + 1], delz = ztmp - X[3 * j + 2];
          const rsq = delx * delx + dely * dely + delz * delz;
          if (rsq <= cutsq) push(j);
        }
      }
    }
  }
  first[nall] = m;
  return {
    order, n, nall, from: gh.from, dim, sgn, img: gh.img, root: gh.root,
    first, list: list.slice(0, m), X, nbuild: nb.nbuild, stamp: s.order ?? null,
  };
};

/**
 * The native positions of this step: owned atoms from the engine, ghosts as CommBrick::forward_comm sets them (the
 * coordinates of the atom each ghost was sent from, plus the swap's periodic shift pbc * prd in its dimension, with
 * the box lengths of this step).
 */
export const positions = (L: MgptLayout, s: SimState, g: Geometry): void => {
  const { order, n, nall, from, dim, sgn, X } = L;
  const prd = [g.lx, g.ly, g.lz];
  for (let k = 0; k < n; k++) {
    const e = order[k];
    X[3 * k] = s.x[3 * e]; X[3 * k + 1] = s.x[3 * e + 1]; X[3 * k + 2] = s.x[3 * e + 2];
  }
  for (let k = n; k < nall; k++) {
    const f = from[k], d = dim[k];
    X[3 * k] = X[3 * f]; X[3 * k + 1] = X[3 * f + 1]; X[3 * k + 2] = X[3 * f + 2];
    X[3 * k + d] = X[3 * f + d] + sgn[k] * prd[d];
  }
};
