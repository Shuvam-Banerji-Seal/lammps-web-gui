/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/VORONOI/compute_voronoi_atom.cpp (ComputeVoronoi::buildCells, loopCells and processCell: the container
 * bounds with the EPS margin and the block counts, the put order of local then ghost atoms, the owner loop and the
 * neighbors rows with the fthresh face filter), src/comm_brick.cpp (CommBrick::setup and borders: the ghost swaps of
 * a single periodic processor, one swap per side while cutghost < prd, the slab tests and the pbc shifts of
 * AtomVec::pack_border in src/atom_vec.cpp), and the voro++ 0.4.6 port in this directory (cell.ts, container.ts).
 *
 * Coverage: orthogonal boxes, 2d and 3d; periodic dimensions with one ghost swap per side (cutghost < box length);
 * non-periodic dimensions (no ghosts, the box bounds clip the cells). Other layouts (triclinic, or cutghost at or
 * beyond a box length, which needs several swaps per side) return null, and the caller keeps its set-based rows.
 *
 * voro++ is Copyright (c) 2008 The Regents of the University of California (LBL) and Copyright (c) 2008-2011 Chris
 * H. Rycroft, BSD three-clause licence (LICENSE.voro++ in this directory). LAMMPS is Copyright (2003) Sandia
 * Corporation, distributed under the GNU General Public License version 2 (src/engine-gpl/LICENSE).
 */
import { VoroCellNeighbor } from './cell';
import { VoroContainer } from './container';

/** BIG of comm_brick.cpp (slab bounds of the outermost swaps). */
const BIG = 1.0e20;
/** The EPS margin of ComputeVoronoi::buildCells. */
const EPS = 0.01;

export interface NativeNeighborInput {
  /** 2 or 3 (compute voronoi/atom: 2d boxes have no z ghosts, and the z bounds are the box) */
  dim: number;
  /** orthogonal box bounds (domain->boxlo, boxhi; the single processor's sub-domain) */
  lo: readonly number[];
  hi: readonly number[];
  periodic: readonly boolean[];
  /** comm->cutghost (neighbor cutneighmax, or comm_modify cutoff) */
  cutghost: number;
  /** local atom positions, wrapped into the box, stride 3, in LAMMPS local order */
  x: Float64Array;
  /** 1 for atoms in the compute group (owners); the container holds the atoms with pool 1 (only_group) */
  owner: Uint8Array;
  pool: Uint8Array;
  /** faces with area at or below this are not reported (compute voronoi/atom face_threshold, default 0) */
  faceThreshold: number;
}

/** One neighbors row: owner and neighbour local indices (-1 for an exterior face), and the face area. */
export interface NativeNeighborRow {
  owner: number;
  neighbor: number;
  area: number;
}

/** Ghost positions and their source local atoms, in the order CommBrick::borders creates them. */
const buildGhosts = (inp: NativeNeighborInput, prd: readonly number[]): { x: number[]; src: number[] } | null => {
  const nlocal = inp.x.length / 3;
  const X: number[] = [];
  const src: number[] = [];
  for (let i = 0; i < nlocal; i++) {
    X.push(inp.x[3 * i], inp.x[3 * i + 1], inp.x[3 * i + 2]);
    src.push(i);
  }
  for (let dim = 0; dim < 3; dim++) {
    // single processor: maxneed = min(int(cutghost/prd)+1, procgrid-1) in a periodic dimension, 0 otherwise
    if (dim === 2 && inp.dim === 2) continue;
    // non-periodic: the single processor has no neighbour to send to (maxneed = min(..., procgrid-1 = 0))
    if (!inp.periodic[dim]) continue;
    const maxneed = Math.trunc(inp.cutghost / prd[dim]) + 1;
    if (maxneed !== 1) return null;
    let nfirst = 0;
    let nlast = 0;
    for (let ineed = 0; ineed < 2; ineed++) {
      if (ineed % 2 === 0) {
        nfirst = nlast;
        nlast = X.length / 3;
      }
      // slab bounds of the swap (comm_brick.cpp setup, ineed < 2): send low atoms (+prd) or high atoms (-prd)
      const slabLo = ineed === 0 ? -BIG : inp.hi[dim] - inp.cutghost;
      const slabHi = ineed === 0 ? inp.lo[dim] + inp.cutghost : BIG;
      const shift = ineed === 0 ? prd[dim] : -prd[dim];
      // the candidates are fixed before the swap appends its ghosts (nfirst..nlast, as borders() does)
      const sel: number[] = [];
      for (let i = nfirst; i < nlast; i++) {
        const v = X[3 * i + dim];
        if (v >= slabLo && v <= slabHi) sel.push(i);
      }
      for (const i of sel) {
        const sx = X[3 * i], sy = X[3 * i + 1], sz = X[3 * i + 2];
        // AtomVec::pack_border: x + pbc*prd in the shifted dimension (pbc 0 adds 0 in the others)
        X.push(dim === 0 ? sx + shift : sx, dim === 1 ? sy + shift : sy, dim === 2 ? sz + shift : sz);
        src.push(src[i]);
      }
    }
  }
  return { x: X, src };
};

/**
 * The neighbors rows of compute voronoi/atom in native order, or null when the layout is not reproduced here.
 * Rows are emitted owner by owner in the container's block order, and per owner in voro++'s face order (the
 * neighbour and area of each face), which the set-based code of core cannot reproduce.
 */
export const nativeNeighborRows = (inp: NativeNeighborInput): NativeNeighborRow[] | null => {
  const prd = [inp.hi[0] - inp.lo[0], inp.hi[1] - inp.lo[1], inp.hi[2] - inp.lo[2]];
  const ghosts = buildGhosts(inp, prd);
  if (ghosts === null) return null;
  const nall = ghosts.x.length / 3;
  const nlocal = inp.x.length / 3;
  // ComputeVoronoi::buildCells: sublo/subhi bounds with the cutoff and EPS, clipped at non-periodic box faces
  const sb = [0, 0, 0], sh = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    sb[i] = inp.lo[i] - inp.cutghost - EPS;
    sh[i] = inp.hi[i] + inp.cutghost + EPS;
    if (!inp.periodic[i]) {
      sb[i] = Math.max(sb[i], inp.lo[i]);
      sh[i] = Math.min(sh[i], inp.hi[i]);
    }
  }
  if (inp.dim === 2) {
    sb[2] = inp.lo[2];
    sh[2] = inp.hi[2];
  }
  const n = [0, 0, 0];
  for (let i = 0; i < 3; i++) n[i] = sh[i] - sb[i];
  const V = n[0] * n[1] * n[2];
  for (let i = 0; i < 3; i++) {
    n[i] = Math.round(n[i] * Math.pow(nall / (V * 8.0), 0.333333));
    n[i] = n[i] === 0 ? 1 : n[i];
  }
  const con = new VoroContainer(sb[0], sh[0], sb[1], sh[1], sb[2], sh[2], n[0], n[1], n[2]);
  for (let i = 0; i < nall; i++) {
    const s = ghosts.src[i];
    if (inp.pool[s]) con.put(i, ghosts.x[3 * i], ghosts.x[3 * i + 1], ghosts.x[3 * i + 2]);
  }
  const rows: NativeNeighborRow[] = [];
  const cell = new VoroCellNeighbor();
  // c_loop_all: blocks with i fastest, then j, then k; the particles of a block in put order
  for (let k = 0; k < n[2]; k++) {
    for (let j = 0; j < n[1]; j++) {
      for (let i = 0; i < n[0]; i++) {
        const ijk = i + n[0] * (j + n[1] * k);
        const cnt = con.co[ijk];
        for (let q = 0; q < cnt; q++) {
          const pid = con.blocks[ijk].id[q];
          if (!con.computeCell(cell, ijk, q, i, j, k)) continue;
          if (pid >= nlocal || !inp.owner[pid]) continue;
          const nbs = cell.neighbors();
          const areas = cell.faceAreas();
          for (let f = 0; f < nbs.length; f++) {
            if (!(areas[f] > inp.faceThreshold)) continue;
            const nb = nbs[f] >= 0 ? ghosts.src[nbs[f]] : -1;
            rows.push({ owner: pid, neighbor: nb, area: areas[f] });
          }
        }
      }
    }
  }
  return rows;
};
