/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/comm_brick.cpp at commit b332f28 (CommBrick::borders, through ./nativeOrder.ts nativeGhosts, for the ghost
 * atoms of one processor, and CommBrick::forward_comm / AtomVec::pack_comm for their coordinates at every step: the
 * coordinates of the atom each ghost was sent from plus the swap's periodic shift pbc * prd). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Original authors: the LAMMPS development team (Sandia National Laboratories).
 *
 * The owned and ghost atoms in native order for a style whose result depends on that order (sums over owned and ghost
 * atoms): the ghost set is native's set of the last reneighboring (kept until the next one, as native keeps its borders),
 * the coordinates are those of the current step. One processor, orthogonal box.
 */
import type { Neighbor } from '../../engine/neighbor';
import type { Geometry } from '../../engine/domain';
import type { SimState } from '../../engine/types';
import { nativeOrder } from '../../engine/atoms';
import { nativeGhosts } from './nativeOrder';

export class NativeGhostFrame {
  private nbuild = -1;
  private stamp: Int32Array | null = null;
  /** native local atom k is engine atom order[k] */
  order: Int32Array = new Int32Array(0);
  n = 0;
  nall = 0;
  /** native atom k (owned or ghost): the engine index of its owner */
  owner = new Int32Array(0);
  /** ghost k (k >= n): the native atom its coordinates are sent from, and the dimension and sign of that swap's shift */
  from: Int32Array = new Int32Array(0);
  private dim = new Int8Array(0);
  private sgn = new Int8Array(0);
  /** coordinates of the native atoms, 3 per atom */
  X = new Float64Array(0);

  /** Brings the frame to the current step: rebuilds the ghosts after a reneighboring, else updates their coordinates. */
  update(s: SimState, nb: Neighbor, g: Geometry): void {
    if (this.nbuild !== nb.nbuild || this.stamp !== (s.order ?? null) || this.n !== s.n) this.rebuild(s, nb, g);
    const { order, n, nall, from, dim, sgn, X } = this;
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
  }

  /** Copies a per-atom value (one per engine owned atom) to every native atom, as forward_comm sends it to the ghosts. */
  spread(values: ArrayLike<number>, out: Float64Array): void {
    for (let k = 0; k < this.nall; k++) out[k] = values[this.owner[k]];
  }

  private rebuild(s: SimState, nb: Neighbor, g: Geometry): void {
    const order = nativeOrder(s);
    const n = s.n;
    const xn = new Float64Array(3 * n);
    for (let k = 0; k < n; k++) {
      const e = order[k];
      xn[3 * k] = s.x[3 * e]; xn[3 * k + 1] = s.x[3 * e + 1]; xn[3 * k + 2] = s.x[3 * e + 2];
    }
    const gh = nativeGhosts(xn, n, g, nb.cutghost, s.dimension === 2);
    const nall = gh.nall;
    const dim = new Int8Array(nall), sgn = new Int8Array(nall), owner = new Int32Array(nall);
    for (let k = 0; k < nall; k++) {
      owner[k] = order[gh.root[k]];
      if (k < n) continue;
      const f = gh.from[k];
      for (let d = 0; d < 3; d++) {
        const di = gh.img[3 * k + d] - gh.img[3 * f + d];
        if (di !== 0) { dim[k] = d; sgn[k] = di; }
      }
    }
    this.order = order;
    this.n = n;
    this.nall = nall;
    this.owner = owner;
    this.from = gh.from;
    this.dim = dim;
    this.sgn = sgn;
    this.X = new Float64Array(3 * nall);
    this.nbuild = nb.nbuild;
    this.stamp = s.order ?? null;
  }
}
