/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/nbin_standard.cpp (NBinStandard::setup_bins), src/nbin.cpp (NBin::coord2bin), src/nstencil.cpp
 * (NStencil::create_setup and bin_distance), src/nstencil_bin.cpp (NStencilBin::create), src/npair_bin.cpp and
 * src/npair_nsq.cpp (which atom of a pair a half list stores it at), src/my_page.h (MyPage::vgot, the one-atom limit)
 * and src/update.cpp (force->angstrom of each unit style) at commit b332f28.
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License (the LAMMPS file headers). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The standard bins and stencils of a neighbor list build (neighbor style bin), shared by dump image autobond
 * (../graphics/autobond.ts) and the neigh_modify one check (nativeMaxCount, the lazily loaded helper neigh_one).
 */
import type { NeighOneInput } from '../../engine/neighbor';

/** force->angstrom of each unit style (Update::set_units). */
export const ANGSTROM: Record<string, number> = { lj: 1.0, real: 1.0, metal: 1.0, si: 1.0e-10, cgs: 1.0e-8, electron: 1.88972612, micro: 1.0e-4, nano: 1.0e-1 };

type Vec3 = readonly [number, number, number] | readonly number[];

/**
 * The bins of a list build (NBinStandard::setup_bins) for a box (the bounding box of a triclinic cell), and the stencil
 * range and bin distance (NStencil::create_setup, bin_distance). Bin numbers are global: 0 is the first bin of the box.
 */
export class NativeBins {
  private readonly lo: Vec3;
  private readonly hi: Vec3;
  private readonly nbin: number[];
  private readonly size: number[];
  private readonly inv: number[];
  /** sx, sy, sz: the stencil range in bins */
  readonly s: number[];
  private readonly cutsq: number;

  constructor(lo: Vec3, hi: Vec3, binsizeUser: number, cutneighmax: number, dimension: number) {
    this.lo = lo;
    this.hi = hi;
    const bbox = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let binsizeOptimal = binsizeUser > 0 ? binsizeUser : 0.5 * cutneighmax;
    if (binsizeOptimal === 0.0) binsizeOptimal = bbox[0];
    const binsizeinv = 1.0 / binsizeOptimal;
    this.nbin = [Math.trunc(bbox[0] * binsizeinv), Math.trunc(bbox[1] * binsizeinv), dimension === 3 ? Math.trunc(bbox[2] * binsizeinv) : 1];
    for (let d = 0; d < 3; d++) if (this.nbin[d] === 0) this.nbin[d] = 1;
    this.size = [bbox[0] / this.nbin[0], bbox[1] / this.nbin[1], bbox[2] / this.nbin[2]];
    this.inv = [1.0 / this.size[0], 1.0 / this.size[1], 1.0 / this.size[2]];
    this.s = [0, 1, 2].map((d) => {
      let sd = Math.trunc(cutneighmax * this.inv[d]);
      if (sd * this.size[d] < cutneighmax) sd++;
      return sd;
    });
    if (dimension === 2) this.s[2] = 0;
    this.cutsq = cutneighmax * cutneighmax;
  }

  /** NBin::coord2bin along dimension d. */
  coord(v: number, d: number): number {
    if (v >= this.hi[d]) return Math.trunc((v - this.hi[d]) * this.inv[d]) + this.nbin[d];
    if (v >= this.lo[d]) return Math.min(Math.trunc((v - this.lo[d]) * this.inv[d]), this.nbin[d] - 1);
    return Math.trunc((v - this.lo[d]) * this.inv[d]) - 1;
  }

  /** The bin of atom k of a position array. */
  of(x: Float64Array, k: number): [number, number, number] {
    return [this.coord(x[3 * k], 0), this.coord(x[3 * k + 1], 1), this.coord(x[3 * k + 2], 2)];
  }

  /** NStencil::bin_distance: the closest distance (squared) between the central bin and bin (i, j, k). */
  private distance(i: number, j: number, k: number): number {
    const delx = i > 0 ? (i - 1) * this.size[0] : i === 0 ? 0.0 : (i + 1) * this.size[0];
    const dely = j > 0 ? (j - 1) * this.size[1] : j === 0 ? 0.0 : (j + 1) * this.size[1];
    const delz = k > 0 ? (k - 1) * this.size[2] : k === 0 ? 0.0 : (k + 1) * this.size[2];
    return delx * delx + dely * dely + delz * delz;
  }

  /** NStencilBin::create, full stencil: every bin of the range within the cutoff (the own bin among them). */
  inFull(i: number, j: number, k: number): boolean {
    if (Math.abs(i) > this.s[0] || Math.abs(j) > this.s[1] || Math.abs(k) > this.s[2]) return false;
    return this.distance(i, j, k) < this.cutsq;
  }

  /**
   * NStencilBin::create, half stencil of an orthogonal box: "only include "upper right" bins" (k >= 0, and in the
   * plane k = 0 the bins with j > 0, or j = 0 and i > 0) within the cutoff. The own bin is apart (npair_bin.cpp).
   */
  inHalf(i: number, j: number, k: number): boolean {
    if (k < 0 || (k <= 0 && j <= 0 && (j !== 0 || i <= 0))) return false;
    return this.inFull(i, j, k);
  }
}

/** Whether position (xg, yg, zg) lies "above and to the right" of (xi, yi, zi): the same-bin ghost test of npair_bin.cpp. */
export const aboveXyz = (xi: number, yi: number, zi: number, xg: number, yg: number, zg: number): boolean => {
  if (zg < zi) return false;
  if (zg === zi) {
    if (yg < yi) return false;
    if (yg === yi && xg < xi) return false;
  }
  return true;
};

/**
 * The largest per-atom entry count of a list as native builds it: the count MyPage::vgot compares with neigh_modify
 * one. The engine's list holds the same pairs; a full list holds each at both atoms (a pair outside the stencil range
 * is not stored, as native's stencil does not reach it), a half list at the atom native's build gives it to. Both atoms
 * of each pair are tested, the owned/ghost pair (a, image g of o) at a with g and at o with the image of a on the other
 * side (as native's mirror ghost):
 * - neighbor bin, orthogonal box, newton on (npair_bin.cpp): a neighbor in another bin is stored when its bin is in the
 *   half stencil; in the own bin an owned neighbor later in the storage order, a ghost when it lies above;
 * - neighbor bin, triclinic box, newton on: full stencil, an owned neighbor later in the storage order, a ghost by the
 *   parity of the two IDs ("if ((itag + jtag) % 2 == 0) continue" for itag > jtag, "== 1" for itag < jtag) and for
 *   equal IDs by the coordinates with tolerance delta = 0.01 force->angstrom;
 * - newton off: an owned neighbor later in the storage order, every ghost (the pair is stored on both sides);
 * - neighbor nsq (npair_nsq.cpp): the bin rules without bins, a ghost by the ID parity (equal IDs: above, or the
 *   triclinic tolerance test).
 */
export const nativeMaxCount = (L: NeighOneInput): number => {
  const { nlocal, list, x, owner, tag, rank } = L;
  const count = new Int32Array(nlocal);
  const bins = L.nsq ? null : new NativeBins(L.boxLo, L.boxHi, L.binsize, L.cutneighmax, L.dimension);
  const nall = owner.length;
  let ball: Int32Array | null = null;
  if (bins) {
    ball = new Int32Array(3 * nall);
    for (let k = 0; k < nall; k++) {
      ball[3 * k] = bins.coord(x[3 * k], 0);
      ball[3 * k + 1] = bins.coord(x[3 * k + 1], 1);
      ball[3 * k + 2] = bins.coord(x[3 * k + 2], 2);
    }
  }
  const nb = list.neighbors, mask = L.mask;
  if (L.full) {
    for (let p = 0; p < nlocal; p++) {
      const k1 = list.firstneigh[p] + list.numneigh[p];
      for (let k = list.firstneigh[p]; k < k1; k++) {
        const q = nb[k] & mask;
        if (!ball || bins!.inFull(ball[3 * q] - ball[3 * p], ball[3 * q + 1] - ball[3 * p + 1], ball[3 * q + 2] - ball[3 * p + 2])) count[p]++;
      }
    }
  } else {
    const delta = 0.01 * (ANGSTROM[L.units] ?? 1.0);
    const tri = L.triclinic, newton = L.newton;
    // the ghost test of a triclinic list (and of nsq for equal IDs): ID parity, then the coordinates
    const tagKeep = (itag: number, jtag: number, p: number, xq: number, yq: number, zq: number): boolean => {
      if (itag > jtag) return (itag + jtag) % 2 !== 0;
      if (itag < jtag) return (itag + jtag) % 2 !== 1;
      const xp = x[3 * p], yp = x[3 * p + 1], zp = x[3 * p + 2];
      if (!tri) return aboveXyz(xp, yp, zp, xq, yq, zq);
      if (Math.abs(zq - zp) > delta) return !(zq < zp);
      if (Math.abs(yq - yp) > delta) return !(yq < yp);
      return !(xq < xp);
    };
    // whether owned atom p's list holds neighbor q (owned: its index; ghost: -1 with the ID and position given)
    const stored = (p: number, q: number, qtag: number, xq: number, yq: number, zq: number): boolean => {
      if (bins) {
        const oi = bins.coord(xq, 0) - ball![3 * p], oj = bins.coord(yq, 1) - ball![3 * p + 1], ok = bins.coord(zq, 2) - ball![3 * p + 2];
        if (newton && !tri) {
          if (oi === 0 && oj === 0 && ok === 0) return q >= 0 ? rank[q] > rank[p] : aboveXyz(x[3 * p], x[3 * p + 1], x[3 * p + 2], xq, yq, zq);
          return bins.inHalf(oi, oj, ok);
        }
        if (!bins.inFull(oi, oj, ok)) return false;
      }
      if (q >= 0) return rank[q] > rank[p];
      return newton ? tagKeep(tag[p], qtag, p, xq, yq, zq) : true;
    };
    for (let a = 0; a < nlocal; a++) {
      const k1 = list.firstneigh[a] + list.numneigh[a];
      const xa = x[3 * a], ya = x[3 * a + 1], za = x[3 * a + 2];
      for (let k = list.firstneigh[a]; k < k1; k++) {
        const j = nb[k] & mask;
        const xj = x[3 * j], yj = x[3 * j + 1], zj = x[3 * j + 2];
        if (j < nlocal) {
          if (stored(a, j, tag[j], xj, yj, zj)) count[a]++;
          if (stored(j, a, tag[a], xa, ya, za)) count[j]++;
        } else {
          const o = owner[j];
          if (stored(a, -1, tag[o], xj, yj, zj)) count[a]++;
          // the mirror pair: o with the image of a on the other side
          const xo = x[3 * o], yo = x[3 * o + 1], zo = x[3 * o + 2];
          if (stored(o, -1, tag[a], xa - (xj - xo), ya - (yj - yo), za - (zj - zo))) count[o]++;
        }
      }
    }
  }
  let max = 0;
  for (let p = 0; p < nlocal; p++) if (count[p] > max) max = count[p];
  return max;
};
