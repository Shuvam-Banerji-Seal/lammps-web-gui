import type { SimState } from './types';
import type { Geometry } from './domain';
import type { SpecialList } from './atoms';

/*
 * Ghost atoms and Verlet neighbor lists.
 *
 * docs.lammps.org/neighbor.html: "All atom pairs within a neighbor cutoff
 * distance equal to the their force cutoff plus the skin distance are stored
 * in the list." docs.lammps.org/neigh_modify.html: "The delay setting means
 * never build new lists until at least N steps after the previous build. The
 * every setting means attempt to build lists every M steps (after the delay
 * has passed). If the check setting is no, the lists are built on the first
 * step that satisfies the delay and every settings. If the check setting is
 * yes, then the every and delay settings determine when a build may possibly
 * be performed, but an actual build only occurs if at least one atom has
 * moved more than half the neighbor skin distance ... since the last
 * neighbor list build." Defaults: "delay = 0, every = 1, check = yes".
 *
 * Periodic boundaries are handled with ghost atoms, as Developer_flow.html
 * describes ("borders() ... identifies ghost atoms surrounding each
 * processor's subdomain"; "If the newton flag is on, forces on ghost atoms
 * are communicated and summed back to their corresponding owned atoms"):
 * every periodic image of an owned atom that lies within the ghost cutoff of
 * the box becomes a ghost with an owner index and integer image offsets, so
 * boxes smaller than the cutoff, triclinic boxes and many-body potentials
 * need no special cases. Ghost positions are refreshed from their owners
 * every step (forward communication); forces on ghosts are summed back into
 * the owners (reverse communication).
 *
 * Half lists hold each pair once: two owned atoms when j > i; an owned atom
 * and a ghost when the ghost lies "above" (larger z, then y, then x) — the
 * mirror pair (the other atom with the opposite image) lies below, so it is
 * skipped. Full lists hold every neighbor of every owned atom.
 *
 * special_bonds (docs.lammps.org/special_bonds.html): "A value of 1.0 means
 * include the full interaction without flagging the pair as a "special
 * pair"; a value of 0.0 means exclude the pair completely from the neighbor
 * list, except for pair styles that require a kspace style". Flagged pairs
 * carry their order (1 = 1-2, 2 = 1-3, 3 = 1-4) in the top bits of j.
 */

export const SBBITS = 30;
export const NEIGHMASK = (1 << SBBITS) - 1;
export const sbOf = (j: number): number => j >>> SBBITS;

export interface NeighList {
  /** Number of owned atoms with lists. */
  inum: number;
  numneigh: Int32Array;
  firstneigh: Int32Array;
  /** Encoded neighbors: index into the owned+ghost arrays | (order << SBBITS). */
  neighbors: Int32Array;
}

export interface SpecialSettings {
  lj: [number, number, number, number];
  coul: [number, number, number, number];
  /** Keep 0/0 pairs (flagged) because a kspace style subtracts them. */
  keepExcluded: boolean;
}

export interface NeighborNeeds {
  half: boolean;
  full: boolean;
  /** Per type pair (ntypes+1)^2, the force cutoff (0 = no interaction). */
  cutoff: Float64Array;
  ntypes: number;
  special: SpecialList | null;
  specialSettings: SpecialSettings;
  /** Optional pair exclusion (neigh_modify exclude), by owned indices. */
  exclude?: ((i: number, j: number) => boolean) | null;
}

export class Neighbor {
  skin: number;
  every = 1;
  delay = 0;
  check = true;
  once = false;
  /** neigh_modify binsize (0 = half the largest neighbor cutoff). */
  binsize = 0;
  /** comm_modify cutoff: ghost cutoff at least this large. */
  commCutoff = 0;
  /**
   * neigh_modify exclude (neigh_modify.html): "type M N = exclude if one atom
   * in pair is type M, other is type N"; "group group1-ID group2-ID = exclude
   * if one atom is in 1st group, other in 2nd"; "molecule/intra group-ID =
   * exclude if both atoms are in the same molecule and in group";
   * "molecule/inter group-ID = exclude if both atoms are in different
   * molecules and in group".
   */
  excludes: ({ kind: 'type'; a: number; b: number } | { kind: 'group'; a: number; b: number }
    | { kind: 'molecule/intra' | 'molecule/inter'; a: number })[] = [];
  /** neigh_modify include group-ID: only atoms of this group get pair neighbors (bit, 0 = all). */
  includeBit = 0;

  nlocal = 0;
  nghost = 0;
  get nall(): number { return this.nlocal + this.nghost; }
  /** For every owned+ghost atom: the owned index it is an image of. */
  owner = new Int32Array(0);
  /** Image offsets (3 per atom) relative to the owner; 0 for owned atoms. */
  gimage = new Int32Array(0);
  xall = new Float64Array(0);
  fall = new Float64Array(0);
  typeall = new Int32Array(0);
  qall = new Float64Array(0);
  half: NeighList | null = null;
  full: NeighList | null = null;

  lastBuild = -1;
  nbuild = 0;
  ndanger = 0;
  cutneighmax = 0;
  cutghost = 0;
  private xhold = new Float64Array(0);
  private boxhold: number[] = [];
  private needs: NeighborNeeds | null = null;

  constructor(skin: number) {
    this.skin = skin;
  }

  /** Records what the force field needs; call before every run (setup). */
  init(needs: NeighborNeeds): void {
    this.needs = needs;
    let mx = 0;
    for (let k = 0; k < needs.cutoff.length; k++) if (needs.cutoff[k] > mx) mx = needs.cutoff[k];
    this.cutneighmax = mx > 0 ? mx + this.skin : 0;
    this.cutghost = Math.max(this.cutneighmax, this.commCutoff);
  }

  /** Whether lists must be rebuilt on this step (neighbor->decide()). */
  decide(step: number, s: SimState, g: Geometry): boolean {
    if (this.once) return false;
    const ago = step - this.lastBuild;
    if (ago < this.delay) return false;
    if (this.every > 1 && step % this.every !== 0) return false;
    if (!this.check) return true;
    const moved = this.movedTooFar(s, g);
    if (moved && ago === this.delay && this.delay > 0) this.ndanger++;
    return moved;
  }

  private movedTooFar(s: SimState, g: Geometry): boolean {
    if (this.xhold.length !== 3 * s.n) return true;
    // a box change moves atoms relative to the ghosts: count the largest corner shift
    let boxDelta = 0;
    const bh = this.boxhold;
    const cur = [g.lo[0], g.lo[1], g.lo[2], g.hi[0], g.hi[1], g.hi[2], g.xy, g.xz, g.yz];
    for (let k = 0; k < cur.length; k++) boxDelta = Math.max(boxDelta, Math.abs(cur[k] - bh[k]));
    const lim = 0.5 * this.skin - 2 * boxDelta;
    if (lim <= 0) return true;
    const lim2 = lim * lim;
    const x = s.x, h = this.xhold;
    for (let k = 0; k < 3 * s.n; k += 3) {
      const dx = x[k] - h[k], dy = x[k + 1] - h[k + 1], dz = x[k + 2] - h[k + 2];
      if (dx * dx + dy * dy + dz * dz > lim2) return true;
    }
    return false;
  }

  /** Rebuilds ghosts and lists from the current (already remapped) positions. */
  build(s: SimState, g: Geometry, step: number): void {
    const needs = this.needs;
    if (!needs) throw new Error('neighbor lists used before init');
    this.makeGhosts(s, g);
    if (this.cutneighmax > 0) {
      if (needs.half) this.half = this.buildList(s, needs, false);
      else this.half = null;
      if (needs.full) this.full = this.buildList(s, needs, true);
      else this.full = null;
    } else {
      this.half = needs.half ? emptyList(s.n) : null;
      this.full = needs.full ? emptyList(s.n) : null;
    }
    this.xhold = Float64Array.from(s.x.subarray(0, 3 * s.n));
    this.boxhold = [g.lo[0], g.lo[1], g.lo[2], g.hi[0], g.hi[1], g.hi[2], g.xy, g.xz, g.yz];
    this.lastBuild = step;
    this.nbuild++;
  }

  /** Ghost atoms: every periodic image within cutghost of the box. */
  private makeGhosts(s: SimState, g: Geometry): void {
    const n = s.n;
    const cut = this.cutghost;
    const two = s.dimension === 2;
    // the cutoff in fractional units along each dimension (row norms of h^-1)
    const ext = [0, 1, 2].map((d) => {
      if (!g.periodic[d] || (two && d === 2) || cut <= 0) return -1;
      const e = [0, 0, 0];
      const l0 = [0, 0, 0];
      g.toLamda(g.lo[0], g.lo[1], g.lo[2], l0);
      // |row d of h^-1| = max change of lambda_d per unit displacement
      let norm2 = 0;
      for (let c = 0; c < 3; c++) {
        const p = [g.lo[0], g.lo[1], g.lo[2]];
        p[c] += 1;
        g.toLamda(p[0], p[1], p[2], e);
        norm2 += (e[d] - l0[d]) * (e[d] - l0[d]);
      }
      return cut * Math.sqrt(norm2);
    });
    const owner: number[] = [];
    const img: number[] = [];
    const lam = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
      const lo = [0, 0, 0], hi = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        if (ext[d] < 0) continue;
        lo[d] = Math.ceil(-ext[d] - lam[d]);
        hi[d] = Math.floor(1 + ext[d] - lam[d]);
        if (1 + ext[d] - lam[d] === hi[d]) hi[d]--;   // half-open on the upper side
      }
      for (let kz = lo[2]; kz <= hi[2]; kz++) {
        for (let ky = lo[1]; ky <= hi[1]; ky++) {
          for (let kx = lo[0]; kx <= hi[0]; kx++) {
            if (kx === 0 && ky === 0 && kz === 0) continue;
            owner.push(i);
            img.push(kx, ky, kz);
          }
        }
      }
    }
    this.nlocal = n;
    this.nghost = owner.length;
    const nall = n + owner.length;
    this.owner = new Int32Array(nall);
    for (let i = 0; i < n; i++) this.owner[i] = i;
    this.owner.set(owner, n);
    this.gimage = new Int32Array(3 * nall);
    this.gimage.set(img, 3 * n);
    if (this.xall.length !== 3 * nall) {
      this.xall = new Float64Array(3 * nall);
      this.fall = new Float64Array(3 * nall);
    }
    this.typeall = new Int32Array(nall);
    this.qall = new Float64Array(nall);
    for (let k = 0; k < nall; k++) {
      const o = this.owner[k];
      this.typeall[k] = s.type[o];
      this.qall[k] = s.q[o];
    }
    this.forwardComm(s, g);
  }

  /** Copies owned positions into xall and places every ghost at its owner's image. */
  forwardComm(s: SimState, g: Geometry): void {
    const n = this.nlocal;
    const xa = this.xall;
    xa.set(s.x.subarray(0, 3 * n));
    const nall = this.nall;
    const o = this.owner, im = this.gimage;
    const { lx, ly, lz, xy, xz, yz } = g;
    for (let k = n; k < nall; k++) {
      const i = o[k];
      const ix = im[3 * k], iy = im[3 * k + 1], iz = im[3 * k + 2];
      xa[3 * k] = s.x[3 * i] + ix * lx + iy * xy + iz * xz;
      xa[3 * k + 1] = s.x[3 * i + 1] + iy * ly + iz * yz;
      xa[3 * k + 2] = s.x[3 * i + 2] + iz * lz;
    }
  }

  /** Charges changed (set charge, fix qeq): refresh the ghost copies. */
  refreshCharges(s: SimState): void {
    for (let k = 0; k < this.nall; k++) this.qall[k] = s.q[this.owner[k]];
  }

  /** Zeroes the owned+ghost force accumulator (force_clear()). */
  clearForces(): void {
    this.fall.fill(0);
  }

  /** Adds owned and ghost forces in fall into the owners' state.f. */
  reverseComm(f: Float64Array): void {
    const fa = this.fall, o = this.owner;
    const nall = this.nall;
    for (let k = 0; k < nall; k++) {
      const i = o[k];
      f[3 * i] += fa[3 * k]; f[3 * i + 1] += fa[3 * k + 1]; f[3 * i + 2] += fa[3 * k + 2];
    }
  }

  /** Sums a per-atom array over ghosts into owners (stride values per atom). */
  reverseSum(a: Float64Array, stride: number, out: Float64Array): void {
    const o = this.owner;
    for (let k = 0; k < this.nall; k++) {
      const i = o[k];
      for (let c = 0; c < stride; c++) out[stride * i + c] += a[stride * k + c];
    }
  }

  /** Copies a per-owned-atom array to every ghost (stride values per atom). */
  forwardCopy(a: Float64Array, stride: number): void {
    const o = this.owner;
    for (let k = this.nlocal; k < this.nall; k++) {
      const i = o[k];
      for (let c = 0; c < stride; c++) a[stride * k + c] = a[stride * i + c];
    }
  }

  /**
   * A list for an arbitrary cutoff over the current ghosts (for computes
   * such as rdf or coord/atom). The cutoff must not exceed the ghost cutoff.
   */
  occasionalList(s: SimState, cutoff: number, fullList: boolean, ntypes: number): NeighList {
    const nt = ntypes + 1;
    const cut = new Float64Array(nt * nt).fill(cutoff);
    const needs: NeighborNeeds = {
      half: !fullList, full: fullList, cutoff: cut, ntypes, special: null,
      specialSettings: { lj: [1, 1, 1, 1], coul: [1, 1, 1, 1], keepExcluded: true },
    };
    return this.buildList(s, needs, fullList, 0);
  }

  /** The neigh_modify exclusions as one predicate over owned indices (null when none). */
  private excludeFn(s: SimState): ((i: number, j: number) => boolean) | null {
    if (!this.excludes.length) return null;
    const ex = this.excludes;
    return (i, j) => {
      for (const e of ex) {
        switch (e.kind) {
          case 'type': {
            const ti = s.type[i], tj = s.type[j];
            if ((ti === e.a && tj === e.b) || (ti === e.b && tj === e.a)) return true;
            break;
          }
          case 'group': {
            const mi = s.mask[i], mj = s.mask[j];
            if (((mi & e.a) && (mj & e.b)) || ((mi & e.b) && (mj & e.a))) return true;
            break;
          }
          case 'molecule/intra':
            if ((s.mask[i] & e.a) && (s.mask[j] & e.a) && s.molecule[i] === s.molecule[j]) return true;
            break;
          case 'molecule/inter':
            if ((s.mask[i] & e.a) && (s.mask[j] & e.a) && s.molecule[i] !== s.molecule[j]) return true;
            break;
        }
      }
      return false;
    };
  }

  private buildList(s: SimState, needs: NeighborNeeds, fullList: boolean, skin = this.skin): NeighList {
    const nlocal = this.nlocal, nall = this.nall;
    const xa = this.xall, ta = this.typeall, owner = this.owner;
    const nt = needs.ntypes + 1;
    const cutsq = new Float64Array(nt * nt);
    let cmax = 0;
    for (let k = 0; k < nt * nt; k++) {
      const c = needs.cutoff[k];
      if (c > 0) { cutsq[k] = (c + skin) * (c + skin); if (c + skin > cmax) cmax = c + skin; }
    }
    if (cmax <= 0) return emptyList(nlocal);
    // bins over the bounding box of all atoms, padded by the stencil reach so
    // the stencil is a flat list of index offsets with no bounds checks
    const binsize = this.binsize > 0 ? this.binsize : 0.5 * cmax;
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    for (let k = 0; k < nall; k++) {
      const x = xa[3 * k], y = xa[3 * k + 1], z = xa[3 * k + 2];
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
      if (z < z0) z0 = z; if (z > z1) z1 = z;
    }
    const mx = Math.max(1, Math.min(512, Math.floor((x1 - x0) / binsize) + 1));
    const my = Math.max(1, Math.min(512, Math.floor((y1 - y0) / binsize) + 1));
    const mz = Math.max(1, Math.min(512, Math.floor((z1 - z0) / binsize) + 1));
    // bins are never narrower than binsize: a tiny but nonzero extent (atoms almost in a line or a
    // plane) would otherwise make the stencil reach ~cmax/extent bins and exhaust memory
    const bx = Math.max(binsize, (x1 - x0) / mx), by = Math.max(binsize, (y1 - y0) / my), bz = Math.max(binsize, (z1 - z0) / mz);
    const sx = Math.ceil(cmax / bx), sy = Math.ceil(cmax / by), sz = Math.ceil(cmax / bz);
    const nbx = mx + 2 * sx, nby = my + 2 * sy, nbz = mz + 2 * sz;
    const nbins = nbx * nby * nbz;
    const binOf = new Int32Array(nall);
    const binCount = new Int32Array(nbins + 1);
    for (let k = 0; k < nall; k++) {
      let ix = Math.floor((xa[3 * k] - x0) / bx); if (ix >= mx) ix = mx - 1; if (ix < 0) ix = 0;
      let iy = Math.floor((xa[3 * k + 1] - y0) / by); if (iy >= my) iy = my - 1; if (iy < 0) iy = 0;
      let iz = Math.floor((xa[3 * k + 2] - z0) / bz); if (iz >= mz) iz = mz - 1; if (iz < 0) iz = 0;
      const b = ((iz + sz) * nby + iy + sy) * nbx + ix + sx;
      binOf[k] = b;
      binCount[b + 1]++;
    }
    for (let b = 0; b < nbins; b++) binCount[b + 1] += binCount[b];
    const binStart = binCount;
    const fill = binStart.slice(0, nbins);
    // atoms in bin order, with their positions and types copied contiguously
    const binAtoms = new Int32Array(nall);
    for (let k = 0; k < nall; k++) binAtoms[fill[binOf[k]]++] = k;
    const bxs = new Float64Array(3 * nall);
    const bts = new Int32Array(nall);
    for (let p = 0; p < nall; p++) {
      const k = binAtoms[p];
      bxs[3 * p] = xa[3 * k]; bxs[3 * p + 1] = xa[3 * k + 1]; bxs[3 * p + 2] = xa[3 * k + 2];
      bts[p] = ta[k];
    }
    // stencil: bins whose closest point can be within cmax, as linear offsets
    const sten: number[] = [];
    for (let dz = -sz; dz <= sz; dz++) {
      for (let dy = -sy; dy <= sy; dy++) {
        for (let dx = -sx; dx <= sx; dx++) {
          const gx = Math.max(0, Math.abs(dx) - 1) * bx, gy = Math.max(0, Math.abs(dy) - 1) * by, gz = Math.max(0, Math.abs(dz) - 1) * bz;
          if (gx * gx + gy * gy + gz * gz < cmax * cmax) sten.push((dz * nby + dy) * nbx + dx);
        }
      }
    }
    const stencil = Int32Array.from(sten);
    const ns = stencil.length;
    const special = needs.special;
    const ss = needs.specialSettings;
    const exclude = needs.exclude ?? this.excludeFn(s);
    const incl = this.includeBit;
    const plain = !special && !exclude && !incl;
    const numneigh = new Int32Array(nlocal);
    const firstneigh = new Int32Array(nlocal + 1);
    let cap = Math.max(64, nlocal * (fullList ? 96 : 48));
    let neigh = new Int32Array(cap);
    let count = 0;
    const id = s.id;
    for (let i = 0; i < nlocal; i++) {
      firstneigh[i] = count;
      if (incl && !(s.mask[i] & incl)) { numneigh[i] = 0; continue; }
      if (count + 2048 > cap) {
        cap = Math.max(2 * cap, count + 4096);
        const nn = new Int32Array(cap);
        nn.set(neigh.subarray(0, count));
        neigh = nn;
      }
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      const ti = ta[i] * nt;
      const b = binOf[i];
      const sp0 = special ? special.offset[i] : 0;
      const sp1 = special ? special.offset[i + 1] : 0;
      for (let q = 0; q < ns; q++) {
        const jb = b + stencil[q];
        const pEnd = binStart[jb + 1];
        for (let p = binStart[jb]; p < pEnd; p++) {
          const j = binAtoms[p];
          if (j === i) continue;
          const dx = bxs[3 * p] - xi, dy = bxs[3 * p + 1] - yi, dz = bxs[3 * p + 2] - zi;
          if (!fullList) {
            if (j < nlocal) { if (j < i) continue; }
            else if (dz < 0 || (dz === 0 && (dy < 0 || (dy === 0 && dx < 0)))) continue;
          }
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= cutsq[ti + bts[p]]) continue;
          if (plain) {
            if (count === cap) {
              cap *= 2;
              const nn = new Int32Array(cap);
              nn.set(neigh);
              neigh = nn;
            }
            neigh[count++] = j;
            continue;
          }
          const jo = owner[j];
          if (incl && !(s.mask[jo] & incl)) continue;
          if (exclude && exclude(i, jo)) continue;
          let enc = j;
          if (sp1 > sp0) {
            const jid = id[jo];
            let skip = false;
            for (let k = sp0; k < sp1; k++) {
              if (special!.partner[k] !== jid) continue;
              const o = special!.order[k];
              const lj = ss.lj[o], cl = ss.coul[o];
              if (lj === 1 && cl === 1) break;
              if (lj === 0 && cl === 0 && !ss.keepExcluded) { skip = true; break; }
              enc = j | (o << SBBITS);
              break;
            }
            if (skip) continue;
          }
          if (count === cap) {
            cap *= 2;
            const nn = new Int32Array(cap);
            nn.set(neigh);
            neigh = nn;
          }
          neigh[count++] = enc;
        }
      }
      numneigh[i] = count - firstneigh[i];
    }
    firstneigh[nlocal] = count;
    return { inum: nlocal, numneigh, firstneigh, neighbors: neigh.subarray(0, count) };
  }
}

const emptyList = (n: number): NeighList => ({
  inum: n, numneigh: new Int32Array(n), firstneigh: new Int32Array(n + 1), neighbors: new Int32Array(0),
});
