import type { SimState } from './types';
import type { Geometry } from './domain';
import type { SpecialList } from './atoms';
import type { BondedCopies } from './force/types';

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

/** One neigh_modify exclude entry (type pair, group pair, or the molecule/intra and molecule/inter group tests). */
export type Exclusion = { kind: 'type'; a: number; b: number } | { kind: 'group'; a: number; b: number }
  | { kind: 'molecule/intra' | 'molecule/inter'; a: number };

export interface NeighList {
  /** Number of owned atoms with lists (a kernel loops over atoms [ilo, inum)). */
  inum: number;
  /** First atom a kernel computes (default 0): the pair threads hand each thread chunks [ilo, inum) of one list. */
  ilo?: number;
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
  /** Size list (Pair.sizeList): a pair is kept only when r <= radius_i + radius_j + skin (atom_style sphere radii). */
  size?: boolean;
}

/** Whether two special lists hold the same partners, orders and offsets. */
const sameSpecial = (a: SpecialList | null, b: SpecialList | null): boolean => {
  if (a === b) return true;
  if (!a || !b) return false;
  if (a.offset.length !== b.offset.length || a.partner.length !== b.partner.length) return false;
  for (let i = 0; i < a.offset.length; i++) if (a.offset[i] !== b.offset[i]) return false;
  for (let k = 0; k < a.partner.length; k++) if (a.partner[k] !== b.partner[k] || a.order[k] !== b.order[k]) return false;
  return true;
};

/**
 * The 8 corners of the box [lo0, lo1, lo2, hi0, hi1, hi2, xy, xz, yz], corner c at fractional coordinates (c & 1,
 * (c >> 1) & 1, (c >> 2) & 1), each x = h lambda + lo summed in the order of native Domain::lamda2x.
 */
const boxCorners = (b: number[]): number[] => {
  const lx = b[3] - b[0], ly = b[4] - b[1], lz = b[5] - b[2];
  const out: number[] = [];
  for (let c = 0; c < 8; c++) {
    const l0 = c & 1, l1 = (c >> 1) & 1, l2 = (c >> 2) & 1;
    out.push(lx * l0 + b[6] * l1 + b[7] * l2 + b[0], ly * l1 + b[8] * l2 + b[1], lz * l2 + b[2]);
  }
  return out;
};

export class Neighbor {
  skin: number;
  every = 1;
  delay = 0;
  check = true;
  once = false;
  /** neighbor style multi (neighbor.html); neigh_modify bin/hash needs it. */
  multiStyle = false;
  /** neighbor style nsq (neighbor.html); dump image autobond reads it to place a periodic pair as native's list does. */
  nsqStyle = false;
  /** neigh_modify binsize (0 = half the largest neighbor cutoff). */
  binsize = 0;
  /** comm_modify cutoff: ghost cutoff at least this large. */
  commCutoff = 0;
  /**
   * The step whose neighbor decision is forced by a restart file being written on it (set by the
   * run loop before decide(); -1 = none). restart.html: "Writing a restart file forces a neighbor
   * list rebuild on that timestep" and this "happens regardless of the neigh_modify every, delay,
   * and check settings".
   */
  forceBuildStep = -1;
  /**
   * The step whose decision a fix forces (Fix::next_reneighbor, set by the run loop before decide(); -1 = none).
   * Unlike forceBuildStep, the build refreshes the check positions (see decide()).
   */
  fixBuildStep = -1;
  /**
   * neigh_modify exclude (neigh_modify.html): "type M N = exclude if one atom
   * in pair is type M, other is type N"; "group group1-ID group2-ID = exclude
   * if one atom is in 1st group, other in 2nd"; "molecule/intra group-ID =
   * exclude if both atoms are in the same molecule and in group";
   * "molecule/inter group-ID = exclude if both atoms are in different
   * molecules and in group".
   */
  excludes: Exclusion[] = [];
  /** neigh_modify include group-ID: only atoms of this group get pair neighbors (bit, 0 = all). */
  includeBit = 0;

  nlocal = 0;
  nghost = 0;
  get nall(): number { return this.nlocal + this.nghost; }
  /** For every owned+ghost atom: the owned index it is an image of. */
  owner = new Int32Array(0);
  /** Image offsets (3 per atom) relative to the owner; 0 for owned atoms. */
  gimage = new Int32Array(0);
  /** Owned atom i's ghosts are the indices ghostStart[i] .. ghostStart[i+1]-1 (length nlocal+1). */
  ghostStart = new Int32Array(0);
  /** The periodic copies for the bonded terms (force/types.ts BondedCopies). */
  bondedCopies(): BondedCopies { return { start: this.ghostStart, image: this.gimage }; }
  xall = new Float64Array(0);
  fall = new Float64Array(0);
  typeall = new Int32Array(0);
  qall = new Float64Array(0);
  half: NeighList | null = null;
  full: NeighList | null = null;
  /** Owned and ghost positions and the box bounds at the last list build (dump image autobond, autobond.ts). */
  xBuilt = new Float64Array(0);
  loBuilt: [number, number, number] = [0, 0, 0];
  hiBuilt: [number, number, number] = [0, 0, 0];

  lastBuild = -1;
  nbuild = 0;
  ndanger = 0;
  cutneighmax = 0;
  /**
   * The smallest neighbor cutoff over the type pairs (a pair without interaction counts as 0), as native's
   * Neighbor::init sets cutneighmin (src/neighbor.cpp at b332f28); dump image autobond warns past it.
   */
  cutneighmin = 0;
  cutghost = 0;
  private xhold = new Float64Array(0);
  private boxhold: number[] = [];
  private needs: NeighborNeeds | null = null;

  constructor(skin: number) {
    this.skin = skin;
  }

  /** Set when the special list changed (bonds broke): the next decide() rebuilds the lists. */
  private specialStale = false;

  /** Replaces the special list of the needs (bonds broke or were created during a run). */
  setSpecial(special: SpecialList | null): void {
    const old = this.needs?.special ?? null;
    if (this.needs) this.needs.special = special;
    // an unchanged list (the same pairs in the same order) needs no extra build: a forced build here would move
    // every later build, and with it the sorting and the random streams of the pair styles that depend on it
    if (!sameSpecial(old, special)) this.specialStale = true;
  }

  /** Records what the force field needs; call before every run (setup). */
  init(needs: NeighborNeeds): void {
    this.needs = needs;
    let mx = 0;
    for (let k = 0; k < needs.cutoff.length; k++) if (needs.cutoff[k] > mx) mx = needs.cutoff[k];
    this.cutneighmax = mx > 0 ? mx + this.skin : 0;
    let mn = Infinity;
    for (let i = 1; i <= needs.ntypes; i++) {
      for (let j = 1; j <= needs.ntypes; j++) {
        const c = needs.cutoff[i * (needs.ntypes + 1) + j];
        mn = Math.min(mn, c > 0 ? c + this.skin : 0);
      }
    }
    this.cutneighmin = mn === Infinity ? 0 : mn;
    this.cutghost = Math.max(this.cutneighmax, this.commCutoff);
  }

  /** Whether lists must be rebuilt on this step (neighbor->decide()). */
  decide(step: number, s: SimState, g: Geometry): boolean {
    this.forcedDecision = false;
    if (this.once) return false;
    if (step === this.forceBuildStep) { this.forcedDecision = true; return true; }
    // a rebuild that a fix requests (fix srd next_reneighbor, rheo): native Neighbor::decide() returns 1 and
    // Neighbor::build() stores the reference positions of the check (xhold), as on every build
    if (step === this.fixBuildStep) return true;
    const ago = step - this.lastBuild;
    if (ago < this.delay) return false;
    if (this.specialStale) { this.specialStale = false; return true; }
    if (this.every > 1 && step % this.every !== 0) return false;
    if (!this.check) return true;
    const moved = this.movedTooFar(s, g);
    if (moved && ago === this.delay && this.delay > 0) this.ndanger++;
    return moved;
  }

  /** The distance check of neigh_modify check yes since the last build (fix deform remap v asks it too). */
  movedTooFar(s: SimState, g: Geometry): boolean {
    if (this.xhold.length !== 3 * s.n) return true;
    // As native LAMMPS does (src/neighbor.cpp at b332f28, check_distance): with a periodic dimension the threshold
    // shrinks by the motion of the box since the last build, 0.5 (skin - (delta1 + delta2)) clamped at 0, where
    // delta1 and delta2 are the displacements of the lower and upper box corners (orthogonal box), or the two values
    // the corner loop keeps for a triclinic box (the largest so far, and a larger one than delta2 that was not a new
    // largest: "if (delta > delta1) delta1 = delta; else if (delta > delta2) delta2 = delta"). Without a periodic
    // dimension (boxcheck 0) it is triggersq = 0.25 skin^2. The corners are Domain::lamda_box_corners's, in its order.
    const bh = this.boxhold;
    let lim2: number;
    if (!(g.periodic[0] || g.periodic[1] || (s.dimension === 3 && g.periodic[2]))) {
      lim2 = 0.25 * this.skin * this.skin;
    } else {
      let delta1 = 0, delta2 = 0;
      if (!g.triclinic) {
        let dx = g.lo[0] - bh[0], dy = g.lo[1] - bh[1], dz = g.lo[2] - bh[2];
        delta1 = Math.sqrt(dx * dx + dy * dy + dz * dz);
        dx = g.hi[0] - bh[3]; dy = g.hi[1] - bh[4]; dz = g.hi[2] - bh[5];
        delta2 = Math.sqrt(dx * dx + dy * dy + dz * dz);
      } else {
        const now = boxCorners([g.lo[0], g.lo[1], g.lo[2], g.hi[0], g.hi[1], g.hi[2], g.xy, g.xz, g.yz]);
        const hold = boxCorners(bh);
        for (let c = 0; c < 24; c += 3) {
          const dx = now[c] - hold[c], dy = now[c + 1] - hold[c + 1], dz = now[c + 2] - hold[c + 2];
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (d > delta1) delta1 = d;
          else if (d > delta2) delta2 = d;
        }
      }
      let delta = 0.5 * (this.skin - (delta1 + delta2));
      if (delta < 0) delta = 0;
      lim2 = delta * delta;
    }
    const x = s.x, h = this.xhold;
    // neigh_modify.html: "The include option limits the building of pairwise neighbor lists to atoms in the
    // specified group." Measured with native LAMMPS (black box): only the atoms of that group take part in the
    // check (fix srd example: fast SRD particles outside the big group do not force a rebuild; the candidate counts
    // of fix srd agree step by step only then).
    const incl = this.includeBit;
    for (let i = 0, k = 0; i < s.n; i++, k += 3) {
      if (incl && !(s.mask[i] & incl)) continue;
      const dx = x[k] - h[k], dy = x[k + 1] - h[k + 1], dz = x[k + 2] - h[k + 2];
      if (dx * dx + dy * dy + dz * dz > lim2) return true;
    }
    return false;
  }

  /**
   * True when the last decide() returned true only because of forceBuildStep. Such a build does
   * not move the check reference positions (see build()).
   */
  private forcedDecision = false;

  /**
   * Rebuilds ghosts and lists from the current (already remapped) positions. A build forced by a
   * restart file leaves the reference positions and box of the check setting alone, so the next
   * check still measures the motion since the last build of its own schedule.
   * Measured with native LAMMPS (black box, LJ melt 500 atoms, restart 25 and neigh_modify every 10
   * check yes): the builds at steps 30, 70 and 130 still happen after the forced builds at 25, 75
   * and 125 (the engine without this rule skipped them), and the trajectories agree after that.
   */
  build(s: SimState, g: Geometry, step: number): void {
    const needs = this.needs;
    if (!needs) throw new Error('neighbor lists used before init');
    // every build uses the current special list, so a list handed over before it (read_data, create_bonds) needs
    // no extra build at the next step
    this.specialStale = false;
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
    this.xBuilt = this.xall.slice(0, 3 * this.nall);
    this.loBuilt = [g.lo[0], g.lo[1], g.lo[2]];
    this.hiBuilt = [g.hi[0], g.hi[1], g.hi[2]];
    if (!this.forcedDecision || this.xhold.length !== 3 * s.n) {
      this.xhold = Float64Array.from(s.x.subarray(0, 3 * s.n));
      this.boxhold = [g.lo[0], g.lo[1], g.lo[2], g.hi[0], g.hi[1], g.hi[2], g.xy, g.xz, g.yz];
    }
    this.forcedDecision = false;
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
    const gstart = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) {
      gstart[i] = n + owner.length;
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
    gstart[n] = n + owner.length;
    this.ghostStart = gstart;
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

  /**
   * Copies owned positions into xall and places every ghost at its owner's image. The owned atoms' types and charges
   * are refreshed too: as native LAMMPS does (src/comm_brick.cpp forward_comm and borders at b332f28), a pair style reads
   * the live type and q of an owned atom, while a ghost keeps the values its owner had at the last borders()
   * (the neighbor build) until the next one.
   */
  forwardComm(s: SimState, g: Geometry): void {
    const n = this.nlocal;
    const xa = this.xall;
    xa.set(s.x.subarray(0, 3 * n));
    if (this.typeall.length >= n) { this.typeall.set(s.type.subarray(0, n)); this.qall.set(s.q.subarray(0, n)); }
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
  /** The neigh_modify exclusions as a test on two owned atoms (null when there are none). */
  exclusion(s: SimState): ((i: number, j: number) => boolean) | null {
    return this.excludeFn(s);
  }

  private excludeFn(s: SimState): ((i: number, j: number) => boolean) | null {
    if (!this.excludes.length) return null;
    // the type exclusions as a table over type pairs (both orders), the others tested in turn: the same predicate, an
    // OR of every exclusion, without a loop over the type entries for each candidate pair
    const nt = s.ntypes + 1;
    const typeTab = new Uint8Array(nt * nt);
    let anyType = false;
    const others: Exclusion[] = [];
    for (const e of this.excludes) {
      if (e.kind !== 'type') { others.push(e); continue; }
      anyType = true;
      if (e.a >= 1 && e.a < nt && e.b >= 1 && e.b < nt) { typeTab[e.a * nt + e.b] = 1; typeTab[e.b * nt + e.a] = 1; }
    }
    const byType = (i: number, j: number): boolean => typeTab[s.type[i] * nt + s.type[j]] === 1;
    if (!others.length) return byType;
    return (i, j) => {
      if (anyType && byType(i, j)) return true;
      for (const e of others) {
        switch (e.kind) {
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

  // Scratch buffers for buildList: grown on demand, never handed out with a list.
  private sBinOf = new Int32Array(0);
  private sBinStart = new Int32Array(0);
  private sBinGhost = new Int32Array(0);
  private sCursor = new Int32Array(0);
  private sAtoms = new Int32Array(0);
  private sAtomsG = new Int32Array(0);
  private sBx = new Float64Array(0);
  private sBxG = new Float64Array(0);
  private sBt = new Int32Array(0);
  private sBtG = new Int32Array(0);
  private sBr = new Float64Array(0);
  private sBrG = new Float64Array(0);
  private sPairA = new Int32Array(0);
  private sPairJ = new Int32Array(0);

  /**
   * Builds one list over the owned and ghost atoms. Owned and ghost atoms are binned
   * into two bin-sorted arrays, so each row of the stencil (five bins along x, as
   * far as the cutoff reaches) is one contiguous range of owned atoms and one of
   * ghosts. Every candidate pair is visited once: owned-owned, owned-ghost (from the
   * owned side) and ghost-owned (from the ghost side), never ghost-ghost. The
   * membership rules are the class comment's: a half list keeps an owned-owned pair
   * at the lower index and an owned-ghost pair only when the ghost lies "above";
   * a full list keeps every neighbor. Special bits, exclusions and the include group
   * apply per entry. Entries are collected as (owner, neighbor) pairs in scratch
   * buffers and counting-sorted into fresh per-atom CSR arrays, so a returned list
   * never aliases a later build.
   */
  private buildList(s: SimState, needs: NeighborNeeds, fullList: boolean, skin = this.skin): NeighList {
    const nlocal = this.nlocal, nall = this.nall, nghost = nall - nlocal;
    const xa = this.xall, ta = this.typeall, owner = this.owner;
    const nt = needs.ntypes + 1;
    const cutsq = new Float64Array(nt * nt);
    let cmax = 0;
    for (let k = 0; k < nt * nt; k++) {
      const c = needs.cutoff[k];
      if (c > 0) { cutsq[k] = (c + skin) * (c + skin); if (c + skin > cmax) cmax = c + skin; }
    }
    if (cmax <= 0) return emptyList(nlocal);
    // neigh_modify exclude type M N (when this list takes the neigh_modify exclusions): the type pair's cutoff is set
    // below any distance, so its pairs are dropped at the distance test instead of after it, as native's npair build
    // tests the exclusion before the distance. The bins (from cmax) and the order of the other pairs are unchanged.
    if (!needs.exclude) {
      for (const e of this.excludes) {
        if (e.kind !== 'type' || e.a < 1 || e.a >= nt || e.b < 1 || e.b >= nt) continue;
        cutsq[e.a * nt + e.b] = -1;
        cutsq[e.b * nt + e.a] = -1;
      }
    }
    // bins over the bounding box of all atoms; the stencil reaches past the box, so every
    // stencil row stays inside the padded bin grid
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

    // stencil rows: for the upper half of the stencil (dz > 0, or dz = 0 and dy > 0) the bins with
    // dx in [-m, m] are contiguous in bin order, as is the row dx in [1, m0] of the own z-y plane
    const cmax2 = cmax * cmax;
    const gapOK = (dx: number, dy: number, dz: number): boolean => {
      const gx = Math.max(0, Math.abs(dx) - 1) * bx, gy = Math.max(0, Math.abs(dy) - 1) * by, gz = Math.max(0, Math.abs(dz) - 1) * bz;
      return gx * gx + gy * gy + gz * gz < cmax2;
    };
    const rowLo: number[] = [], rowHi: number[] = [];
    for (let dz = 0; dz <= sz; dz++) {
      for (let dy = -sy; dy <= sy; dy++) {
        if (dz === 0 && dy <= 0) continue;
        let m = -1;
        for (let d = 0; d <= sx; d++) if (gapOK(d, dy, dz)) m = d;
        if (m < 0) continue;
        const base = (dz * nby + dy) * nbx;
        rowLo.push(base - m); rowHi.push(base + m);
      }
    }
    let m0 = 0;
    for (let d = 1; d <= sx; d++) if (gapOK(d, 0, 0)) m0 = d;
    if (m0 >= 1) { rowLo.push(1); rowHi.push(m0); }
    const nrows = rowLo.length;
    const rLo = Int32Array.from(rowLo), rHi = Int32Array.from(rowHi);

    // counting sort of the owned and of the ghost atoms into bins (two grids, one per class)
    const binOf = (this.sBinOf = grownI32(this.sBinOf, nall));
    const bL = (this.sBinStart = grownI32(this.sBinStart, nbins + 1));
    const bG = (this.sBinGhost = grownI32(this.sBinGhost, nbins + 1));
    const cursor = (this.sCursor = grownI32(this.sCursor, Math.max(2 * nbins, nlocal, nghost)));
    bL.fill(0, 0, nbins + 1);
    bG.fill(0, 0, nbins + 1);
    for (let k = 0; k < nall; k++) {
      // the box covers every atom, so the scaled offsets are non-negative
      let ix = ((xa[3 * k] - x0) / bx) | 0; if (ix >= mx) ix = mx - 1;
      let iy = ((xa[3 * k + 1] - y0) / by) | 0; if (iy >= my) iy = my - 1;
      let iz = ((xa[3 * k + 2] - z0) / bz) | 0; if (iz >= mz) iz = mz - 1;
      const b = ((iz + sz) * nby + iy + sy) * nbx + ix + sx;
      binOf[k] = b;
      if (k < nlocal) bL[b + 1]++; else bG[b + 1]++;
    }
    for (let b = 0; b < nbins; b++) { bL[b + 1] += bL[b]; bG[b + 1] += bG[b]; }
    const lk = (this.sAtoms.length >= nlocal ? this.sAtoms : (this.sAtoms = new Int32Array(Math.max(nlocal, 2 * this.sAtoms.length))));
    const gk = (this.sAtomsG.length >= nghost ? this.sAtomsG : (this.sAtomsG = new Int32Array(Math.max(nghost, 2 * this.sAtomsG.length))));
    const lxs = (this.sBx.length >= 3 * nlocal ? this.sBx : (this.sBx = new Float64Array(Math.max(3 * nlocal, 2 * this.sBx.length))));
    const gxs = (this.sBxG.length >= 3 * nghost ? this.sBxG : (this.sBxG = new Float64Array(Math.max(3 * nghost, 2 * this.sBxG.length))));
    const lt = (this.sBt.length >= nlocal ? this.sBt : (this.sBt = new Int32Array(Math.max(nlocal, 2 * this.sBt.length))));
    const gt = (this.sBtG.length >= nghost ? this.sBtG : (this.sBtG = new Int32Array(Math.max(nghost, 2 * this.sBtG.length))));
    // size list (NeighborNeeds.size): the radii in bin order, owned and ghost (a ghost has its owner's radius)
    const radius = needs.size ? s.radius : null;
    const lr = radius ? (this.sBr.length >= nlocal ? this.sBr : (this.sBr = new Float64Array(Math.max(nlocal, 2 * this.sBr.length)))) : null;
    const gr = radius ? (this.sBrG.length >= nghost ? this.sBrG : (this.sBrG = new Float64Array(Math.max(nghost, 2 * this.sBrG.length)))) : null;
    for (let b = 0; b < nbins; b++) cursor[b] = bL[b];
    for (let b = 0; b < nbins; b++) cursor[nbins + b] = bG[b];
    for (let k = 0; k < nall; k++) {
      const b = binOf[k];
      if (k < nlocal) {
        const p = cursor[b]++;
        lk[p] = k; lt[p] = ta[k];
        lxs[3 * p] = xa[3 * k]; lxs[3 * p + 1] = xa[3 * k + 1]; lxs[3 * p + 2] = xa[3 * k + 2];
        if (lr) lr[p] = radius![k];
      } else {
        const p = cursor[nbins + b]++;
        gk[p] = k; gt[p] = ta[k];
        gxs[3 * p] = xa[3 * k]; gxs[3 * p + 1] = xa[3 * k + 1]; gxs[3 * p + 2] = xa[3 * k + 2];
        if (gr) gr[p] = radius![owner[k]];
      }
    }

    const special = needs.special;
    const ss = needs.specialSettings;
    const exclude = needs.exclude ?? this.excludeFn(s);
    const incl = this.includeBit;
    const plain = !special && !exclude && !incl;
    const id = s.id;
    // Measured with native LAMMPS (black box): a special partner seen through a periodic image more
    // than half a box length away (in any periodic dimension) is an ordinary neighbor. A bonded pair
    // 1.2 apart in a 5-wide periodic box with lj/cut 4.0 has its image pair at 3.8 counted at weight
    // 1 for special_bonds lj 0, 0.5 and 1 alike, while the direct pair takes the special weight. The
    // test is per Cartesian dimension against half the box edge, in triclinic boxes too (xy = 2 in a
    // 5-wide box: of the images (-2.3, 2.4), (2.7, 2.4) and (0.7, -2.6) only the first is special).
    const box = s.box, per = box.periodic;
    const hx = 0.5 * (box.hi[0] - box.lo[0]), hy = 0.5 * (box.hi[1] - box.lo[1]), hz = 0.5 * (box.hi[2] - box.lo[2]);
    const farImage = (a: number, j: number): boolean => {
      const dx = xa[3 * j] - xa[3 * a], dy = xa[3 * j + 1] - xa[3 * a + 1], dz = xa[3 * j + 2] - xa[3 * a + 2];
      return (per[0] && Math.abs(dx) > hx) || (per[1] && Math.abs(dy) > hy) || (per[2] && Math.abs(dz) > hz);
    };
    // the entry for owned atom a and neighbor j (index into owned+ghost), NaN when rejected
    const encode = (a: number, j: number): number => {
      const jo = owner[j];
      if (incl && (!(s.mask[a] & incl) || !(s.mask[jo] & incl))) return NaN;
      if (exclude && exclude(a, jo)) return NaN;
      if (!special) return j;
      const sp1 = special.offset[a + 1];
      for (let k = special.offset[a]; k < sp1; k++) {
        if (special.partner[k] !== id[jo]) continue;
        if (farImage(a, j)) return j;
        const o = special.order[k];
        const lj = ss.lj[o], cl = ss.coul[o];
        if (lj === 1 && cl === 1) return j;
        if (lj === 0 && cl === 0 && !ss.keepExcluded) return NaN;
        return j | (o << SBBITS);
      }
      return j;
    };

    // entries (owned source, neighbor code) as parallel buffers
    let cap = Math.min(this.sPairA.length, this.sPairJ.length);
    if (cap < 1024) cap = Math.max(1024, nlocal * (fullList ? 96 : 48));
    let pa = this.sPairA.length >= cap ? this.sPairA : new Int32Array(cap);
    let pj = this.sPairJ.length >= cap ? this.sPairJ : new Int32Array(cap);
    cap = Math.min(pa.length, pj.length);
    let np = 0;

    for (let b = 0; b < nbins; b++) {
      const l0 = bL[b], l1 = bL[b + 1], g0 = bG[b], g1 = bG[b + 1];
      if (l1 === l0 && g1 === g0) continue;
      // row -1 is the bin itself (its owned atoms pair with later owned atoms of the bin), then the
      // stencil rows: every row is a contiguous range of owned atoms and of ghosts
      for (let r = -1; r < nrows; r++) {
        const own = r < 0;
        let lA = l0, lB = l1, gA = g0, gB = g1;
        if (!own) {
          const ba = b + rLo[r], bb = b + rHi[r];
          lA = bL[ba]; lB = bL[bb + 1];
          gA = bG[ba]; gB = bG[bb + 1];
          if (lB === lA && gB === gA) continue;
        }
        // owned p with owned q (q later in the bin, for the own row) and with ghost q
        for (let p = l0; p < l1; p++) {
          const kp = lk[p], tp = lt[p];
          const px = lxs[3 * p], py = lxs[3 * p + 1], pz = lxs[3 * p + 2];
          const qs = own ? p + 1 : lA;
          if (lB > qs) {
            const need = np + 2 * (lB - qs);
            if (need > cap) {
              cap = Math.max(2 * cap, need);
              const na = new Int32Array(cap); na.set(pa.subarray(0, np)); pa = na;
              const nj = new Int32Array(cap); nj.set(pj.subarray(0, np)); pj = nj;
            }
            for (let q = qs; q < lB; q++) {
              const kq = lk[q], tq = lt[q];
              if (cutsq[tp * nt + tq] < 0) continue; // an excluded type pair (both orders)
              const dx = lxs[3 * q] - px, dy = lxs[3 * q + 1] - py, dz = lxs[3 * q + 2] - pz;
              let r2 = dx * dx + dy * dy + dz * dz;
              // a size list keeps the pair only within radsum + skin (Pair.sizeList); NaN fails every test below
              if (lr) { const rc = lr[p] + lr[q] + skin; if (!(r2 <= rc * rc)) r2 = NaN; }
              if (fullList) {
                const c1 = r2 < cutsq[tp * nt + tq], c2 = r2 < cutsq[tq * nt + tp];
                if (plain) {
                  pa[np] = kp; pj[np] = kq; np += +c1;
                  pa[np] = kq; pj[np] = kp; np += +c2;
                } else {
                  if (c1) { const e = encode(kp, kq); if (e === e) { pa[np] = kp; pj[np] = e; np++; } }
                  if (c2) { const e = encode(kq, kp); if (e === e) { pa[np] = kq; pj[np] = e; np++; } }
                }
              } else {
                // half: the lower index owns the pair
                const lo = kp < kq ? kp : kq, hi = kp < kq ? kq : kp;
                const c = r2 < (kp < kq ? cutsq[tp * nt + tq] : cutsq[tq * nt + tp]);
                if (plain) { pa[np] = lo; pj[np] = hi; np += +c; }
                else if (c) { const e = encode(lo, hi); if (e === e) { pa[np] = lo; pj[np] = e; np++; } }
              }
            }
          }
          if (gB > gA) {
            const need = np + 2 * (gB - gA);
            if (need > cap) {
              cap = Math.max(2 * cap, need);
              const na = new Int32Array(cap); na.set(pa.subarray(0, np)); pa = na;
              const nj = new Int32Array(cap); nj.set(pj.subarray(0, np)); pj = nj;
            }
            for (let q = gA; q < gB; q++) {
              const kq = gk[q], tq = gt[q];
              if (cutsq[tp * nt + tq] < 0) continue;
              const dx = gxs[3 * q] - px, dy = gxs[3 * q + 1] - py, dz = gxs[3 * q + 2] - pz;
              let r2 = dx * dx + dy * dy + dz * dz;
              if (lr) { const rc = lr[p] + gr![q] + skin; if (!(r2 <= rc * rc)) r2 = NaN; }
              // ghost q lies "above" the owned atom p when its offset is lex-positive
              const c = r2 < cutsq[tp * nt + tq] && (fullList || keepAbove(dx, dy, dz));
              if (plain) { pa[np] = kp; pj[np] = kq; np += +c; }
              else if (c) { const e = encode(kp, kq); if (e === e) { pa[np] = kp; pj[np] = e; np++; } }
            }
          }
        }
        // ghost p with owned q of the stencil row (the own row is covered from the owned side)
        if (!own && lB > lA) {
          for (let p = g0; p < g1; p++) {
            const kp = gk[p], tp = gt[p];
            const px = gxs[3 * p], py = gxs[3 * p + 1], pz = gxs[3 * p + 2];
            const need = np + 2 * (lB - lA);
            if (need > cap) {
              cap = Math.max(2 * cap, need);
              const na = new Int32Array(cap); na.set(pa.subarray(0, np)); pa = na;
              const nj = new Int32Array(cap); nj.set(pj.subarray(0, np)); pj = nj;
            }
            for (let q = lA; q < lB; q++) {
              const kq = lk[q], tq = lt[q];
              if (cutsq[tq * nt + tp] < 0) continue;
              const dx = lxs[3 * q] - px, dy = lxs[3 * q + 1] - py, dz = lxs[3 * q + 2] - pz;
              let r2 = dx * dx + dy * dy + dz * dz;
              if (lr) { const rc = lr[q] + gr![p] + skin; if (!(r2 <= rc * rc)) r2 = NaN; }
              // the owned atom is q: the ghost lies at offset p - q
              const c = r2 < cutsq[tq * nt + tp] && (fullList || keepAbove(-dx, -dy, -dz));
              if (plain) { pa[np] = kq; pj[np] = kp; np += +c; }
              else if (c) { const e = encode(kq, kp); if (e === e) { pa[np] = kq; pj[np] = e; np++; } }
            }
          }
        }
      }
    }

    // counting sort of the entries into per-atom rows
    const numneigh = new Int32Array(nlocal);
    for (let e = 0; e < np; e++) numneigh[pa[e]]++;
    const firstneigh = new Int32Array(nlocal + 1);
    for (let i = 0; i < nlocal; i++) firstneigh[i + 1] = firstneigh[i] + numneigh[i];
    const neighbors = new Int32Array(np);
    const pos = this.sCursor;
    pos.set(firstneigh.subarray(0, nlocal));
    for (let e = 0; e < np; e++) neighbors[pos[pa[e]]++] = pj[e];
    this.sPairA = pa;
    this.sPairJ = pj;
    return { inum: nlocal, numneigh, firstneigh, neighbors };
  }
}

/** Whether a ghost at offset (dx, dy, dz) from an owned atom belongs to its half list (the lower mirror keeps the pair). */
const keepAbove = (dx: number, dy: number, dz: number): boolean =>
  !(dz < 0 || (dz === 0 && (dy < 0 || (dy === 0 && dx < 0))));

const grownI32 = (a: Int32Array<ArrayBuffer>, n: number): Int32Array<ArrayBuffer> => (a.length >= n ? a : new Int32Array(Math.max(n, 2 * a.length)));

const emptyList = (n: number): NeighList => ({
  inum: n, numneigh: new Int32Array(n), firstneigh: new Int32Array(n + 1), neighbors: new Int32Array(0),
});
