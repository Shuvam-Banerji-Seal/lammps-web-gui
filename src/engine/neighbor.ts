import type { SimState } from './types';
import type { Geometry } from './domain';
import { nativeOrder, type SpecialList } from './atoms';
import { StyleError, type BondedCopies } from './force/types';
import { binBound, halfBinsPairs, halfBinsPairsSpecial, keepGhostPair, type HalfBins, type ListSegment, type NeighborThreads, type SpecialBins } from './cpu/neighborKernel';
import { gplHook } from './gplHooks';

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
  /** The neighbors are already in the order the style asked for (Pair.threadNeighbors), on the pair threads. */
  nativeOrdered?: boolean;
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
  /** newton_pair (default on); off, native's half list keeps an owned/ghost pair at both atoms (the one-atom count). */
  newton?: boolean;
  /** A pair hybrid style: native gives each sub-style a list of its own cutoff and type pairs (the one-atom count). */
  hybrid?: boolean;
}

/** What the neigh_modify one count (the lazily loaded helper neigh_one) reads of a list build. */
export interface NeighOneInput {
  /** the list (half: each pair once; full: each pair at both atoms) */
  list: NeighList;
  full: boolean;
  nlocal: number;
  /** owned and ghost positions; owner of each (the owned atom a ghost is an image of); NEIGHMASK */
  x: Float64Array;
  owner: Int32Array;
  mask: number;
  /** atom IDs and each owned atom's position in native's storage order */
  tag: Int32Array;
  rank: Int32Array;
  /** neighbor nsq (else bin), the box (bounding box of a triclinic cell), neigh_modify binsize, cutneighmax */
  nsq: boolean;
  triclinic: boolean;
  newton: boolean;
  boxLo: [number, number, number];
  boxHi: [number, number, number];
  binsize: number;
  cutneighmax: number;
  dimension: number;
  units: string;
}

/** The helper of the neigh_modify one check: the largest per-atom entry count of the list as native builds it. */
export interface NeighOneHook {
  maxCount(input: NeighOneInput): number;
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
  /**
   * A fix of the run makes a perpetual full list request (Fix.nativeFullList), so native derives the half lists from it
   * (src/neighbor.cpp morph_halffull at b332f28): the native pair order follows the full list's rows. Set at setup.
   */
  nativeFullParent = false;
  /** neigh_modify binsize (0 = half the largest neighbor cutoff). */
  binsize = 0;
  /** comm_modify cutoff: ghost cutoff at least this large. */
  commCutoff = 0;
  /**
   * comm_modify mode multi (comm_modify.html: "The *multi* communication mode is only compatible with the *multi*
   * neighbor style."). The ghost atoms of a single process are those of the largest cutoff/multi value either way.
   */
  commMulti = false;
  /**
   * neigh_modify page and one (neigh_modify.html: "The option defaults are ... page = 100000, one = 2000"). The engine
   * stores its lists in growable arrays; the values are kept for native's check of the two at run setup.
   */
  pgsize = 100000;
  oneatom = 2000;
  /** neigh_modify collection/type or collection/interval: the number of custom collections (0 = one per atom type). */
  ncollections = 0;
  /**
   * comm_modify group (bit, 0 = all): comm_modify.html: "The *group* keyword will limit communication to atoms in the
   * specified group." As native LAMMPS does (src/comm_brick.cpp Comm::borders at b332f28): only the atoms of the group
   * (atom_modify first) are sent as ghosts, so the other atoms have no periodic images.
   */
  borderBit = 0;
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
  bondedCopies(): BondedCopies { return { start: this.ghostStart, image: this.gimage, x: this.xall, built: this.xBuilt }; }
  xall = new Float64Array(0);
  /** Forces on owned and ghost atoms; in shared memory when there are threads (they add their parts in parallel). */
  fall: Float64Array = new Float64Array(0);
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
    // As native LAMMPS does (src/neighbor.cpp Neighbor::init at b332f28): the settings are checked at run setup
    if (this.delay > 0 && this.delay % this.every !== 0) throw new StyleError('Neighbor delay must be 0 or multiple of every setting');
    if (this.pgsize < 10 * this.oneatom) throw new StyleError('Neighbor page size must be >= 10x the one atom setting');
    for (const e of this.excludes) {
      if (e.kind === 'type' && (e.a <= 0 || e.a > needs.ntypes || e.b <= 0 || e.b > needs.ntypes)) {
        throw new StyleError('Invalid atom type in neighbor exclusion list');
      }
    }
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
      this.checkOneAtom(s, needs);
    } else {
      this.half = needs.half ? emptyList(s.n) : null;
      this.full = needs.full ? emptyList(s.n) : null;
    }
    this.xBuilt = this.xall.slice(0, 3 * this.nall);
    this.loBuilt = [g.lo[0], g.lo[1], g.lo[2]];
    this.hiBuilt = [g.hi[0], g.hi[1], g.hi[2]];
    this.recordBuild(s, g, step);
  }

  /**
   * A build of the accelerated run loop (run/accel.ts), whose backends bin the atoms themselves each step: what build()
   * records for the next decide() (the reference positions and box of the check, the step, the count), without the
   * ghosts and lists, so the box resets and lost-atom checks of System.pbc() come on the steps of the CPU path.
   */
  markBuilt(s: SimState, g: Geometry, step: number): void {
    this.specialStale = false;
    // the lamda System.pbc left for this build's ghosts (a triclinic box) has no use without them
    this.freshLamda = null;
    this.recordBuild(s, g, step);
  }

  private recordBuild(s: SimState, g: Geometry, step: number): void {
    if (!this.forcedDecision || this.xhold.length !== 3 * s.n) {
      this.xhold = Float64Array.from(s.x.subarray(0, 3 * s.n));
      this.boxhold = [g.lo[0], g.lo[1], g.lo[2], g.hi[0], g.hi[1], g.hi[2], g.xy, g.xz, g.yz];
    }
    this.forcedDecision = false;
    this.lastBuild = step;
    this.nbuild++;
  }

  /**
   * neigh_modify one (neigh_modify.html: "one value = N  N = max # of neighbors of one atom"). As native LAMMPS does
   * (src/npair_bin.cpp, src/npair_nsq.cpp and src/my_page.h at b332f28): a list keeps the neighbors of each atom in one
   * chunk of at most N entries, and the build stops with "Neighbor list overflow, boost neigh_modify one" when an atom
   * has more. A full list holds each pair at both atoms; a half list at one of them, as native's build decides (its
   * bins, the storage order and the atom IDs), which the lazily loaded helper neigh_one works out only when some atom
   * has more than N neighbors in all. The half list native derives from a full list when both are needed has no more
   * entries than the full list, so the full list is the one checked. Not checked: neighbor style multi (its bins are
   * per collection) and pair hybrid styles (src/pair_hybrid.cpp at b332f28: each sub-style's list has the sub-style's
   * cutoff, "request->set_cutoff_max(cutmax_style[istyle])", and skips the type pairs of the other sub-styles, while
   * the engine's list holds every pair within the largest cutoff).
   */
  private checkOneAtom(s: SimState, needs: NeighborNeeds): void {
    const list = needs.full ? this.full : this.half;
    if (!list || this.multiStyle || needs.hybrid) return;
    const one = this.oneatom, nlocal = this.nlocal;
    if (this.oneBound <= one) return;
    let over = false;
    if (needs.full) {
      for (let i = 0; i < nlocal && !over; i++) over = list.numneigh[i] > one;
    } else if (list.neighbors.length > one) {
      // every neighbor of each atom, the pair counted at both of its atoms: no native count is larger
      const all = Int32Array.from(list.numneigh);
      const nb = list.neighbors, owner = this.owner;
      for (let k = 0; k < nb.length; k++) all[owner[nb[k] & NEIGHMASK]]++;
      for (let i = 0; i < nlocal && !over; i++) over = all[i] > one;
    }
    if (!over) return;
    const order = nativeOrder(s);
    const rank = new Int32Array(nlocal);
    for (let p = 0; p < nlocal; p++) rank[order[p]] = p;
    // the bins cover the box, or the bounding box of a triclinic cell (Domain::set_global_box boxlo_bound, boxhi_bound)
    const b = s.box;
    const lo: [number, number, number] = [b.lo[0], b.lo[1], b.lo[2]];
    const hi: [number, number, number] = [b.hi[0], b.hi[1], b.hi[2]];
    if (b.triclinic) {
      const [xy, xz, yz] = b.tilt;
      lo[0] = Math.min(lo[0], lo[0] + xy);
      lo[0] = Math.min(lo[0], lo[0] + xz);
      lo[1] = Math.min(lo[1], lo[1] + yz);
      hi[0] = Math.max(hi[0], hi[0] + xy);
      hi[0] = Math.max(hi[0], hi[0] + xz);
      hi[1] = Math.max(hi[1], hi[1] + yz);
    }
    const max = gplHook<NeighOneHook>('neigh_one').maxCount({
      list, full: needs.full, nlocal, x: this.xall, owner: this.owner.subarray(0, this.nall), mask: NEIGHMASK,
      tag: s.id, rank, nsq: this.nsqStyle, triclinic: b.triclinic, newton: needs.newton !== false, boxLo: lo, boxHi: hi,
      binsize: this.binsize, cutneighmax: this.cutneighmax, dimension: s.dimension, units: s.units.style,
    });
    if (max > one) throw new StyleError('Neighbor list overflow, boost neigh_modify one\nFor more information see https://docs.lammps.org/err0036');
  }

  private sGhostOwner = new Int32Array(0);
  private sGhostImage = new Int32Array(0);

  /**
   * The owned atoms' lamda after the wrap of the reneighboring (System.pbc in a triclinic box), for the next ghost build
   * only; null otherwise (the ghosts are then placed in Cartesian coordinates, as forwardComm does).
   */
  freshLamda: Float64Array | null = null;

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
    // the ghosts' owners and image offsets, in scratch arrays grown as needed
    let owner = this.sGhostOwner, img = this.sGhostImage;
    let ng = 0;
    const lam = [0, 0, 0];
    const gstart = new Int32Array(n + 1);
    const border = this.borderBit;
    const fresh = g.triclinic && this.freshLamda !== null && this.freshLamda.length === 3 * n ? this.freshLamda : null;
    this.freshLamda = null;
    for (let i = 0; i < n; i++) {
      gstart[i] = n + ng;
      if (border !== 0 && (s.mask[i] & border) === 0) continue;
      if (fresh) { lam[0] = fresh[3 * i]; lam[1] = fresh[3 * i + 1]; lam[2] = fresh[3 * i + 2]; } else g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
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
            if (ng === owner.length) {
              const o2 = new Int32Array(Math.max(1024, 2 * ng)); o2.set(owner); owner = o2;
              const i2 = new Int32Array(3 * o2.length); i2.set(img); img = i2;
            }
            owner[ng] = i;
            img[3 * ng] = kx; img[3 * ng + 1] = ky; img[3 * ng + 2] = kz;
            ng++;
          }
        }
      }
    }
    this.sGhostOwner = owner;
    this.sGhostImage = img;
    gstart[n] = n + ng;
    this.ghostStart = gstart;
    this.nlocal = n;
    this.nghost = ng;
    const nall = n + ng;
    this.owner = new Int32Array(nall);
    for (let i = 0; i < n; i++) this.owner[i] = i;
    this.owner.set(owner.subarray(0, ng), n);
    this.gimage = new Int32Array(3 * nall);
    this.gimage.set(img.subarray(0, 3 * ng), 3 * n);
    if (this.xall.length !== 3 * nall || (this.listThreads !== null && !(this.fall.buffer instanceof SharedArrayBuffer))) {
      this.xall = new Float64Array(3 * nall);
      this.fall = this.listThreads !== null && typeof SharedArrayBuffer !== 'undefined' ? new Float64Array(new SharedArrayBuffer(24 * nall)) : new Float64Array(3 * nall);
    }
    this.typeall = new Int32Array(nall);
    this.qall = new Float64Array(nall);
    for (let k = 0; k < nall; k++) {
      const o = this.owner[k];
      this.typeall[k] = s.type[o];
      this.qall[k] = s.q[o];
    }
    this.forwardComm(s, g);
    if (fresh) {
      // As native LAMMPS does (src/comm_brick.cpp borders and src/domain.cpp lamda2x at b332f28): in a triclinic box the
      // borders are exchanged in lamda, each ghost at its owner's lamda plus the image shift, and lamda2x then gives
      // x = h lamda + lo; forward_comm adds the Cartesian edge vectors only between reneighborings
      const xa = this.xall, im = this.gimage, o = this.owner;
      const { lx, ly, lz, xy, xz, yz } = g;
      for (let k = n; k < nall; k++) {
        const i = o[k];
        const l0 = fresh[3 * i] + im[3 * k], l1 = fresh[3 * i + 1] + im[3 * k + 1], l2 = fresh[3 * i + 2] + im[3 * k + 2];
        xa[3 * k] = lx * l0 + xy * l1 + xz * l2 + g.lo[0];
        xa[3 * k + 1] = ly * l1 + yz * l2 + g.lo[1];
        xa[3 * k + 2] = lz * l2 + g.lo[2];
      }
    }
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

  /** Adds each ghost's values of a per-atom array to its owner's, in ghost order (a pair style's reverse communication). */
  reverseAddGhosts(a: Float64Array, stride: number): void {
    const o = this.owner;
    for (let k = this.nlocal; k < this.nall; k++) {
      const i = o[k];
      for (let c = 0; c < stride; c++) a[stride * i + c] += a[stride * k + c];
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
    return this.buildList(s, needs, fullList, 0, true);
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
  /**
   * An upper bound of the neighbors of any owned atom in a list built from these bins (checkOneAtom): the atoms of the
   * full stencil around its bin (the rows of both halves and the bin itself), less the atom. The fullest bin times the
   * number of stencil bins first; the sums per bin only when that exceeds neigh_modify one.
   */
  private stencilBound(bL: Int32Array, bG: Int32Array, nbins: number, rLo: Int32Array, rHi: Int32Array, nrows: number): number {
    let nst = 1;
    for (let r = 0; r < nrows; r++) nst += 2 * (rHi[r] - rLo[r] + 1);
    let fullest = 0;
    for (let b = 0; b < nbins; b++) {
      const c = bL[b + 1] - bL[b] + bG[b + 1] - bG[b];
      if (c > fullest) fullest = c;
    }
    if (fullest * nst - 1 <= this.oneatom) return fullest * nst - 1;
    let bound = 0;
    for (let b = 0; b < nbins; b++) {
      if (bL[b + 1] === bL[b]) continue;
      let c = bL[b + 1] - bL[b] + bG[b + 1] - bG[b] - 1;
      for (let r = 0; r < nrows; r++) {
        c += bL[b + rHi[r] + 1] - bL[b + rLo[r]] + bG[b + rHi[r] + 1] - bG[b + rLo[r]];
        c += bL[b - rLo[r] + 1] - bL[b - rHi[r]] + bG[b - rLo[r] + 1] - bG[b - rHi[r]];
      }
      if (c > bound) bound = c;
    }
    return bound;
  }

  /** The atom IDs of the owned and the ghost atoms in bin order (neighborKernel.ts HalfBins ltag, gtag). */
  private sTagL = new Int32Array(0);
  private sTagG = new Int32Array(0);

  /** Shared-memory threads for the pair search of the common half list (set by the session with the pair threads). */
  listThreads: NeighborThreads | null = null;

  /** stencilBound of the last pair-list build (checkOneAtom). */
  private oneBound = 0;

  private buildList(s: SimState, needs: NeighborNeeds, fullList: boolean, skin = this.skin, skip0 = false): NeighList {
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
    // a dimension with a single bin holds every atom in that bin (a 2d system in z): stencil rows reaching past it would
    // only visit empty padding bins, so its reach is 0 (as native LAMMPS's 2d stencils, which have no z rows); the pairs
    // and their order are unchanged
    const sx = mx === 1 ? 0 : Math.ceil(cmax / bx), sy = my === 1 ? 0 : Math.ceil(cmax / by), sz = mz === 1 ? 0 : Math.ceil(cmax / bz);
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
    // each row's bin offsets in y and z, and the bin index of its dx = 0 (for the bin gaps of a size list, below)
    const rowDy: number[] = [], rowDz: number[] = [], rowBase: number[] = [];
    for (let dz = 0; dz <= sz; dz++) {
      for (let dy = -sy; dy <= sy; dy++) {
        if (dz === 0 && dy <= 0) continue;
        let m = -1;
        for (let d = 0; d <= sx; d++) if (gapOK(d, dy, dz)) m = d;
        if (m < 0) continue;
        const base = (dz * nby + dy) * nbx;
        rowLo.push(base - m); rowHi.push(base + m);
        rowDy.push(dy); rowDz.push(dz); rowBase.push(base);
      }
    }
    let m0 = 0;
    for (let d = 1; d <= sx; d++) if (gapOK(d, 0, 0)) m0 = d;
    if (m0 >= 1) { rowLo.push(1); rowHi.push(m0); rowDy.push(0); rowDz.push(0); rowBase.push(0); }
    const nrows = rowLo.length;
    const rLo = Int32Array.from(rowLo), rHi = Int32Array.from(rowHi);

    // counting sort of the owned and of the ghost atoms into bins (two grids, one per class)
    const binOf = (this.sBinOf = grownI32(this.sBinOf, nall));
    const bL = (this.sBinStart = grownI32(this.sBinStart, nbins + 1));
    const bG = (this.sBinGhost = grownI32(this.sBinGhost, nbins + 1));
    const cursor = (this.sCursor = grownI32(this.sCursor, Math.max(2 * nbins, nlocal, nghost)));
    bL.fill(0, 0, nbins + 1);
    bG.fill(0, 0, nbins + 1);
    // neigh_modify include: as native LAMMPS does (src/nbin_standard.cpp bin_atoms and src/npair_bin.cpp at b332f28,
    // "if (includegroup)"), only the atoms of the group (owned, and ghosts of the group's atoms) are binned, so the
    // other atoms are never visited; the pairs of the group and their order are those of the full binning
    const inclBin = this.includeBit;
    for (let k = 0; k < nall; k++) {
      if (inclBin && !(s.mask[k < nlocal ? k : owner[k]] & inclBin)) { binOf[k] = -1; continue; }
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
    if (!skip0) this.oneBound = this.stencilBound(bL, bG, nbins, rLo, rHi, nrows);
    for (let b = 0; b < nbins; b++) cursor[b] = bL[b];
    for (let b = 0; b < nbins; b++) cursor[nbins + b] = bG[b];
    for (let k = 0; k < nall; k++) {
      const b = binOf[k];
      if (b < 0) continue;
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
    // a size list: the largest radius of each bin's atoms, so that a stencil bin farther from an owned atom than its
    // radius + that radius + skin (every pair test there would fail) is not visited; the pairs and their order are
    // unchanged (Neighbor.buildList with neighbor multi and a polydisperse granular system visits mostly such bins)
    let binRmax: Float64Array | null = null;
    if (lr) {
      binRmax = new Float64Array(nbins).fill(-Infinity);
      for (let k = 0; k < nall; k++) {
        const b = binOf[k];
        if (b < 0) continue;
        const r = radius![k < nlocal ? k : owner[k]];
        if (r > binRmax[b]) binRmax[b] = r;
      }
    }
    // the [start, end) segments of a row's owned (or ghost) atoms that owned atom p may pair with: one segment, or for a
    // size list one per bin within reach, in bin order (the atom order)
    let segA = new Int32Array(64), segB = new Int32Array(64);
    const rowSegments = (starts: Int32Array, r: number, b: number, rp: number): number => {
      const ba = b + rLo[r], bb = b + rHi[r];
      if (!binRmax) { segA[0] = starts[ba]; segB[0] = starts[bb + 1]; return 1; }
      if (segA.length < bb - ba + 1) { segA = new Int32Array(2 * (bb - ba + 1)); segB = new Int32Array(2 * (bb - ba + 1)); }
      const gy = Math.max(0, Math.abs(rowDy[r]) - 1) * by, gz = Math.max(0, Math.abs(rowDz[r]) - 1) * bz;
      const gyz = gy * gy + gz * gz;
      let n = 0;
      for (let bi = ba; bi <= bb; bi++) {
        const q0 = starts[bi], q1 = starts[bi + 1];
        if (q1 === q0) continue;
        const gx = Math.max(0, Math.abs(bi - b - rowBase[r]) - 1) * bx;
        const reach = (rp + binRmax[bi] + skin) * (1 + 1e-9);
        if (gx * gx + gyz > reach * reach) continue;
        if (n > 0 && segB[n - 1] === q0) segB[n - 1] = q1;
        else { segA[n] = q0; segB[n] = q1; n++; }
      }
      return n;
    };

    // which copy of a pair through a periodic image the half list keeps (keepGhostPair): as native LAMMPS does
    // (src/npair_bin.cpp and src/npair_nsq.cpp at b332f28), by the atom IDs in a triclinic box and for neighbor nsq when
    // newton is on, an atom and its own image within delta = 0.01 Angstrom compared coordinate by coordinate (triclinic)
    const rule = fullList || needs.newton === false ? 0 : s.box.triclinic ? 1 : this.nsqStyle ? 2 : 0;
    const delta = 0.01 * (ANGSTROM_UNITS[s.units.style] ?? 1);
    const keepPair = (i: number, j: number, ox: number, oy: number, oz: number): boolean =>
      rule === 0 ? keepAbove(ox, oy, oz) : keepGhostPair(rule, s.id[i], s.id[owner[j]], ox, oy, oz, delta);

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
    // comm_modify group: an owned atom outside the border group has no ghosts, so its pairs with the ghosts of the
    // group's atoms have no mirror pair. Native keeps such a pair or not by its bin order (npair_bin.cpp), which this
    // half list does not reproduce: count them, and refuse below when there are any
    const accepted = (a: number, j: number): boolean => { const e = encode(a, j); return e === e; };
    let unmirrored: Uint8Array | null = null;
    let nUnmirrored = 0;
    if (this.borderBit !== 0 && !fullList) {
      unmirrored = new Uint8Array(nlocal);
      for (let i = 0; i < nlocal; i++) unmirrored[i] = (s.mask[i] & this.borderBit) === 0 ? 1 : 0;
    }

    // the common list (half, no special bonds, exclusions, include group or radii) is built by
    // cpu/neighborKernel.ts, on the shared-memory threads when there are some; it stores the same entries in the same
    // order without the per-pair branches of the others
    // the threads' entries are segments of their arrays, which the counting sort below reads in place
    let segs: ListSegment[] | null = null;
    if (!fullList && !exclude && !incl && !lr && unmirrored === null) {
      const B: HalfBins = { bL, bG, rLo, rHi, nrows, lk, lt, lxs, gk, gt, gxs, cutsq, nt, rule, delta };
      if (rule !== 0) {
        // the atom IDs in bin order (a ghost has its owner's ID)
        const ltag = (this.sTagL = grownI32(this.sTagL, nlocal)), gtag = (this.sTagG = grownI32(this.sTagG, nghost));
        for (let p = 0; p < nlocal; p++) ltag[p] = s.id[lk[p]];
        for (let q = 0; q < nghost; q++) gtag[q] = s.id[owner[gk[q]]];
        B.ltag = ltag; B.gtag = gtag;
      }
      // with special bonds the kernel encodes each pair as encode() does
      const S: SpecialBins | null = special ? {
        offset: special.offset, partner: special.partner, order: special.order, id, owner,
        lj: Float64Array.from(ss.lj), coul: Float64Array.from(ss.coul), keep: ss.keepExcluded,
        hx: per[0] ? hx : Infinity, hy: per[1] ? hy : Infinity, hz: per[2] ? hz : Infinity, sbbits: SBBITS,
      } : null;
      const binMax = binBound(B, nbins);
      const threaded = this.listThreads?.halfList(B, S, nbins, nlocal, binMax) ?? null;
      if (threaded) {
        segs = threaded.segs; np = threaded.np;
      } else {
        const out = new Int32Array(2);
        for (let b = 0; b < nbins;) {
          if (S) halfBinsPairsSpecial(B, S, b, nbins, pa, pj, np, cap, binMax, out);
          else halfBinsPairs(B, b, nbins, pa, pj, np, cap, binMax, out);
          b = out[0]; np = out[1];
          if (b < nbins) {
            cap = Math.max(2 * cap, np + binMax);
            const na = new Int32Array(cap); na.set(pa.subarray(0, np)); pa = na;
            const nj = new Int32Array(cap); nj.set(pj.subarray(0, np)); pj = nj;
          }
        }
      }
    } else for (let b = 0; b < nbins; b++) {
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
            let nseg = 1;
            if (own) { segA[0] = qs; segB[0] = lB; } else nseg = rowSegments(bL, r, b, lr ? lr[p] : 0);
            for (let sg = 0; sg < nseg; sg++) for (let q = segA[sg], q1 = segB[sg]; q < q1; q++) {
              const kq = lk[q], tq = lt[q];
              if (cutsq[tp * nt + tq] < 0) continue; // an excluded type pair (both orders)
              const dx = lxs[3 * q] - px, dy = lxs[3 * q + 1] - py, dz = lxs[3 * q + 2] - pz;
              let r2 = dx * dx + dy * dy + dz * dz;
              // a size list keeps the pair only within radsum + skin (Pair.sizeList); NaN fails every test below
              if (lr) { const rc = lr[p] + lr[q] + skin; if (!(r2 <= rc * rc)) r2 = NaN; }
              if (fullList) {
                const c1 = r2 <= cutsq[tp * nt + tq], c2 = r2 <= cutsq[tq * nt + tp];
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
                const c = r2 <= (kp < kq ? cutsq[tp * nt + tq] : cutsq[tq * nt + tp]);
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
            let ngs = 1;
            if (own) { segA[0] = gA; segB[0] = gB; } else ngs = rowSegments(bG, r, b, lr ? lr[p] : 0);
            for (let sg = 0; sg < ngs; sg++) for (let q = segA[sg], q1 = segB[sg]; q < q1; q++) {
              const kq = gk[q], tq = gt[q];
              if (cutsq[tp * nt + tq] < 0) continue;
              const dx = gxs[3 * q] - px, dy = gxs[3 * q + 1] - py, dz = gxs[3 * q + 2] - pz;
              let r2 = dx * dx + dy * dy + dz * dz;
              if (lr) { const rc = lr[p] + gr![q] + skin; if (!(r2 <= rc * rc)) r2 = NaN; }
              // ghost q lies "above" the owned atom p when its offset is lex-positive
              const c = r2 <= cutsq[tp * nt + tq] && (fullList || keepPair(kp, kq, dx, dy, dz));
              if (unmirrored !== null && unmirrored[kp] && r2 <= cutsq[tp * nt + tq] && (plain || accepted(kp, kq))) nUnmirrored++;
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
              const c = r2 <= cutsq[tq * nt + tp] && (fullList || keepPair(kq, kp, -dx, -dy, -dz));
              if (unmirrored !== null && unmirrored[kq] && r2 <= cutsq[tq * nt + tp] && (plain || accepted(kq, kp))) nUnmirrored++;
              if (plain) { pa[np] = kq; pj[np] = kp; np += +c; }
              else if (c) { const e = encode(kq, kp); if (e === e) { pa[np] = kq; pj[np] = e; np++; } }
            }
          }
        }
      }
    }

    if (nUnmirrored > 0) {
      throw new StyleError(`comm_modify group: ${nUnmirrored} pair(s) between an atom outside the group and a periodic image of a group atom are within the cutoff; the browser engine does not reproduce which of them native's half neighbor list keeps`);
    }

    // counting sort of the entries into per-atom rows
    segs ??= [{ a: pa, j: pj, o: 0, n: np }];
    const numneigh = new Int32Array(nlocal);
    for (const { a, o, n } of segs) for (let e = o; e < o + n; e++) numneigh[a[e]]++;
    const firstneigh = new Int32Array(nlocal + 1);
    for (let i = 0; i < nlocal; i++) firstneigh[i + 1] = firstneigh[i] + numneigh[i];
    const neighbors = new Int32Array(np);
    const pos = this.sCursor;
    pos.set(firstneigh.subarray(0, nlocal));
    for (const { a, j, o, n } of segs) for (let e = o; e < o + n; e++) neighbors[pos[a[e]]++] = j[e];
    this.sPairA = pa;
    this.sPairJ = pj;
    return { inum: nlocal, numneigh, firstneigh, neighbors };
  }
}

/**
 * One Angstrom in each unit style's distance unit, as native LAMMPS sets force->angstrom (src/update.cpp set_units at
 * b332f28; docs.lammps.org/units.html: "distance = Angstroms" for real and metal, meters for si, centimeters for cgs,
 * Bohr for electron, micrometers for micro, nanometers for nano, and 1 for lj).
 */
const ANGSTROM_UNITS: Record<string, number> = { lj: 1.0, real: 1.0, metal: 1.0, si: 1.0e-10, cgs: 1.0e-8, electron: 1.88972612, micro: 1.0e-4, nano: 1.0e-1 };

/** Whether a ghost at offset (dx, dy, dz) from an owned atom belongs to its half list (the lower mirror keeps the pair). */
const keepAbove = (dx: number, dy: number, dz: number): boolean =>
  !(dz < 0 || (dz === 0 && (dy < 0 || (dy === 0 && dx < 0))));

const grownI32 = (a: Int32Array<ArrayBuffer>, n: number): Int32Array<ArrayBuffer> => (a.length >= n ? a : new Int32Array(Math.max(n, 2 * a.length)));

const emptyList = (n: number): NeighList => ({
  inum: n, numneigh: new Int32Array(n), firstneigh: new Int32Array(n + 1), neighbors: new Int32Array(0),
});
