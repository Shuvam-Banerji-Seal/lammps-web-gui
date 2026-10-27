import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { logical } from '../force/util';
import { hasMolecule } from '../atoms';
import { gplHook } from '../gplHooks';

/*
 * Pairwise group computes (docs.lammps.org/compute_group_group.html and
 * docs.lammps.org/compute_tally.html): compute group/group, pe/tally,
 * force/tally and stress/tally. All four visit the pairs of the half neighbour
 * list that have one atom in group1 and the other in group2, and use the pair
 * style's single() for the energy and force of each pair. Bonded, kspace and
 * many-body terms are not part of them.
 *
 * Quotes used:
 *   compute_group_group.html: "Define a computation that calculates the total energy and force
 *   interaction between two groups of atoms: the compute group and the specified group2."
 *   "The two groups can be the same."
 *   "If the *pair* keyword is set to *yes*, which is the default, then the the interaction energy
 *   will include a pair component which is defined as the pairwise energy between all pairs of
 *   atoms where one atom in the pair is in the first group and the other is in the second group."
 *   "For the *inter* setting, the two atoms must be in different molecules."
 *   "For the *intra* setting, the two atoms must be in the same molecule."
 *   "This compute calculates a global scalar (the energy) and a global vector of length 3 (force),
 *   which can be accessed by indices 1--3."
 *   "The option defaults are pair = yes, kspace = no, boundary = yes, molecule = off."
 *   "If the *kspace* keyword is set to *yes*, which is not the default, and if a
 *   :doc:`kspace_style <kspace_style>` is defined, then the interaction energy will include a Kspace
 *   component which is the long-range Coulombic energy between all the atoms in the first group and
 *   all the atoms in the second group."
 *   compute_tally.html: "Define a computation that calculates properties between two groups of atoms by
 *   accumulating them from pairwise non-bonded computations."
 *   "Compute *pe/tally* calculates a global scalar (the energy) and a per atom scalar (the contributions
 *   of the single atom to the global scalar)."
 *   "Compute *force/tally* calculates a global scalar (the force magnitude) and a per atom 3-element
 *   vector (force contribution from each atom)."
 *   "Compute *stress/tally* calculates a global scalar (average of the diagonal elements of the stress
 *   tensor) and a per atom vector (the six elements of stress tensor contributions from the individual
 *   atom)."
 *   "The pairwise contributions are computing via a callback that the compute registers with the
 *   non-bonded pairwise force computation."
 *
 * Measured with native LAMMPS (black box, 4 atoms, lj/cut/coul/cut, units lj, which print thermo
 * values normalised by the atom count; the dumps are not normalised):
 *   - a pair (i, j) counts once, whatever the two groups are: group1 = group2 = one atom gives 0, and
 *     compute group/group of a group with itself counts each unordered pair once;
 *   - the scalar of pe/tally, group/group and force/tally sums the counted pairs once: pe/tally with both
 *     groups "all" equals the total pair energy;
 *   - pe/tally per-atom values: each atom of a counted pair gets half of the pair energy, in two columns
 *     (evdwl, ecoul); summed over all atoms that is the scalar;
 *   - force/tally per-atom values: atom i gets the force on it from j and atom j the opposite force; the
 *     scalar is the magnitude of the sum over the atoms of group1, which is the group/group vector;
 *   - stress/tally per-atom values: each atom of a counted pair gets -1/2 of the pair virial (the outer
 *     product d (x) F, d = x_i - x_j, F the force on i); the scalar is the pair virial trace divided by 3
 *     and is not divided by the atom count in thermo output (unlike the energy and force scalars).
 *   - group/group with lj/cut/coul/cut and special bonds: the special factors enter through the neighbour
 *     list, as for the pair energy.
 * The per-atom outputs are listed by compute ID in the engine's per-atom convention (length n, indexed by
 * the local atom index); ghosts add to their owner.
 *
 * kspace yes for group/group: the long-range part comes from the lazily loaded helper (GroupGroupKspaceHook).
 * pe/mol/tally, heat/flux/tally and heat/flux/virial/tally are ports in src/engine-gpl (lazyStyles.ts).
 */

export type MolMode = 'off' | 'inter' | 'intra';

/** The values a pair style passes to ev_tally for one pair (the tally callbacks' evdwl, ecoul and fpair). */
type TallyTerms = (i: number, j: number, itype: number, jtype: number, rsq: number) => { evdwl: number; ecoul: number; fforce: number };

const NEIGH_MASK = (1 << 30) - 1;

/**
 * Calls cb once per pair counted by a tally compute: a pair of the half list with one atom in
 * bit1 and the other in bit2 (in either order), filtered by the molecule mode. cb gets the
 * local indices, the pair energy split into evdwl and ecoul, the force-over-r of the pair
 * (force on i = fforce * d) and d = x_i - x_j (minimum image through the neighbour list).
 */
export const walkTallyPairs = (
  sys: System,
  owner: string,
  bit1: number,
  bit2: number,
  mol: MolMode,
  cb: (i: number, j: number, evdwl: number, ecoul: number, fforce: number, dx: number, dy: number, dz: number) => void,
): void => {
  const pair = sys.ff.pair;
  if (!pair) throw new StyleError(`compute ${owner} needs a pair_style (none is defined)`);
  if (!pair.single || pair.singleNeedsSeparation()) throw new StyleError(`compute ${owner} needs a pair_style that supports single() (pairwise evaluation); pair_style ${pair.name} does not`);
  sys.forces();
  const s = sys.state;
  const nb = sys.nb;
  const list = nb.half;
  if (!list) throw new StyleError(`compute ${owner}: no half neighbour list is available`);
  // special-bond weights of the last force evaluation (index = special bits of the neighbour)
  const ff = sys.ff as unknown as { specialLJ: Float64Array; specialCoul: Float64Array };
  const sLJ = ff.specialLJ, sC = ff.specialCoul;
  const nt = pair.ntypes + 1;
  const nbr = list.neighbors;
  const xa = nb.xall;
  for (let i = 0; i < list.inum; i++) {
    const mi = s.mask[i];
    const k0: number = list.firstneigh[i], k1: number = k0 + list.numneigh[i];
    for (let k: number = k0; k < k1; k++) {
      const jj: number = nbr[k];
      const j: number = jj & NEIGH_MASK;
      const sb = jj >>> 30;
      const oj = nb.owner[j];
      const mj = s.mask[oj];
      const counted = ((mi & bit1) !== 0 && (mj & bit2) !== 0) || ((mj & bit1) !== 0 && (mi & bit2) !== 0);
      if (!counted) continue;
      if (mol !== 'off') {
        const same = s.molecule[i] === s.molecule[oj];
        if (mol === 'inter' && same) continue;
        if (mol === 'intra' && !same) continue;
      }
      const dx = xa[3 * i] - xa[3 * j], dy = xa[3 * i + 1] - xa[3 * j + 1], dz = xa[3 * i + 2] - xa[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const ti = nb.typeall[i], tj = nb.typeall[j];
      if (rsq >= pair.cutsq[ti * nt + tj]) continue;
      const fc = sC[sb], fl = sLJ[sb];
      const qi = nb.qall[i], qj = nb.qall[j];
      const oi = nb.owner[i];
      // a style whose single() is not what its compute tallies (pair eam: native's ev_tally values) gives them itself
      const tt = (pair as { tallyTerms?: TallyTerms }).tallyTerms;
      if (tt) {
        const v = tt.call(pair, oi, oj, ti, tj, rsq);
        cb(i, j, v.evdwl, v.ecoul, v.fforce, dx, dy, dz);
        continue;
      }
      const tot = pair.single(oi, oj, ti, tj, rsq, fc, fl, qi, qj);
      // Coulomb part alone (the LJ factor 0 switches the LJ term off); the rest is the LJ part
      const ecoul = pair.single(oi, oj, ti, tj, rsq, fc, 0, qi, qj).eng;
      cb(i, j, tot.eng - ecoul, ecoul, tot.fforce, dx, dy, dz);
    }
  }
};

/**
 * As native LAMMPS's tally computes do (src/TALLY/compute_*_tally.cpp compute_scalar, compute_vector and compute_peratom
 * at b332f28: "if (!force->pair->did_tally_callback())"): a warning when the pair style's force computation tallies no
 * pair, as pair zero and zero/coul (src/pair_zero.cpp compute, no ev_tally call) do. what: Energy for pe/tally and
 * pe/mol/tally, Stress for the others.
 */
export const warnUntallied = (sys: System, what: 'Energy' | 'Stress'): void => {
  const name = sys.ff.pair?.name;
  if (name === 'zero' || name === 'zero/coul') sys.warn(`${what} was not tallied by pair style\nFor more information see https://docs.lammps.org/err0011`);
};

/** Per-atom tally values are indexed by the local atom index; a ghost neighbour adds to its owner. */
const atomIndex = (sys: System, j: number): number => sys.nb.owner[j];

/** The self-energy and k = 0 terms of the long-range part of compute group/group, found at init. */
export interface GroupGroupKspaceTerms { eSelf: number; eCorrection: number }

/**
 * The long-range part of compute group/group (kspace yes), from the lazily loaded helper of the style word
 * (src/engine-gpl/compute/group_group_kspace.ts, through gplHook('group/group')).
 */
export interface GroupGroupKspaceHook {
  /** At init: the kspace style checks. */
  init(sys: System): void;
  /** The self-energy and k = 0 terms of the two groups (native's init, with the solver's G-ewald). */
  terms(sys: System, bitA: number, bitB: number): GroupGroupKspaceTerms;
  /** The long-range energy of the two groups (with the k = 0 term when boundary is yes) and the force on the first. */
  contribution(sys: System, bitA: number, bitB: number, terms: GroupGroupKspaceTerms, boundary: boolean): { e: number; f: number[] };
}

/** compute group/group group-ID group2-ID [pair yes|no] [kspace yes|no] [boundary yes|no] [molecule off|inter|intra]. */
export class ComputeGroupGroup extends Compute {
  readonly style = 'group/group';
  private readonly group2Bit: number;
  private readonly pairOn: boolean;
  private readonly kspaceOn: boolean;
  private readonly boundaryOn: boolean;
  private readonly molMode: MolMode;
  private kspaceTerms: GroupGroupKspaceTerms | null = null;
  private cacheEpoch = -1;
  private energy = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/compute_group_group.cpp at b332f28): group2 (Group::get_bitmask_by_id), then the
    // keywords; every bad word is "Illegal compute group/group command"
    const illegal = (): StyleError => new StyleError('Illegal compute group/group command');
    if (!args.length) throw illegal();
    if (sys.groups.find(args[0]) < 0) throw new StyleError(`Group ID ${args[0]} requested by compute group/group does not exist`);
    this.group2Bit = sys.groups.bit(args[0]);
    let pairOn = true, kspaceOn = false, boundaryOn = true, mol: MolMode = 'off';
    for (let a = 1; a < args.length; a += 2) {
      const key = args[a], val = args[a + 1];
      if (key !== 'pair' && key !== 'kspace' && key !== 'boundary' && key !== 'molecule') throw illegal();
      if (val === undefined) throw illegal();
      if (key === 'pair' || key === 'kspace' || key === 'boundary') {
        const on = logical(val);
        if (key === 'pair') pairOn = on;
        else if (key === 'kspace') kspaceOn = on;
        else boundaryOn = on;
      } else {
        if (val !== 'off' && val !== 'inter' && val !== 'intra') throw illegal();
        if (val !== 'off' && !hasMolecule(sys.state)) throw new StyleError('Compute group/group molecule requires molecule IDs');
        mol = val;
      }
    }
    this.pairOn = pairOn;
    this.kspaceOn = kspaceOn;
    this.boundaryOn = boundaryOn;
    this.molMode = mol;
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extscalar = 1;
    this.extvector = 1;
    this.vector = new Float64Array(3);
  }

  init(): void {
    // As native LAMMPS does (src/compute_group_group.cpp ComputeGroupGroup::init at b332f28): the pair style checks (a
    // hybrid style is left to its sub-styles), then the kspace style's and its correction terms (the lazily loaded
    // helper), with the G-ewald the run's force init has chosen
    const pair = this.sys.ff.pair;
    if (this.pairOn && !pair) throw new StyleError('No pair style defined for compute group/group');
    if (pair && !pair.name.startsWith('hybrid') && !pair.single) throw new StyleError('Pair style does not support compute group/group');
    this.kspaceTerms = null;
    if (this.kspaceOn) {
      const hook = gplHook<GroupGroupKspaceHook>('group/group');
      hook.init(this.sys);
      this.kspaceTerms = hook.terms(this.sys, this.groupBit, this.group2Bit);
    }
  }

  /** One pass over the counted pairs gives the energy (scalar) and the force on group1 (vector); kspace yes adds the long-range part. */
  private evaluate(): void {
    if (this.cacheEpoch === this.sys.epoch) return;
    const f = new Float64Array(3);
    let e = 0;
    if (this.pairOn) {
      const s = this.sys.state;
      walkTallyPairs(this.sys, this.id, this.groupBit, this.group2Bit, this.molMode, (i, j, evdwl, ecoul, fforce, dx, dy, dz) => {
        e += evdwl + ecoul;
        // As native LAMMPS does (src/compute_group_group.cpp pair_contribution at b332f28): the force of group2 on
        // group1 is added when I is in group1 and J in group2, and subtracted when J is in group1 and I in group2 (a
        // pair with both atoms in both groups adds nothing)
        const mi = s.mask[i], mj = s.mask[atomIndex(this.sys, j)];
        const fx = fforce * dx, fy = fforce * dy, fz = fforce * dz;
        if ((mi & this.groupBit) && (mj & this.group2Bit)) { f[0] += fx; f[1] += fy; f[2] += fz; }
        if ((mj & this.groupBit) && (mi & this.group2Bit)) { f[0] -= fx; f[1] -= fy; f[2] -= fz; }
      });
    }
    if (this.kspaceOn) {
      const hook = gplHook<GroupGroupKspaceHook>('group/group');
      // a compute evaluated before any run's init finds the terms here
      this.kspaceTerms ??= hook.terms(this.sys, this.groupBit, this.group2Bit);
      const k = hook.contribution(this.sys, this.groupBit, this.group2Bit, this.kspaceTerms, this.boundaryOn);
      e += k.e;
      f[0] += k.f[0]; f[1] += k.f[1]; f[2] += k.f[2];
    }
    this.energy = e;
    this.vector.set(f);
    this.cacheEpoch = this.sys.epoch;
  }

  protected computeScalar(): number {
    this.evaluate();
    return this.energy;
  }

  protected computeVector(): void {
    this.evaluate();
  }
}

/**
 * compute pe/tally, force/tally and stress/tally group1-ID style group2-ID. The per-atom values are
 * kept in arrayAtom (pe/tally: 2 columns evdwl, ecoul; force/tally: 3; stress/tally: 6, xx yy zz xy xz yz).
 */
export class ComputePairTally extends Compute {
  readonly style: string;
  private readonly group2Bit: number;
  private readonly cols: number;
  private cacheEpoch = -1;
  private total = 0;

  constructor(sys: System, id: string, group: string, args: string[], style: 'pe/tally' | 'force/tally' | 'stress/tally') {
    super(sys, id, group, args);
    this.style = style;
    // As native LAMMPS does (src/TALLY/compute_pe_tally.cpp, compute_force_tally.cpp, compute_stress_tally.cpp at b332f28):
    // only the group2-ID is read; measured with native LAMMPS, words after it are ignored (force/tally even molecule inter runs)
    if (args.length === 0) throw new StyleError(`Illegal compute ${style} command: missing argument(s)`);
    if (sys.groups.find(args[0]) < 0) throw new StyleError(`Could not find compute ${style} second group ID ${args[0]}`);
    this.group2Bit = sys.groups.bit(args[0]);
    this.cols = style === 'pe/tally' ? 2 : style === 'force/tally' ? 3 : 6;
    this.scalarFlag = true;
    this.peratomFlag = true;
    this.sizePeratomCols = this.cols;
    this.extscalar = style === 'stress/tally' ? 0 : 1;
    this.extvector = 0;
  }

  private evaluate(): void {
    if (this.cacheEpoch === this.sys.epoch) return;
    const s = this.sys.state;
    const n = s.n;
    const arr = new Float64Array(this.cols * n);
    let total = 0;
    const style = this.style;
    walkTallyPairs(this.sys, this.id, this.groupBit, this.group2Bit, 'off', (i, j, evdwl, ecoul, fforce, dx, dy, dz) => {
      const oj = atomIndex(this.sys, j);
      if (style === 'pe/tally') {
        total += evdwl + ecoul;
        const h = 0.5 * (evdwl), hc = 0.5 * ecoul;
        arr[2 * i] += h; arr[2 * i + 1] += hc;
        arr[2 * oj] += h; arr[2 * oj + 1] += hc;
      } else if (style === 'force/tally') {
        // force on i from j, and the opposite force on j
        const fx = fforce * dx, fy = fforce * dy, fz = fforce * dz;
        arr[3 * i] += fx; arr[3 * i + 1] += fy; arr[3 * i + 2] += fz;
        arr[3 * oj] -= fx; arr[3 * oj + 1] -= fy; arr[3 * oj + 2] -= fz;
      } else {
        // pair virial d (x) F with F = fforce d: trace fforce |d|^2, and -1/2 of the outer product per atom
        total += fforce * (dx * dx + dy * dy + dz * dz);
        const xx = -0.5 * fforce * dx * dx, yy = -0.5 * fforce * dy * dy, zz = -0.5 * fforce * dz * dz;
        const xy = -0.5 * fforce * dx * dy, xz = -0.5 * fforce * dx * dz, yz = -0.5 * fforce * dy * dz;
        for (const a of [i, oj]) {
          arr[6 * a] += xx; arr[6 * a + 1] += yy; arr[6 * a + 2] += zz;
          arr[6 * a + 3] += xy; arr[6 * a + 4] += xz; arr[6 * a + 5] += yz;
        }
      }
    });
    if (style === 'force/tally') {
      // scalar: magnitude of the force summed over the atoms of group1
      let fx = 0, fy = 0, fz = 0;
      for (let a = 0; a < n; a++) {
        if (!(s.mask[a] & this.groupBit)) continue;
        fx += arr[3 * a]; fy += arr[3 * a + 1]; fz += arr[3 * a + 2];
      }
      total = Math.sqrt(fx * fx + fy * fy + fz * fz);
    } else if (style === 'stress/tally') {
      // scalar: average of the diagonal of the stress tensor, the pair virial trace over 3
      total = total / 3;
      // per-atom values are in pressure*volume units (as compute stress/atom); the scalar is not converted
      // (measured with native LAMMPS, black box: in units real the per-atom values are nktv2p = 68568.415
      // times the energy-unit sums, while compute stress/tally's scalar equals the energy-unit trace / 3)
      const nktv2p = this.sys.state.units.nktv2p;
      if (nktv2p !== 1) for (let k = 0; k < arr.length; k++) arr[k] *= nktv2p;
    }
    this.arrayAtom = arr;
    this.total = total;
    this.cacheEpoch = this.sys.epoch;
  }

  protected computeScalar(): number {
    warnUntallied(this.sys, this.style === 'pe/tally' ? 'Energy' : 'Stress');
    this.evaluate();
    return this.total;
  }

  protected computePeratom(): void {
    warnUntallied(this.sys, this.style === 'pe/tally' ? 'Energy' : 'Stress');
    this.evaluate();
  }
}
