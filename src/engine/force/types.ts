import { bounds } from './util';
import type { AtomStyle, SimState } from '../types';
import type { Geometry } from '../domain';
import type { Neighbor, NeighList } from '../neighbor';
import type { System } from '../system';
import type { Fix } from '../fix/fix';

/*
 * Contracts for force-field styles (pair, bond, angle, dihedral, improper,
 * kspace). One file per style implements one of these; registry.ts maps the
 * LAMMPS style name to a factory.
 *
 * Energy and virial bookkeeping follows docs.lammps.org/compute_pressure.html:
 * the virial of each atom is the sum W = sum r_i . f_i, summed over pairs
 * as r_ij . F_ij; the 6 components are xx, yy, zz, xy, xz, yz. Energies
 * are split as in docs.lammps.org/thermo_style.html: "evdwl = van der Waals
 * pairwise energy (includes etail)", "ecoul = Coulombic pairwise energy",
 * "elong = long-range kspace energy", "ebond", "eangle", "edihed", "eimp".
 */

/** Errors a style raises for bad input; the interpreter adds the line number. */
export class StyleError extends Error {}

/** What a pair style's constructor checks: the atom style and the unit style when pair_style makes it. */
export interface PairConstructorWhere { atomStyle: AtomStyle; units: string }

/**
 * A bond, angle, dihedral or improper style that takes no settings was given some: the style command reports it in
 * native's words with the style name as given (Bond::settings and friends, src/bond.cpp at b332f28).
 */
export class NoSettingsError extends StyleError {
  constructor(readonly word: string) {
    super(`Illegal style argument: ${word}`);
  }
}

/** A bond style's store/local request: the fix (made by create) that receives its rows. */
export interface LocalStoreRequest {
  id: string;
  freq: number;
  names: string[];
  create(sys: unknown): { style: string; nvalues: number; addRow(row: readonly number[]): void } & { id: string };
}

export interface Accum {
  evdwl: number;
  ecoul: number;
  elong: number;
  ebond: number;
  eangle: number;
  edihed: number;
  eimp: number;
  /** Pair virial, xx yy zz xy xz yz. */
  virial: Float64Array;
  /** Virial of each other contribution (compute pressure keywords select them). */
  vbond: Float64Array;
  vangle: Float64Array;
  vdihed: Float64Array;
  vimp: Float64Array;
  vlong: Float64Array;
}

export const newAccum = (): Accum => ({
  evdwl: 0, ecoul: 0, elong: 0, ebond: 0, eangle: 0, edihed: 0, eimp: 0, virial: new Float64Array(6),
  vbond: new Float64Array(6), vangle: new Float64Array(6), vdihed: new Float64Array(6),
  vimp: new Float64Array(6), vlong: new Float64Array(6),
});

export const clearAccum = (a: Accum): void => {
  a.evdwl = a.ecoul = a.elong = a.ebond = a.eangle = a.edihed = a.eimp = 0;
  for (const v of [a.virial, a.vbond, a.vangle, a.vdihed, a.vimp, a.vlong]) v.fill(0);
};

/** Sum of every force-field virial contribution. */
export const totalVirial = (a: Accum, out: Float64Array): Float64Array => {
  for (let c = 0; c < 6; c++) out[c] = a.virial[c] + a.vbond[c] + a.vangle[c] + a.vdihed[c] + a.vimp[c] + a.vlong[c];
  return out;
};

/** What a pair style sees during compute(). */
export interface PairCompute {
  s: SimState;
  nb: Neighbor;
  geom: Geometry;
  /** Owned + ghost positions, forces, types, charges (nb.xall etc.). */
  x: Float64Array;
  f: Float64Array;
  type: Int32Array;
  q: Float64Array;
  nlocal: number;
  nall: number;
  half: NeighList | null;
  full: NeighList | null;
  /** [1, 1-2, 1-3, 1-4] weights. */
  specialLJ: Float64Array;
  specialCoul: Float64Array;
  /** qqr2e / dielectric. */
  qqrd2e: number;
  acc: Accum;
  /** Per-atom energy (length nall) and virial (6 * nall), or null when not requested. */
  eatom: Float64Array | null;
  vatom: Float64Array | null;
  /** True inside a timestep; false at run setup and between runs (granular contact history is then left alone). */
  historyUpdate?: boolean;
  /**
   * Set by a fdotr style that summed the global virial (sum x.f) into acc.virial itself, in native LAMMPS's atom order,
   * and folded its ghost forces into their owners: the force field then skips its own dot product.
   */
  fdotrDone?: boolean;
}

/**
 * Tallies one pair (i owned or ghost, j owned or ghost; i, j are indices
 * into the owned+ghost arrays): energies are global totals; per-atom
 * energy and virial are split half to each atom.
 */
export const evTally = (
  pc: PairCompute, i: number, j: number, evdwl: number, ecoul: number,
  fpair: number, dx: number, dy: number, dz: number,
): void => {
  const a = pc.acc;
  a.evdwl += evdwl;
  a.ecoul += ecoul;
  const v0 = dx * dx * fpair, v1 = dy * dy * fpair, v2 = dz * dz * fpair;
  const v3 = dx * dy * fpair, v4 = dx * dz * fpair, v5 = dy * dz * fpair;
  const v = a.virial;
  v[0] += v0; v[1] += v1; v[2] += v2; v[3] += v3; v[4] += v4; v[5] += v5;
  if (pc.eatom) {
    const e = 0.5 * (evdwl + ecoul);
    pc.eatom[i] += e; pc.eatom[j] += e;
  }
  if (pc.vatom) {
    const va = pc.vatom;
    for (const k of [i, j]) {
      va[6 * k] += 0.5 * v0; va[6 * k + 1] += 0.5 * v1; va[6 * k + 2] += 0.5 * v2;
      va[6 * k + 3] += 0.5 * v3; va[6 * k + 4] += 0.5 * v4; va[6 * k + 5] += 0.5 * v5;
    }
  }
};

/**
 * The per-atom part of one pair's tally with a force that need not lie along the pair vector (a granular contact):
 * d = x_i - x_j, (fx, fy, fz) the force on i and its opposite on j; half of the energy and of d (x) f go to each atom
 * (only when requested). As native LAMMPS does in Pair::ev_tally_xyz with newton on (src/pair.cpp at b332f28).
 */
export const tallyAtomXyz = (
  pc: PairCompute, i: number, j: number, e: number, fx: number, fy: number, fz: number, dx: number, dy: number, dz: number,
): void => {
  if (pc.eatom) { pc.eatom[i] += 0.5 * e; pc.eatom[j] += 0.5 * e; }
  if (pc.vatom) {
    const va = pc.vatom;
    const v0 = 0.5 * dx * fx, v1 = 0.5 * dy * fy, v2 = 0.5 * dz * fz, v3 = 0.5 * dx * fy, v4 = 0.5 * dx * fz, v5 = 0.5 * dy * fz;
    va[6 * i] += v0; va[6 * i + 1] += v1; va[6 * i + 2] += v2; va[6 * i + 3] += v3; va[6 * i + 4] += v4; va[6 * i + 5] += v5;
    va[6 * j] += v0; va[6 * j + 1] += v1; va[6 * j + 2] += v2; va[6 * j + 3] += v3; va[6 * j + 4] += v4; va[6 * j + 5] += v5;
  }
};

/**
 * Tallies one ordered pair of a full neighbor list whose force goes to the owned atom i only, as native LAMMPS does in
 * Pair::ev_tally_full (src/pair.cpp at b332f28): half of the energies to the totals and to atom i, half of
 * del (x) del fpair to the global virial and to atom i (the style then has no fdotr virial).
 */
export const evTallyFull = (
  pc: PairCompute, i: number, evdwl: number, ecoul: number, fpair: number, dx: number, dy: number, dz: number,
): void => {
  const a = pc.acc;
  a.evdwl += 0.5 * evdwl;
  a.ecoul += 0.5 * ecoul;
  if (pc.eatom) pc.eatom[i] += 0.5 * (evdwl + ecoul);
  const h = 0.5 * fpair;
  const v0 = dx * dx * h, v1 = dy * dy * h, v2 = dz * dz * h, v3 = dx * dy * h, v4 = dx * dz * h, v5 = dy * dz * h;
  const v = a.virial;
  v[0] += v0; v[1] += v1; v[2] += v2; v[3] += v3; v[4] += v4; v[5] += v5;
  if (pc.vatom) {
    const va = pc.vatom;
    va[6 * i] += v0; va[6 * i + 1] += v1; va[6 * i + 2] += v2; va[6 * i + 3] += v3; va[6 * i + 4] += v4; va[6 * i + 5] += v5;
  }
};

/**
 * Parses a type argument: "n", "*", "*n", "n*", "m*n" (docs.lammps.org/pair_coeff.html), as native LAMMPS does
 * (utils::bounds at b332f28, with its messages: "Invalid range string: ...", "Numeric index ... is out of bounds").
 */
export const typeBounds = (word: string, n: number): [number, number] => bounds(word, 1, n);

/**
 * A pair_coeff line of the wrong shape, with native LAMMPS's message (by default "Incorrect args for pair coefficients"
 * and its error page; a few styles have their own `text` without the page); the engine's usage of the style follows
 * as a hint.
 */
/**
 * A bond_coeff, angle_coeff, dihedral_coeff or improper_coeff line of the wrong shape, with native LAMMPS's message
 * ("Incorrect args for bond coefficients" and its error page); the engine's usage of the style follows as a hint.
 */
export const bondedCoeffArgs = (kind: string, usage?: string): StyleError =>
  new StyleError(`Incorrect args for ${kind} coefficients\nFor more information see https://docs.lammps.org/err0021${usage ? `\n${usage}` : ''}`);

export const pairCoeffArgs = (usage?: string, text = 'Incorrect args for pair coefficients', page = true): StyleError =>
  new StyleError(`${text}${page ? '\nFor more information see https://docs.lammps.org/err0021' : ''}${usage ? `\n${usage}` : ''}`);

/**
 * Per type pair parameters, symmetric. "For the asterisk syntax, only type
 * pairs with I <= J are considered; if asterisks imply type pairs where
 * J < I, they are ignored. Again internally, LAMMPS will set the
 * coefficients for the symmetric J,I interactions to the same values"
 * (pair_coeff.html).
 */
export class PairParams {
  readonly nt: number;
  private data: Float64Array[];
  /** 1 = set explicitly by pair_coeff (or the data file). */
  readonly explicit: Uint8Array;

  constructor(readonly ntypes: number, readonly names: readonly string[]) {
    this.nt = ntypes + 1;
    this.data = names.map(() => new Float64Array(this.nt * this.nt));
    this.explicit = new Uint8Array(this.nt * this.nt);
  }

  idx(i: number, j: number): number { return i * this.nt + j; }
  p(name: string): Float64Array {
    const k = this.names.indexOf(name);
    if (k < 0) throw new Error(`no pair parameter ${name}`);
    return this.data[k];
  }

  /** Sets values for every I <= J in the wildcard ranges; returns the count. */
  setRange(iw: string, jw: string, values: readonly number[]): number {
    const [ilo, ihi] = typeBounds(iw, this.ntypes);
    const [jlo, jhi] = typeBounds(jw, this.ntypes);
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.set(i, j, values);
        count++;
      }
    }
    // "I J" given as e.g. "2 1" without wildcards still means the pair (1, 2)
    if (count === 0 && ilo === ihi && jlo === jhi) { this.set(jlo, ilo, values); count = 1; }
    if (count === 0) throw new StyleError(`pair coefficients: no type pairs with I <= J in ${iw} ${jw}`);
    return count;
  }

  set(i: number, j: number, values: readonly number[]): void {
    for (let k = 0; k < this.names.length; k++) {
      const v = values[k] ?? Number.NaN;
      this.data[k][this.idx(i, j)] = v;
      this.data[k][this.idx(j, i)] = v;
    }
    this.explicit[this.idx(i, j)] = 1;
    this.explicit[this.idx(j, i)] = 1;
  }

  /** Writes a mixed (not explicit) value for both orders. */
  setMixed(i: number, j: number, name: string, v: number): void {
    const a = this.p(name);
    a[this.idx(i, j)] = v;
    a[this.idx(j, i)] = v;
  }

  isSet(i: number, j: number): boolean { return this.explicit[this.idx(i, j)] === 1; }
  get(name: string, i: number, j: number): number { return this.p(name)[this.idx(i, j)]; }
}

export type MixRule = 'geometric' | 'arithmetic' | 'sixthpower';

/** pair_modify mix formulas for epsilon and sigma (the cutoff mixes like sigma). */
export const mixEpsilon = (rule: MixRule, e1: number, e2: number, s1: number, s2: number): number => {
  if (rule === 'sixthpower') {
    const s13 = s1 ** 3, s23 = s2 ** 3;
    return (2 * Math.sqrt(e1 * e2) * s13 * s23) / (s13 * s13 + s23 * s23);
  }
  return Math.sqrt(e1 * e2);
};

export const mixDistance = (rule: MixRule, s1: number, s2: number): number => {
  if (rule === 'geometric') return Math.sqrt(s1 * s2);
  if (rule === 'arithmetic') return 0.5 * (s1 + s2);
  return Math.pow(0.5 * (s1 ** 6 + s2 ** 6), 1 / 6);
};

export interface StyleContext {
  /** The system, or null when a style is defined before the box exists. */
  s: SimState | null;
  /**
   * Reads a file the notebook has (uploads or files written by the session). A missing file is an error starting
   * "{opener} {name}: No such file or directory" (opener default "Cannot open file"; native's readers name the kind of
   * file, as TextFileReader's "cannot open ReaxFF control file").
   */
  readFile(name: string, opener?: string): string;
  log(text: string): void;
  /** comm_modify vel yes: ghost atoms carry velocities (granular pair styles require it). */
  ghostVelocity?: boolean;
  /** Group bit of a fix freeze (0 when none): granular contacts with a frozen particle use the other one's mass. */
  freezeGroupBit?: number;
  /** newton pair setting (default on); a few styles require it off. */
  newtonPair?: boolean;
  /** Value of an equal-style variable (lepton expressions with v_name references). */
  equalVariable?(name: string): number;
  /** The unit style of the units command (set before the box exists too). */
  unitStyle?: string;
  /** An atom type word with a type label replaced by its numeric type (labelmap.html); other words pass through. */
  atomTypeWord?(word: string): string;
  /** Lattice spacing in x of the lattice command (1 when none is defined): peri styles take half of it as the node radius. */
  xlattice?: number;
  /**
   * Adds a fix the style needs at pair_style time (pair tracker's fix STORE/LOCAL, which the style's own
   * keywords name); a fix with the same ID is replaced. make receives the system the fix joins.
   */
  addFix?(make: (sys: System) => Fix): void;
  /**
   * The fix of a style defined in the system (native Modify::get_fix_by_style, first match), or null. A pair style
   * that needs a fix at pair_coeff time (pair exp6/rx needs fix rx) reads its data through it.
   */
  fixOf?(style: string): Fix | null;
  /** The fix with this ID (null when there is none). */
  fixById?(id: string): Fix | null;
  /**
   * The amount by which create_box moved each shrink-wrapped (s) face out (System.initSmall), while the box has not
   * been reset since; null otherwise. Until that first reset native LAMMPS's box includes the expansion (src/domain.cpp
   * set_initial_box at b332f28), and a style that reads the box at init (kspace_style msm) sees those lengths.
   */
  initialBoxExpansion?: readonly [number, number, number] | null;
  /**
   * The begin and end steps of the current run (native update->beginstep and endstep: the run's start and stop values),
   * for styles that ramp a value over a run (pair_style dpd/tstat with t_start != t_stop).
   */
  runSteps?(): { begin: number; end: number };
}

/**
 * Equilibrium bond lengths and angles by type, for Pair.linkBonded: the angle in radians, as native
 * Angle::equilibrium_angle gives it (Bonded.equilibriumRadians).
 */
export interface BondedEquilibria {
  bond(type: number): number;
  angle(type: number): number;
}

/** Base for pair styles. */
/**
 * Per-atom minimizer degrees of freedom a pair style adds. pull(s, x, f) writes the minimizer's coordinate x and its
 * force f for every owned atom from the current state (called after each force evaluation); push(s, x) writes the
 * state from the coordinates (called before each energy evaluation of a line search trial). maxStep caps the change
 * of any coordinate in one iteration.
 */
export interface PairMinPerAtom {
  readonly maxStep: number;
  pull(s: SimState, x: Float64Array, f: Float64Array): void;
  push(s: SimState, x: Float64Array): void;
}

/** The separation vector x_i - x_j of a pair (Pair.single's del argument). */
export interface PairSep { dx: number; dy: number; dz: number }

export abstract class Pair {
  abstract readonly name: string;
  ntypes = 0;
  /** pair_modify settings: "mix = geometric, shift = no, ... tail = no". */
  mix: MixRule = 'geometric';
  shift = false;
  tail = false;
  /** pair_modify table N (coul/long): 0 = direct evaluation. */
  table = 12;
  /** Many-body styles ignore special_bonds and need full lists. */
  manybody = false;
  /** Set by pair_style hybrid/overlay on its sub-styles (native "^hybrid/overlay" checks; false when the style is alone). */
  hybridOverlay = false;
  /**
   * Set by pair_style hybrid, hybrid/overlay and hybrid/scaled on their sub-styles. A style that folds its ghost forces
   * and sums its own fdotr virial (PairCompute.fdotrDone) does so only when it is the whole pair style, since a
   * sub-style shares the forces and the virial sum with the others.
   */
  hybridSub = false;
  needsFull = false;
  needsHalf = true;
  /**
   * The neighbor list keeps a pair only within the two radii plus the skin. As native LAMMPS does (src/npair_bin.cpp at
   * b332f28) for the size lists that finite-size styles request (NeighConst::REQ_SIZE): "radsum = radius[i] +
   * radius[j]; cut = radsum + skin; cutsq = cut * cut;" and the pair is stored "if (rsq <= cutsq)". The pairs left out
   * are not in contact, so forces are unchanged; with a wide size distribution the list is far shorter (the type cutoff
   * covers the largest pair of radii).
   */
  sizeList = false;
  /** Styles with a long-range Coulomb part (need a kspace style). */
  coulLong = false;
  /**
   * The style asks for a k-space solver with a dispersion part (native Pair::dispersionflag). As native LAMMPS does
   * (src/kspace.cpp KSpace::pair_check at b332f28): a kspace style whose own dispersion flag (KSpace.dispersion)
   * differs from the pair's is refused, "KSpace style X is incompatible with Pair style Y".
   */
  dispersionFlag = false;
  /**
   * Long-range solver flags that the style's own settings keywords set (native pair_style keywords "ewald", "pppm",
   * "msm", e.g. pair_lepton_coul.cpp settings). Read by kspaceCheck.ts; a style without such keywords leaves it empty.
   */
  solverKeywords: readonly string[] = [];
  /**
   * Keep pairs whose special_bonds weights are both 0.0 in the neighbor list
   * (special_bonds.html: "a value of 0.0 means exclude the pair completely
   * from the neighbor list, except for pair styles that require a kspace
   * style and pair styles amoeba, hippo, thole, coul/exclude, and pair styles
   * that include “coul/dsf” or “coul/wolf”.").
   */
  keepExcluded = false;
  /**
   * Factor on the Coulomb conversion constant the whole force field uses (pair and kspace); set in
   * init(). pair_charmm.html: "The newest CHARMM pair styles reset the Coulombic energy conversion
   * factor used internally in the code, from the LAMMPS value to the CHARMM value, as if it were
   * effectively a parameter of the force field." Those styles set CHARMM / LAMMPS in units real.
   */
  coulConstScale = 1;
  /**
   * Bumped whenever the style's coefficients may have changed (ForceField.init, fix adapt), so
   * copies of the style held by force threads (cpu/pairThreads.ts) are refreshed.
   */
  version = 0;
  /**
   * Force-field hook, called before init(): the kspace style (null when none). A pair style that needs a particular
   * solver (the coul/msm styles read the splitting function of kspace_style msm) keeps it and refuses others.
   */
  linkKSpace?(kspace: KSpace | null): void;
  /**
   * Force-field hook, called before init(): the equilibrium length of each bond type and angle
   * (degrees) of each angle type, NaN without a bond or angle style. The TIP4P styles place
   * their massless charge site from them.
   */
  linkBonded?(link: BondedEquilibria): void;
  /**
   * The style does not tally the global virial itself; the force field takes
   * it as sum_k x_k . f_k over owned and ghost atoms right after compute()
   * (Developer_flow.html: "the global virial ... to be calculated cheaply (at
   * O(N) cost instead of O(N**2) at the end of the Pair::compute() method), by
   * a dot product of atom coordinates and forces. By including owned and ghost
   * atoms in the dot product, the effect of periodic boundary conditions is
   * correctly accounted for.").
   */
  virialFdotr = false;
  /**
   * The style tallies its global virial into pc.acc.virial itself, pair by pair (as with newton off), and is not covered
   * by the fdotr sum; a hybrid pair then skips its own fdotr sum (hybrid.ts).
   */
  explicitVirial = false;
  /**
   * The style's pair-by-pair virial differs from x.f over its own forces (a force applied away from the atom pair,
   * such as a sphere-surface contact counted from the contact point). In a hybrid with an explicit sub-style, native
   * LAMMPS tallies every sub-style pair by pair (src/pair_hybrid.cpp compute at b332f28: "no_virial_fdotr_compute"),
   * so the hybrid sets explicitVirial on such a style (hybrid.ts).
   */
  explicitInHybrid = false;
  /** G-ewald splitting parameter, set from the kspace style at init (coul/long styles). */
  gEwald = 0;
  /** Cutoff per type pair after init: (ntypes+1)^2. */
  cut = new Float64Array(0);
  cutsq = new Float64Array(0);
  /** Long-range tail corrections (energy*V and pressure*V*V... see tailSums). */
  etail = 0;
  ptail = 0;
  /** pair_modify compute yes|no (pair_modify.html): false skips the pairwise term (energy, force, virial). */
  computeOn = true;
  /** Squared inner cutoff of the erfc tables (pair_modify tabinner squared; default sqrt(2)^2 = 2). */
  tabinnerRsq = 2;

  /**
   * The checks of the native style's constructor, which run before its settings() when pair_style makes the style
   * (atom attributes, unit style); the pair_style command calls it with the atom style and the unit style.
   */
  constructorCheck?(where: PairConstructorWhere): void;
  /** pair_style arguments. */
  abstract settings(args: string[], ctx: StyleContext): void;
  /** pair_coeff arguments including I and J. */
  abstract coeff(args: string[], ctx: StyleContext): void;
  /** Called by create_box / read_data once the number of types is known. */
  allocate(ntypes: number): void { this.ntypes = ntypes; }
  /**
   * Per type pair setup after mixing (i <= j); returns the cutoff.
   * Throws a StyleError for a pair that is not set and cannot be mixed.
   */
  abstract initOne(i: number, j: number): number;
  /** Optional style-wide setup before initOne (e.g. reading tables). */
  initStyle(_ctx: StyleContext): void {}

  init(ctx: StyleContext): void {
    // pair_modify.html: "You cannot use shift yes with tail yes, since those are conflicting
    // options. You cannot use tail yes with 2d simulations."
    if (this.shift && this.tail) throw new StyleError('Cannot have both pair_modify shift and tail set to yes');
    if (this.tail && ctx.s?.dimension === 2) throw new StyleError('cannot use pair_modify tail yes with 2d simulations');
    this.initStyle(ctx);
    const nt = this.ntypes + 1;
    this.cut = new Float64Array(nt * nt);
    this.cutsq = new Float64Array(nt * nt);
    this.etail = 0;
    this.ptail = 0;
    for (let i = 1; i < nt; i++) {
      for (let j = i; j < nt; j++) {
        const c = this.initOne(i, j);
        this.cut[i * nt + j] = this.cut[j * nt + i] = c;
        this.cutsq[i * nt + j] = this.cutsq[j * nt + i] = c * c;
      }
    }
  }

  /**
   * Per-atom minimizer degrees of freedom of this style (one value per owned atom), or
   * undefined when the style adds none. The minimizer appends them after the box DOF in cg and sd.
   */
  minPerAtom?(): PairMinPerAtom;
  /**
   * True when pair_coeff args are the style's own form rather than "I J" atom types (pair_eff_cut.cpp: an
   * ECP line "I s A B C" or "I p A B C D E" keeps its words). The command then passes the words unchanged.
   */
  rawCoeff?(args: string[]): boolean;
  /**
   * True when the style's last pair_coeff set the type pair I J (native setflag[I][J]). A hybrid style assigns a
   * sub-style with "pair_coeff * *" only to the type pairs it sets (pair_hybrid.cpp PairHybrid::coeff). Styles that
   * set every pair of the range do not define it.
   */
  setsPair?(i: number, j: number): boolean;
  /**
   * pair_modify shift/tail on a style that has no such term. The style's docs say it "does not support"
   * the option (pair_morse.rst, pair_soft.rst, pair_yukawa.rst, pair_lj_cubic.rst, pair_lj_smooth.rst,
   * pair_lj_smooth_linear.rst, pair_eam.rst, pair_sw.rst, pair_tersoff.rst); native LAMMPS accepts it and
   * gives the unshifted / untailed result (measured for yukawa, morse, lj/cubic, lj/smooth/linear, soft, eam).
   * The engine does the same and logs one line per init saying the option has no effect for this style.
   */
  noTermLog(ctx: StyleContext, key: 'shift' | 'tail'): void {
    if (this[key]) ctx.log(`pair_modify ${key} yes has no effect for pair style ${this.name} (native LAMMPS ignores it too)`);
  }

  /** Long-range tail energy/pressure need per-type counts; override when tail is supported. */
  tailSums(_counts: Float64Array): { etail: number; ptail: number } { return { etail: 0, ptail: 0 }; }

  abstract compute(pc: PairCompute): void;

  /** Lines of the "Pair Coeffs" section (one per type), or null if not writable. */
  dataCoeffs(): string[] | null { return null; }
  /** Lines of the "PairIJ Coeffs" section, or null. */
  dataCoeffsIJ(): string[] | null { return null; }

  /**
   * Energy and force of one pair (for pair_write and compute pair/local); optional. i and j are owned-atom indices
   * (a ghost is passed as its owner). del = x_i - x_j, for styles whose extra quantities need the separation vector.
   * fforce is the force divided by the distance (native Pair::single fforce): compute pair/local multiplies it by the
   * separation components for fx, fy, fz. svector: the pN values (native svector), index N-1 for pN; only a style with
   * singleExtra() > 0 returns it.
   */
  single?(i: number, j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number, del?: PairSep): { eng: number; fforce: number; svector?: Float64Array };

  /**
   * Native single_extra: the number of pN values (compute pair/local p1 .. pN) that single() reports. 0 for a style that
   * defines none; the engine refuses pN for such a style, naming it. docs.lammps.org/compute_pair_local.html: "A pair style
   * may define additional pairwise quantities which can be accessed as *p1* to *pN*, where :math:`N` is defined by the
   * pair style. Most pair styles do not define any additional quantities, so :math:`N = 0`."
   */
  singleExtra(): number { return 0; }

  /**
   * True when single() needs the separation vector del = x_i - x_j (the granular and ellipsoid styles: the contact normal
   * and the shear are defined by it). The callers that have no separation (widom's probe sum, tally, stress/prof, hbond/local
   * energies) treat such a style as one without single(), so they refuse or fall back as they do for that case.
   */
  singleNeedsSeparation(): boolean { return false; }

  /** Named values for other styles (e.g. 'cut_coul' for kspace). */
  extract(_name: string): unknown { return undefined; }
}

/**
 * Ghost copies of owned atoms: the ghosts of owned atom i are the indices
 * start[i] .. start[i+1]-1 (absolute, after the owned atoms); ghost k is the
 * image (image[3k], image[3k+1], image[3k+2]) of its owner, in lattice units.
 */
export interface BondedCopies {
  start: Int32Array;
  image: Int32Array;
}

/** What a bonded style sees during compute(). Positions are owned atoms only. */
/**
 * Bond type a bond style sets on a bond it deletes (native: the bond is removed from the atoms' lists, as bond rheo/shell's
 * process_ineligibility does). The force field drops those entries after the bond compute (forcefield.ts).
 */
export const BOND_REMOVED = -1;

export interface BondedCompute {
  s: SimState;
  geom: Geometry;
  /**
   * The neighbor-list build count (Neighbor.nbuild) at this evaluation. A bond style that keeps per-bond history
   * copies it at each rebuild, as native LAMMPS does in fix bond/history pre_exchange (bond_bpm_rotational.ts).
   */
  nbuild?: number;
  /** id -> index. */
  map: Int32Array;
  f: Float64Array;
  acc: Accum;
  /** This style's virial accumulator (acc.vbond, acc.vangle, ...). */
  virial: Float64Array;
  eatom: Float64Array | null;
  vatom: Float64Array | null;
  /**
   * The per-atom centroid virial of this term (compute centroid/stress/atom), 9 values per atom (xx yy zz xy xz yz
   * yx zx zy), or null. A term over m atoms gives each atom (r_a - r_0) (x) f_a with r_0 the term's centroid, as
   * native Angle/Dihedral/Improper::ev_tally do with cvflag_atom; a pairwise part gives the per-atom virial halves.
   */
  cvatom?: Float64Array | null;
  /**
   * Per-atom energy and virial of the pairwise parts a bonded style tallies through the pair style (the CHARMM 1-4
   * pair, the 1-3 pair of angle cosine/buck6d), as native does with force->pair->ev_tally: they belong to the pair
   * term of pe/atom and stress/atom (and to its centroid virial). Null or absent: the style's own arrays.
   */
  pairEatom?: Float64Array | null;
  pairVatom?: Float64Array | null;
  /**
   * The periodic copies of the owned atoms within the ghost cutoff (from the
   * neighbor build). Bonded terms take the copy nearest to their owning atom
   * (bonded_util.ts `delta`); without this table the minimum image is used.
   */
  copies?: BondedCopies | null;
  /** Emits a "WARNING: ..." line in the run log (e.g. FENE bond too long). */
  warn?: (text: string) => void;
  /** False during setup (run start, run 0, between runs); true inside a timestep (bpm bonds break only then). */
  historyUpdate?: boolean;
}

/**
 * Tallies one bonded term: energy e on `atoms` (indices), split evenly for
 * per-atom energy; virial from the force on each atom times its position
 * relative to the first atom (r_k - r_0) . f_k, which equals the
 * pairwise-sum form for internal forces that sum to zero.
 */
/** Adds a (x) f to atom a's centroid virial (9 values: xx yy zz xy xz yz yx zx zy). */
export const addCentroid = (cv: Float64Array, a: number, ax: number, ay: number, az: number, fx: number, fy: number, fz: number): void => {
  const o = 9 * a;
  cv[o] += ax * fx; cv[o + 1] += ay * fy; cv[o + 2] += az * fz;
  cv[o + 3] += ax * fy; cv[o + 4] += ax * fz; cv[o + 5] += ay * fz;
  cv[o + 6] += ay * fx; cv[o + 7] += az * fx; cv[o + 8] += az * fy;
};

/**
 * A pairwise part of a bonded term (CHARMM 1-4, an angle's 1-3 pair): half of its 6-value virial w to each of i and j,
 * as a pair style's per-atom virial (native CENTROID_SAME: yx zx zy copy xy xz yz).
 */
export const addPairCentroid = (cv: Float64Array, i: number, j: number, w: readonly number[]): void => {
  for (const a of [i, j]) {
    const o = 9 * a;
    for (let c = 0; c < 6; c++) cv[o + c] += 0.5 * w[c];
    cv[o + 6] += 0.5 * w[3]; cv[o + 7] += 0.5 * w[4]; cv[o + 8] += 0.5 * w[5];
  }
};

/** The centroid virial of an m-body term: rel (3 per atom, any common origin) and the forces fk on the atoms. */
export const termCentroid = (cv: Float64Array, atoms: readonly number[], rel: readonly number[], fk: readonly number[]): void => {
  const m = atoms.length;
  let cx = 0, cy = 0, cz = 0;
  for (let k = 0; k < m; k++) { cx += rel[3 * k]; cy += rel[3 * k + 1]; cz += rel[3 * k + 2]; }
  cx /= m; cy /= m; cz /= m;
  for (let k = 0; k < m; k++) {
    addCentroid(cv, atoms[k], rel[3 * k] - cx, rel[3 * k + 1] - cy, rel[3 * k + 2] - cz, fk[3 * k], fk[3 * k + 1], fk[3 * k + 2]);
  }
};

export const bondedVirial = (
  bc: BondedCompute, atoms: readonly number[], rel: readonly number[], fk: readonly number[], e: number,
): void => {
  // rel and fk: 3 values per atom, rel relative to atom 0 (the copies used for the forces)
  const v = bc.virial;
  const w = [0, 0, 0, 0, 0, 0];
  for (let k = 0; k < atoms.length; k++) {
    const rx = rel[3 * k], ry = rel[3 * k + 1], rz = rel[3 * k + 2];
    const fx = fk[3 * k], fy = fk[3 * k + 1], fz = fk[3 * k + 2];
    w[0] += rx * fx; w[1] += ry * fy; w[2] += rz * fz;
    w[3] += rx * fy; w[4] += rx * fz; w[5] += ry * fz;
  }
  for (let c = 0; c < 6; c++) v[c] += w[c];
  const m = atoms.length;
  if (bc.eatom) for (const i of atoms) bc.eatom[i] += e / m;
  if (bc.vatom) for (const i of atoms) for (let c = 0; c < 6; c++) bc.vatom[6 * i + c] += w[c] / m;
  if (bc.cvatom) termCentroid(bc.cvatom, atoms, rel, fk);
};

/** Base for bond, angle, dihedral and improper styles. */
export abstract class Bonded {
  abstract readonly name: string;
  abstract readonly kind: 'bond' | 'angle' | 'dihedral' | 'improper';
  /**
   * True when this style's per-atom centroid virial is native's even though it does not sum to the per-atom virial:
   * native tallies some of its terms without the centroid virial (angle_amoeba.cpp uses Angle::ev_tally4 and ev_tally2,
   * which leave cvatom alone, angle.cpp at b332f28). compute centroid/stress/atom skips its sum check for such a style
   * only; every other style must have a centroid sum equal to its virial.
   */
  readonly centroidNative?: boolean;
  ntypes = 0;
  abstract settings(args: string[], ctx: StyleContext): void;
  /** *_coeff arguments including the type. */
  abstract coeff(args: string[], ctx: StyleContext): void;
  allocate(ntypes: number): void { this.ntypes = ntypes; }
  /** Throws a StyleError if a type has no coefficients. */
  abstract init(ctx: StyleContext): void;
  /**
   * Styles that compute pair-like terms themselves (dihedral charmm / charmmfsw add the 1-4 LJ and
   * Coulomb) get the pair style and the special_bonds weights before init().
   */
  linkForceField?(link: { pair: Pair | null; special: { lj: readonly number[]; coul: readonly number[] } }): void;
  abstract compute(bc: BondedCompute): void;
  /** Bonds broken by the last compute() (bond_style bpm/spring): true once, then cleared. */
  takeTopologyChange?(): boolean;
  /**
   * bond_style bpm/* keyword store/local ID N values: the fix store/local ID the style writes its broken-bond rows to
   * (bond_bpm.cpp settings). bond_style creates the fix from this request and hands the style its addRow sink.
   */
  localStoreRequest?(): LocalStoreRequest | null;
  attachLocalSink?(sink: (row: number[]) => void): void;
  /**
   * Bond types whose bonds are not 1-2 neighbors for the special lists (bond_rheo_shell.cpp sets ignore_special_flag:
   * the bonds of such a type do not switch off the pair interaction between their atoms). Default: none.
   */
  ignoresSpecial?(type: number): boolean;
  /** Lines of the "<Kind> Coeffs" data-file section, or null. */
  dataCoeffs(): string[] | null { return null; }
  /**
   * Extra data-file sections written right after this style's Coeffs section (write_data), e.g. the
   * class 2 cross terms; read_data reads them back (output/data.ts CLASS2_SECTIONS).
   */
  dataCrossSections(): { title: string; lines: string[] }[] { return []; }
  /** Equilibrium length (bond) or angle in degrees (angle), for fix shake. Angle styles derive it from equilibriumRadians. */
  equilibrium(_type: number): number { return Number.NaN; }
  /**
   * The equilibrium angle in radians, the value native Angle::equilibrium_angle returns (fix shake, fix ilves, fix
   * filter/corotate, fix efield/tip4p, angle_write). NaN where the style has none (bond styles, and the default).
   */
  equilibriumRadians(_type: number): number { return Number.NaN; }
}

/** What a kspace style sees during compute(). */
export interface KSpaceCompute {
  s: SimState;
  geom: Geometry;
  f: Float64Array;
  /** Per-atom torque (3N), accumulated by styles with point dipoles (ewald/dipole); null otherwise. */
  torque?: Float64Array | null;
  qqrd2e: number;
  acc: Accum;
  eatom: Float64Array | null;
  vatom: Float64Array | null;
}

export abstract class KSpace {
  abstract readonly name: string;
  /** Relative accuracy target from kspace_style. */
  accuracy = 1e-4;
  /**
   * The factor of the Coulomb kspace terms (native KSpace::scale, 1 by default; fix adapt and fix adapt/fep set it
   * through extract("scale")). As native LAMMPS does (e.g. src/KSPACE/pppm.cpp at b332f28: "const double qscale =
   * qqrd2e * scale"), it multiplies qqrd2e for the kspace forces, energy and virial, so the force field passes the
   * product to compute().
   */
  scale = 1.0;
  gEwald = 0;
  /** kspace_modify compute: false skips the kspace force, energy and virial (setup still runs; kspace_modify.html). */
  computeOn = true;
  /**
   * The solver splits its work between an Rspace and a Kspace partition (native KSpace::rk_flag, kspace_style pppm/rk;
   * kspace_style.html: "The *pppm/rk* kspace style must be paired with an analogous *rk* type run style").
   */
  rkFlag = false;
  /**
   * When true, the charge sums of the self and background energies keep the values of the last refresh instead of
   * following the charges at each compute. A fix that changes charges without changing the atom count sets it (the
   * browser port of fix bond/react does). Measured with native LAMMPS (black box): after a reaction changes the
   * charges and deletes no atom, the elong self term keeps the sum of squares of the earlier charges: 7.10488 was
   * kept where the refreshed sum is 6.97503, and the energy is 5.69 kcal/mol lower than the refreshed one.
   */
  chargeSumsFrozen = false;
  /**
   * When true, ForceField.init keeps the parameters chosen at the last full init (G-ewald, kmax, mesh) and only
   * refreshes the charge sums (updateChargeSums). A fix that changes charges between MD steps sets it while it runs
   * (fix charge/regulation). Measured with native LAMMPS (black box): its trial moves change the charges and the
   * atom count, and the Ewald or PPPM parameters stay those of the run's init (native KSpace::qsum_qsq).
   */
  paramsFrozen = false;
  /** Recomputes qsum and qsqsum (and the atom count) from the current charges (native KSpace::qsum_qsq). */
  updateChargeSums(_s: SimState): void { /* styles without charge sums need nothing */ }
  abstract settings(args: string[], ctx: StyleContext): void;
  /** kspace_modify keyword handler; returns the number of values consumed. */
  abstract modify(key: string, values: string[]): number;
  /** Sets up parameters for the current box and pair cutoff. */
  abstract init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void;
  abstract compute(kc: KSpaceCompute): void;
  /** Force-field hook, called before init(): the pair style (pppm/tip4p reads its charge sites). */
  linkPair?(pair: Pair | null): void;
  /**
   * The solver has a dispersion part (native KSpace::dispersionflag), checked against Pair.dispersionFlag before init.
   * Called after linkPair: a solver whose dispersion terms follow the pair (ewald/disp) reads them from it.
   */
  dispersion(): boolean { return false; }
  /**
   * The solver has a dipole part (native KSpace::dipoleflag), checked against Pair dipole support by kspaceCheck.ts.
   * As for dispersion(), a solver whose dipole terms follow the pair (ewald/disp) reads them from the linked pair.
   */
  dipole(): boolean { return false; }
  /**
   * kspace_modify pressure/scalar (MSM). kspace_modify.html: "If this option is turned on, only the scalar pressure
   * (i.e. (Pxx + Pyy + Pzz)/3.0) will be computed" and "This option cannot be used to access individual components of
   * the pressure tensor, to compute per-atom virial". compute pressure refuses its tensor while it is set.
   */
  scalarPressure = false;
  /**
   * Force-field hook, called before the pair style's init() (and before the neighbor list is set up): a solver that
   * sets the pair style's Coulomb cutoff does it here, so the pair cutoffs and the neighbor list are built with the
   * new value. kspace_modify.html: "The *cutoff/adjust* keyword applies only to MSM. If this option is turned on, the
   * Coulombic cutoff will be automatically adjusted at the beginning of the run to give the desired estimated error."
   */
  preparePair?(pair: Pair | null, s: SimState, geom: Geometry, qqrd2e: number, ctx: StyleContext): void;
}
