import type { SimState } from '../types';
import type { Geometry } from '../domain';
import type { Neighbor, NeighList } from '../neighbor';

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

/** Parses a type argument: "n", "*", "*n", "n*", "m*n" (docs.lammps.org/pair_coeff.html). */
export const typeBounds = (word: string, n: number): [number, number] => {
  const bad = () => new StyleError(`invalid type '${word}' (expected 1..${n}, or a wildcard *, *n, n*, m*n)`);
  if (word === '*') return [1, n];
  const star = word.indexOf('*');
  const num = (w: string) => {
    if (!/^\d+$/.test(w)) throw bad();
    return Number(w);
  };
  let lo: number, hi: number;
  if (star < 0) lo = hi = num(word);
  else {
    const a = word.slice(0, star), b = word.slice(star + 1);
    lo = a === '' ? 1 : num(a);
    hi = b === '' ? n : num(b);
  }
  if (lo < 1 || hi > n || lo > hi) throw bad();
  return [lo, hi];
};

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
  /** Reads a file the notebook has (uploads or files written by the session). */
  readFile(name: string): string;
  log(text: string): void;
  /** comm_modify vel yes: ghost atoms carry velocities (granular pair styles require it). */
  ghostVelocity?: boolean;
  /** Group bit of a fix freeze (0 when none): granular contacts with a frozen particle use the other one's mass. */
  freezeGroupBit?: number;
  /** newton pair setting (default on); a few styles require it off. */
  newtonPair?: boolean;
  /** Value of an equal-style variable (lepton expressions with v_name references). */
  equalVariable?(name: string): number;
  /** Lattice spacing in x of the lattice command (1 when none is defined): peri styles take half of it as the node radius. */
  xlattice?: number;
}

/** Equilibrium bond lengths and angles (degrees) by type, for Pair.linkBonded. */
export interface BondedEquilibria {
  bond(type: number): number;
  angle(type: number): number;
}

/** Base for pair styles. */
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
  needsFull = false;
  needsHalf = true;
  /** Styles with a long-range Coulomb part (need a kspace style). */
  coulLong = false;
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
  /** G-ewald splitting parameter, set from the kspace style at init (coul/long styles). */
  gEwald = 0;
  /** Cutoff per type pair after init: (ntypes+1)^2. */
  cut = new Float64Array(0);
  cutsq = new Float64Array(0);
  /** Long-range tail corrections (energy*V and pressure*V*V... see tailSums). */
  etail = 0;
  ptail = 0;

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
    if (this.shift && this.tail) throw new StyleError('cannot have both pair_modify shift and tail set to yes');
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

  /** Long-range tail energy/pressure need per-type counts; override when tail is supported. */
  tailSums(_counts: Float64Array): { etail: number; ptail: number } { return { etail: 0, ptail: 0 }; }

  abstract compute(pc: PairCompute): void;

  /** Lines of the "Pair Coeffs" section (one per type), or null if not writable. */
  dataCoeffs(): string[] | null { return null; }
  /** Lines of the "PairIJ Coeffs" section, or null. */
  dataCoeffsIJ(): string[] | null { return null; }

  /** pair_modify keywords the style accepts beyond the common ones. */
  modify(key: string, _values: string[]): number {
    throw new StyleError(`pair_modify keyword '${key}' is not supported for pair style ${this.name}`);
  }

  /** Energy and force of one pair (for pair_write, compute pair/local); optional. */
  single?(i: number, j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number): { eng: number; fforce: number };

  /** Named values for other styles (e.g. 'cut_coul' for kspace). */
  extract(_name: string): unknown { return undefined; }
}

/** What a bonded style sees during compute(). Positions are owned atoms only. */
export interface BondedCompute {
  s: SimState;
  geom: Geometry;
  /** id -> index. */
  map: Int32Array;
  f: Float64Array;
  acc: Accum;
  /** This style's virial accumulator (acc.vbond, acc.vangle, ...). */
  virial: Float64Array;
  eatom: Float64Array | null;
  vatom: Float64Array | null;
  /** Emits a "WARNING: ..." line in the run log (e.g. FENE bond too long). */
  warn?: (text: string) => void;
}

/**
 * Tallies one bonded term: energy e on `atoms` (indices), split evenly for
 * per-atom energy; virial from the force on each atom times its position
 * relative to the first atom (r_k - r_0) . f_k, which equals the
 * pairwise-sum form for internal forces that sum to zero.
 */
export const bondedVirial = (
  bc: BondedCompute, atoms: readonly number[], rel: readonly number[], fk: readonly number[], e: number,
): void => {
  // rel and fk: 3 values per atom, rel relative to atom 0 (minimum image)
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
};

/** Base for bond, angle, dihedral and improper styles. */
export abstract class Bonded {
  abstract readonly name: string;
  abstract readonly kind: 'bond' | 'angle' | 'dihedral' | 'improper';
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
  /** Lines of the "<Kind> Coeffs" data-file section, or null. */
  dataCoeffs(): string[] | null { return null; }
  /** Equilibrium length (bond) or angle in degrees (angle), for fix shake. */
  equilibrium(_type: number): number { return Number.NaN; }
}

/** What a kspace style sees during compute(). */
export interface KSpaceCompute {
  s: SimState;
  geom: Geometry;
  f: Float64Array;
  qqrd2e: number;
  acc: Accum;
  eatom: Float64Array | null;
  vatom: Float64Array | null;
}

export abstract class KSpace {
  abstract readonly name: string;
  /** Relative accuracy target from kspace_style. */
  accuracy = 1e-4;
  gEwald = 0;
  abstract settings(args: string[], ctx: StyleContext): void;
  /** kspace_modify keyword handler; returns the number of values consumed. */
  abstract modify(key: string, values: string[]): number;
  /** Sets up parameters for the current box and pair cutoff. */
  abstract init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void;
  abstract compute(kc: KSpaceCompute): void;
  /** Force-field hook, called before init(): the pair style (pppm/tip4p reads its charge sites). */
  linkPair?(pair: Pair | null): void;
}
