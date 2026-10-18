/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MC/fix_atom_swap.cpp and
 * src/MC/fix_atom_swap.h (Paul Crozier (SNL), Alexander Stukowski). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix atom/swap (docs.lammps.org/fix_atom_swap.html): every N steps, ncycles Monte Carlo attempts. Canonical mode swaps
 * the types of an atom of types[0] and one of types[1] (exactly two types); semi-grand mode (semi-grand yes) changes the
 * type of one random atom to another of the listed types, with the chemical potentials mu.
 *
 * Native structure kept here, for one processor: the fix runs in pre_exchange on the steps where its reneighbouring is
 * due (next_reneighbor), after the initial integration and before the force evaluation of the step. The atom lists
 * follow the native local order (nativeOrder), the random draws follow native's sequence: random_equal (the seed) for
 * the picks and, in canonical mode, the acceptance test; random_unequal (the seed, a second stream) for the semi-grand
 * type choice and its acceptance test. Each energy is a full force evaluation of the system with the trial types
 * (energy_full: the pair, bond, angle, dihedral, improper and kspace energies and the fix energies, as compute pe).
 * Metropolis acceptance uses the glibc exp (src/engine-gpl/libm). ke yes rescales an accepted atom's velocity by
 * sqrt(m_old / m_new).
 *
 * Not reproduced: the visualization (fix_modify vizsteps and dump image spheres), restart files (write_restart and
 * restart), and the first-group check of atom_modify first (the engine keeps no first group). The engine rebuilds the
 * neighbour lists before every energy evaluation, which native does only when the cutoffs of the swapped types differ;
 * the energies agree to rounding. A per-atom-mass (rmass) semi-grand run is refused: native's per-type masses are not
 * set in that case.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { hasCharge, nativeOrder } from '../../engine/atoms';
import { RanPark } from '../random_park';
import { exp as libmExp } from '../libm/exp';

/** utils::inumeric (utils.cpp): an integer or an error naming the argument. */
const inumeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isInteger(v)) {
    throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return v;
};
/** utils::numeric (utils.cpp). */
const numeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) {
    throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return v;
};
/** utils::logical (utils.cpp). */
const logical = (w: string | undefined): boolean => {
  const t = (w ?? '').toLowerCase();
  if (t === 'yes' || t === 'on' || t === 'true' || t === '1') return true;
  if (t === 'no' || t === 'off' || t === 'false' || t === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w ?? ''}' in input script or data file`);
};

export class FixAtomSwap extends Fix {
  readonly style = 'atom/swap';
  private readonly neveryN: number;
  private readonly ncycles: number;
  private readonly seed: number;
  private readonly beta: number;
  private readonly typeList: number[] = [];
  /** mu[type] for type 1..ntypes (native indexes mu by type, the k-th mu value going to mu[k]) */
  private readonly mu: Float64Array;
  private keFlag = true;
  private semiGrand = false;
  private nswaptypes = 0;
  private nmutypes = 0;
  private regionId: string | null = null;
  private readonly equal: RanPark;
  private readonly unequal: RanPark;
  private nextReneighbor: number;
  private attempts = 0;
  private successes = 0;
  private energyStored = 0;
  /** per-type charges and masses of the swapped types (canonical mode), index 0 = type_list[0] */
  private qtype: number[] = [];
  private mtype: number[] = [];
  private sqrtMassRatio = new Float64Array(0);
  /** native local swap lists (engine atom indices), rebuilt after every accepted swap */
  private swapAtomList = new Int32Array(0);
  private nswapAll = 0;
  private iList = new Int32Array(0);
  private jList = new Int32Array(0);
  private niswap = 0;
  private njswap = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 7) throw new StyleError('Illegal fix atom/swap command: missing argument(s)');
    const s = sys.state;
    this.vectorFlag = true;
    this.sizeVector = 2;
    this.extvector = 0;
    this.globalFreq = 1;
    this.neveryN = inumeric(args[0]);
    this.ncycles = inumeric(args[1]);
    this.seed = inumeric(args[2]);
    const temperature = numeric(args[3]);
    if (this.neveryN <= 0) throw new StyleError('Illegal fix atom/swap command nevery value');
    if (this.ncycles < 0) throw new StyleError('Illegal fix atom/swap command ncycles value');
    if (this.seed <= 0) throw new StyleError('Illegal fix atom/swap command random seed');
    if (temperature <= 0.0) throw new StyleError('Illegal fix atom/swap command temperature value');
    this.beta = 1.0 / (s.units.boltz * temperature);
    this.mu = new Float64Array(s.ntypes + 1);
    this.options(args.slice(4));
    this.equal = new RanPark(this.seed);
    this.unequal = new RanPark(this.seed);
    this.nextReneighbor = s.step + 1;
  }

  /** FixAtomSwap::options: region, ke, semi-grand, types and mu, the keywords after the seven required arguments. */
  private options(args: string[]): void {
    const s = this.sys.state;
    let i = 0;
    while (i < args.length) {
      const key = args[i];
      if (key === 'region') {
        if (i + 2 > args.length) throw new StyleError('Illegal fix atom/swap command');
        const name = args[i + 1];
        if (!this.sys.regions.has(name)) throw new StyleError(`Region ${name} for fix atom/swap does not exist`);
        this.regionId = name;
        i += 2;
      } else if (key === 'ke') {
        if (i + 2 > args.length) throw new StyleError('Illegal fix atom/swap command');
        this.keFlag = logical(args[i + 1]);
        i += 2;
      } else if (key === 'semi-grand') {
        if (i + 2 > args.length) throw new StyleError('Illegal fix atom/swap command');
        this.semiGrand = logical(args[i + 1]);
        i += 2;
      } else if (key === 'types') {
        if (i + 3 > args.length) throw new StyleError('Illegal fix atom/swap command');
        i++;
        while (i < args.length) {
          if (/^[A-Za-z]/.test(args[i])) break;
          if (this.nswaptypes >= s.ntypes) throw new StyleError('Illegal fix atom/swap command');
          this.typeList[this.nswaptypes] = inumeric(args[i]);
          this.nswaptypes++;
          i++;
        }
      } else if (key === 'mu') {
        if (i + 2 > args.length) throw new StyleError('Illegal fix atom/swap command');
        i++;
        while (i < args.length) {
          if (/^[A-Za-z]/.test(args[i])) break;
          this.nmutypes++;
          if (this.nmutypes > s.ntypes) throw new StyleError('Illegal fix atom/swap command');
          this.mu[this.nmutypes] = numeric(args[i]);
          i++;
        }
      } else {
        throw new StyleError('Illegal fix atom/swap command');
      }
    }
  }

  /** setmask PRE_EXCHANGE only: the run loop calls preExchange on the steps this returns true (native force_reneighbor). */
  reneighborDue(step: number): boolean {
    return step === this.nextReneighbor;
  }

  init(): void {
    const sys = this.sys;
    const s = sys.state;
    if (s.rmass) sys.warn('Fix atom/swap will use per-atom masses for velocity rescaling');
    if (this.nswaptypes < 2) throw new StyleError('Must specify at least 2 atom types in fix atom/swap command');
    if (this.semiGrand) {
      if (this.nswaptypes !== this.nmutypes) throw new StyleError('Need nswaptypes mu values in fix atom/swap command');
    } else {
      if (this.nswaptypes !== 2) {
        throw new StyleError('Exactly 2 atom types must be used without semi-grand keyword in fix atom/swap');
      }
      if (this.nmutypes !== 0) throw new StyleError('Mu not allowed when not using semi-grand in fix atom/swap command');
    }
    const ff = sys.ff;
    if (!ff.pair) throw new StyleError('Fix atom/swap requires a pair style');
    if (this.regionId !== null && !sys.regions.has(this.regionId)) {
      throw new StyleError(`Region ${this.regionId} for fix atom/swap does not exist`);
    }
    for (let k = 0; k < this.nswaptypes; k++) {
      const t = this.typeList[k];
      if (t <= 0 || t > s.ntypes) throw new StyleError('Invalid atom type in fix atom/swap command');
    }
    const T = s.ntypes + 1;
    if (hasCharge(s) && !this.semiGrand) {
      this.qtype = new Array<number>(this.nswaptypes).fill(0);
      for (let k = 0; k < this.nswaptypes; k++) {
        let first = true;
        for (let i = 0; i < s.n; i++) {
          if (!(s.mask[i] & this.groupBit) || s.type[i] !== this.typeList[k]) continue;
          if (first) {
            this.qtype[k] = s.q[i];
            first = false;
          } else if (this.qtype[k] !== s.q[i]) {
            throw new StyleError('All atoms of a swapped type must have the same charge.');
          }
        }
        if (first) throw new StyleError('At least one atom of each swapped type must be present to define charges.');
      }
    }
    if (s.rmass && !this.semiGrand) {
      this.mtype = new Array<number>(this.nswaptypes).fill(0);
      for (let k = 0; k < this.nswaptypes; k++) {
        let first = 1;
        for (let i = 0; i < s.n; i++) {
          if (!(s.mask[i] & this.groupBit) || s.type[i] !== this.typeList[k]) continue;
          if (first > 0) {
            this.mtype[k] = s.rmass[i];
            first = 0;
          } else if (this.mtype[k] !== s.rmass[i]) {
            first = -1;
          }
        }
        if (first < 0) throw new StyleError('All atoms of a swapped type must have the same per-atom mass');
        if (first > 0) throw new StyleError('At least one atom of each swapped type must be present to define masses');
      }
    }
    if (s.rmass && this.semiGrand) {
      throw new StyleError('Fix atom/swap with semi-grand yes does not support per-atom masses');
    }
    this.sqrtMassRatio = new Float64Array(T * T);
    if (s.rmass) {
      this.sqrtMassRatio.fill(1.0);
      for (let a = 0; a < this.nswaptypes; a++) {
        const it = this.typeList[a];
        for (let b = 0; b < this.nswaptypes; b++) {
          const jt = this.typeList[b];
          this.sqrtMassRatio[it * T + jt] = Math.sqrt(this.mtype[a] / this.mtype[b]);
        }
      }
    } else {
      for (let it = 1; it <= s.ntypes; it++) {
        for (let jt = 1; jt <= s.ntypes; jt++) {
          this.sqrtMassRatio[it * T + jt] = Math.sqrt(s.massByType[it] / s.massByType[jt]);
        }
      }
    }
  }

  /** The Metropolis rule of attempt_swap / attempt_semi_grand, the swap lists, and the attempts of each event. */
  preExchange(): void {
    const s = this.sys.state;
    if (this.nextReneighbor !== s.step) return;
    // energy_stored = energy of the current state (energy_full)
    this.energyStored = this.energyFull();
    let nsuccess = 0;
    if (this.semiGrand) {
      this.updateSemiGrandList();
      for (let k = 0; k < this.ncycles; k++) nsuccess += this.attemptSemiGrand();
    } else {
      this.updateSwapList();
      for (let k = 0; k < this.ncycles; k++) nsuccess += this.attemptSwap();
    }
    this.attempts += this.ncycles;
    this.successes += nsuccess;
    this.nextReneighbor = s.step + this.neveryN;
  }

  /** attempt_semi_grand: one atom of the swap types takes another type of the list; the energies compared. */
  private attemptSemiGrand(): number {
    if (this.nswapAll === 0) return 0;
    const s = this.sys.state;
    const T = s.ntypes + 1;
    const energyBefore = this.energyStored;
    let i = this.pickSemiGrandAtom();
    let itype = 0, jtype = 0;
    if (i >= 0) {
      let jswaptype = Math.trunc(this.nswaptypes * this.unequal.uniform());
      jtype = this.typeList[jswaptype];
      itype = s.type[i];
      while (itype === jtype) {
        jswaptype = Math.trunc(this.nswaptypes * this.unequal.uniform());
        jtype = this.typeList[jswaptype];
      }
      s.type[i] = jtype;
    }
    const energyAfter = this.energyFull();
    let success = false;
    if (i >= 0) {
      const arg = this.beta * (energyBefore - energyAfter + this.mu[jtype] - this.mu[itype]);
      if (this.unequal.uniform() < libmExp(arg)) success = true;
    }
    if (success) {
      this.updateSemiGrandList();
      this.energyStored = energyAfter;
      if (this.keFlag && i >= 0) {
        const r = this.sqrtMassRatio[itype * T + jtype];
        s.v[3 * i] *= r;
        s.v[3 * i + 1] *= r;
        s.v[3 * i + 2] *= r;
      }
      return 1;
    }
    // rejected: restore the type (the lists are unchanged, as native)
    if (i >= 0) s.type[i] = itype;
    return 0;
  }

  /** attempt_swap: an atom of types[0] and one of types[1] exchange types; the Metropolis test decides. */
  private attemptSwap(): number {
    if (this.niswap === 0 || this.njswap === 0) return 0;
    const s = this.sys.state;
    const T = s.ntypes + 1;
    const energyBefore = this.energyStored;
    const i = this.pickIAtom();
    const j = this.pickJAtom();
    const itype = this.typeList[0];
    const jtype = this.typeList[1];
    if (i >= 0) {
      s.type[i] = jtype;
      if (hasCharge(s)) s.q[i] = this.qtype[1];
      if (s.rmass) s.rmass[i] = this.mtype[1];
    }
    if (j >= 0) {
      s.type[j] = itype;
      if (hasCharge(s)) s.q[j] = this.qtype[0];
      if (s.rmass) s.rmass[j] = this.mtype[0];
    }
    const energyAfter = this.energyFull();
    if (this.equal.uniform() < libmExp(this.beta * (energyBefore - energyAfter))) {
      this.updateSwapList();
      if (this.keFlag) {
        if (i >= 0) {
          const r = this.sqrtMassRatio[itype * T + jtype];
          s.v[3 * i] *= r; s.v[3 * i + 1] *= r; s.v[3 * i + 2] *= r;
        }
        if (j >= 0) {
          const r = this.sqrtMassRatio[jtype * T + itype];
          s.v[3 * j] *= r; s.v[3 * j + 1] *= r; s.v[3 * j + 2] *= r;
        }
      }
      this.energyStored = energyAfter;
      return 1;
    }
    // rejected: restore the types (and charges and masses) of both atoms
    if (i >= 0) {
      s.type[i] = this.typeList[0];
      if (hasCharge(s)) s.q[i] = this.qtype[0];
      if (s.rmass) s.rmass[i] = this.mtype[0];
    }
    if (j >= 0) {
      s.type[j] = this.typeList[1];
      if (hasCharge(s)) s.q[j] = this.qtype[1];
      if (s.rmass) s.rmass[j] = this.mtype[1];
    }
    return 0;
  }

  /**
   * energy_full: the force evaluation of the current state and the potential energy (compute pe: pair, bond, angle,
   * dihedral, improper, kspace, then the fix energies). The neighbour lists are rebuilt for the new types.
   */
  private energyFull(): number {
    const sys = this.sys;
    sys.bump();
    const a = sys.forces();
    return a.evdwl + a.ecoul + a.ebond + a.eangle + a.edihed + a.eimp + a.elong + sys.fixEnergy();
  }

  /** pick_semi_grand_atom: one draw from the equal stream over the swap atoms. */
  private pickSemiGrandAtom(): number {
    const iwhich = Math.trunc(this.nswapAll * this.equal.uniform());
    return this.swapAtomList[iwhich];
  }

  private pickIAtom(): number {
    const iwhich = Math.trunc(this.niswap * this.equal.uniform());
    return this.iList[iwhich];
  }

  private pickJAtom(): number {
    const jwhich = Math.trunc(this.njswap * this.equal.uniform());
    return this.jList[jwhich];
  }

  /** The atoms of the group (and region) whose type is one of the swap types, in native local order. */
  private inSwapRegion(e: number): boolean {
    if (this.regionId === null) return true;
    const x = this.sys.state.x;
    return this.sys.regions.get(this.regionId)!.match(x[3 * e], x[3 * e + 1], x[3 * e + 2]);
  }

  /** update_semi_grand_atoms_list */
  private updateSemiGrandList(): void {
    const s = this.sys.state;
    const order = nativeOrder(s);
    const list: number[] = [];
    for (let k = 0; k < s.n; k++) {
      const e = order[k];
      if (!this.inSwapRegion(e)) continue;
      if (!(s.mask[e] & this.groupBit)) continue;
      if (!this.typeList.slice(0, this.nswaptypes).includes(s.type[e])) continue;
      list.push(e);
    }
    this.swapAtomList = Int32Array.from(list);
    this.nswapAll = list.length;
  }

  /** update_swap_atoms_list */
  private updateSwapList(): void {
    const s = this.sys.state;
    const order = nativeOrder(s);
    const ilist: number[] = [], jlist: number[] = [];
    for (let k = 0; k < s.n; k++) {
      const e = order[k];
      if (!this.inSwapRegion(e)) continue;
      if (!(s.mask[e] & this.groupBit)) continue;
      if (s.type[e] === this.typeList[0]) ilist.push(e);
      else if (s.type[e] === this.typeList[1]) jlist.push(e);
    }
    this.iList = Int32Array.from(ilist);
    this.jList = Int32Array.from(jlist);
    this.niswap = ilist.length;
    this.njswap = jlist.length;
  }

  computeVector(i: number): number {
    if (i === 0) return this.attempts;
    if (i === 1) return this.successes;
    return 0.0;
  }
}

