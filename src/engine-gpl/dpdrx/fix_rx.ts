/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-REACT/fix_rx.cpp and fix_rx.h (FixRX:
 * fix rx, the reaction kinetics of the DPD-REACT package: the species concentrations of each particle are integrated by
 * the rate laws of a reaction file, with a classic Runge-Kutta 4th-order stepper (lammps_rk4) or a Runge-Kutta-Fehlberg
 * 4(5) stepper (lammps_rkf45), in the dense or the sparse stoichiometric form). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Contributing authors (from fix_rx.cpp): James Larentzos (U.S. Army Research Laboratory),
 * Timothy Mattox (Engility Corporation) and John Brennan (U.S. Army Research Laboratory).
 *
 * docs.lammps.org/fix_rx.html: "fix ID group-ID rx file-name local-temperature-type kinetics-type ode-integration-type
 * ..." (the local temperature types are none and lucy; lucy needs a neighbor-list weighting and is refused here).
 * The species are the names in the reaction file; each species is a per-atom custom vector d_name with its old value
 * d_nameOld (the fix's own property/atom fix, as native's FixPropertyAtom pair). fix_modify rx_chemistry off|on skips
 * the reactions in pre_force (setup always copies the species to their old values).
 *
 * The species are numbered in the iteration order of native's std::unordered_set (unordered_order.ts, verified against
 * libstdc++), because native's species sums, and a clamp in pair exp6/rx that depends on their rounding, follow it.
 * Restart files (restartFlag), diagnostics (the optional last argument of rkf45) and the local temperature lucy are not
 * supported and are refused with a StyleError. More than 13 species keep the order of first reading (see the constructor).
 */
import { Fix } from '../../engine/fix/fix';
import { FixPropertyAtom } from '../../engine/fix/property_atom';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { SimState } from '../../engine/types';
import { DPD_THETA } from '../../engine/atoms';
import { TextLines, Tokens } from './rx_table_reader';
import { pairMatch } from './pair_match';
import { unorderedSetOrder } from './unordered_order';

/** fix_rx.cpp: "static constexpr double MY_EPSILON = 10.0*DBL_EPSILON;". */
export const MY_EPSILON = 10.0 * 2.220446049250313e-16;
const UROUND = 2.220446049250313e-16;

/** math_special.h powint: x**n by repeated squaring (exact for integer n, as native). */
export const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

/** One reaction line, read as FixRX::read_file does: stoichiometry by species index, then Arr, nArr and Ea. */
interface Reaction {
  stoich: Map<string, number>;
  arr: number;
  nArr: number;
  ea: number;
}

/**
 * The walk of a reaction line (fix_rx.cpp post_constructor and read_file): "coeff species [+|=] coeff species ..." with
 * the reactants before "=" (negative), then "Arr nArr Ea" after the last species. onTerm gets each species with its signed
 * coefficient (reactants negative, products positive).
 */
const walkReaction = (line: string, file: string, onTerm: (name: string, coeff: number, reactant: boolean) => void): { arr: number; nArr: number; ea: number } => {
  const w = line.trim().split(/\s+/);
  const tok = new Tokens(w, line);
  let sign = -1.0;
  for (;;) {
    const coeff = tok.double();
    if (coeff < 0.0) throw new StyleError(`Negative coefficient (${coeff}) found. Bad line is ${line}`);
    const species = tok.next();
    onTerm(species, coeff, sign < 0.0);
    const op = tok.next();
    if (op === '=') sign = 1.0;
    if (op !== '+' && op !== '=') {
      const arr = Number(op);
      if (!Number.isFinite(arr)) throw new StyleError(`Expected floating-point number instead of '${op}' in file ${file}. Full line: ${line}`);
      const nArr = tok.double();
      const ea = tok.double();
      if (tok.hasNext()) throw new StyleError(`Misplaced characters at end of line in file ${file}. Full line: ${line}`);
      return { arr, nArr, ea };
    }
  }
};

/** The stoichiometric data of the reactions in the sparse form (fix_rx.cpp initSparse). */
interface Sparse {
  maxReactants: number;
  maxSpecies: number;
  nu: Float64Array;
  nuk: Int32Array;
  inu: Int32Array;
  integral: Uint8Array;
}

/** Index of an invalid species slot of the sparse form (SparseKinetics_invalidIndex). */
const INVALID = -1;

export class FixRX extends Fix {
  readonly style = 'rx';
  readonly nspecies: number;
  /** Species names in index order; the per-atom vectors are d_NAME and d_NAMEOld. */
  readonly speciesNames: string[];
  private readonly speciesIndex = new Map<string, number>();
  private readonly propFix: FixPropertyAtom;
  private readonly nreactions: number;
  private readonly arr: Float64Array;
  private readonly nArr: Float64Array;
  private readonly ea: Float64Array;
  /** stoich (signed), stoichReactants and stoichProducts, row-major (reaction, species). */
  private readonly stoich: Float64Array;
  private readonly stoichR: Float64Array;
  private readonly stoichP: Float64Array;
  private readonly useSparse: boolean;
  private readonly sparse: Sparse | null = null;
  private readonly rk4Solver: boolean;
  private readonly minSteps: number;
  private readonly maxIters: number;
  private readonly relTol: number;
  private readonly absTol: number;
  private skipChemistry = false;
  /** Forward rate constants of the current atom and the rate laws (rxnRateLaw). */
  private readonly kFor: Float64Array;
  private readonly rateLaw: Float64Array;
  /** Solver work arrays (sized once: rk4 k1..k4, yp; rkf45 f1..f6, the error and the trial solution). */
  private readonly y: Float64Array;
  private readonly yp: Float64Array;
  private readonly k: Float64Array[];
  private readonly yout: Float64Array;
  private readonly eout: Float64Array;
  private readonly fs: Float64Array[];
  private readonly h0ydot: Float64Array;
  private readonly h0y1: Float64Array;
  private readonly h0ydot1: Float64Array;
  /** Per-species data of the state the current call uses (bound at each entry point). */
  private cur: Float64Array[] = [];
  private old: Float64Array[] = [];
  private vdpd = 1;
  private dt = 0;
  private boltz = 1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError('Illegal fix rx command');
    if (args.length > 9) throw new StyleError('Too many arguments for fix rx');
    this.nevery = 1;
    const kineticsFile = args[0];
    // fix_rx.cpp: "Unknown fix rx local temperature weighting technique"; lucy needs the neighbor-list weights (refused)
    if (args[1] === 'lucy') throw new StyleError('fix rx local temperature weighting lucy is not supported by the browser engine');
    if (args[1] !== 'none') throw new StyleError(`Unknown fix rx local temperature weighting technique ${args[1]}`);
    if (args[2] === 'sparse') this.useSparse = true;
    else if (args[2] === 'dense') this.useSparse = false;
    else throw new StyleError(`Unknown kinetics keyword ${args[2]}. Expected "sparse" or "dense"`);
    if (args[3] === 'lammps_rk4' || args[3] === 'rk4') this.rk4Solver = true;
    else if (args[3] === 'lammps_rkf45' || args[3] === 'rkf45') this.rk4Solver = false;
    else throw new StyleError(`Unknown ODE integration type ${args[3]}`);
    // fix_rx.cpp constructor: the defaults, then the solver's own arguments
    this.minSteps = 1;
    this.maxIters = 100;
    this.relTol = 1.0e-6;
    this.absTol = 1.0e-8;
    if (this.rk4Solver) {
      if (args.length === 5) this.minSteps = intArg(args[4]);
      else if (args.length > 5) throw new StyleError(`Too many arguments (${args.length + 4}) for RK4 solver`);
    } else {
      if (args.length < 7) throw new StyleError('Missing arguments for fix rx RKF45');
      this.minSteps = intArg(args[4]);
      this.maxIters = intArg(args[5]);
      this.relTol = numArg(args[6]);
      this.absTol = numArg(args[7]);
      if (args.length === 9 && intArg(args[8]) !== 0) {
        throw new StyleError('fix rx ODE diagnostics (the optional last argument) are not supported by the browser engine');
      }
      this.maxIters = Math.max(this.minSteps, this.maxIters);
    }
    // fix_rx.cpp post_constructor: the species are the names in the reactions, in the order they first appear here
    const lines = new TextLines(sys.readFile(kineticsFile), kineticsFile);
    const reactionLines: string[] = [];
    for (let line = lines.next(); line; line = lines.next()) reactionLines.push(line.join(' '));
    this.nreactions = reactionLines.length;
    // the names in the order they are first read; native puts them in a std::unordered_set and numbers them in its
    // iteration order (unordered_order.ts), which sets the summation order of the species sums
    const firstSeen: string[] = [];
    const firstSeenSet = new Set<string>();
    for (const line of reactionLines) {
      walkReaction(line, kineticsFile, (name) => {
        if (!firstSeenSet.has(name)) {
          firstSeenSet.add(name);
          firstSeen.push(name);
        }
      });
    }
    if (firstSeen.length === 0) throw new StyleError('There are no rx species specified.');
    // above 13 species the bucket table of unordered_order.ts does not cover native's growth; the order of first reading is used
    const names = firstSeen.length <= 13 ? unorderedSetOrder(firstSeen) : firstSeen;
    names.forEach((n, k) => this.speciesIndex.set(n, k));
    this.speciesNames = names;
    this.nspecies = names.length;
    const nr = this.nreactions, ns = this.nspecies;
    this.arr = new Float64Array(nr);
    this.nArr = new Float64Array(nr);
    this.ea = new Float64Array(nr);
    this.stoich = new Float64Array(nr * ns);
    this.stoichR = new Float64Array(nr * ns);
    this.stoichP = new Float64Array(nr * ns);
    reactionLines.forEach((line, r) => {
      const res = walkReaction(line, kineticsFile, (name, coeff, reactant) => {
        const k = this.speciesIndex.get(name)!;
        this.stoich[r * ns + k] += (reactant ? -1.0 : 1.0) * coeff;
        if (reactant) this.stoichR[r * ns + k] += coeff;
        else this.stoichP[r * ns + k] += coeff;
      });
      this.arr[r] = res.arr;
      this.nArr[r] = res.nArr;
      this.ea[r] = res.ea;
    });
    if (this.useSparse) this.sparse = this.buildSparse();
    this.kFor = new Float64Array(nr);
    this.rateLaw = new Float64Array(nr);
    this.y = new Float64Array(ns);
    this.yp = new Float64Array(ns);
    this.k = [0, 1, 2, 3].map(() => new Float64Array(ns));
    this.yout = new Float64Array(ns);
    this.eout = new Float64Array(ns);
    this.fs = [0, 1, 2, 3, 4, 5].map(() => new Float64Array(ns));
    this.h0ydot = new Float64Array(ns);
    this.h0y1 = new Float64Array(ns);
    this.h0ydot1 = new Float64Array(ns);
    // the per-atom species vectors (the fix property/atom pair of native's post_constructor, ghost yes)
    const props: string[] = [];
    for (const n of names) props.push(`d_${n}`);
    for (const n of names) props.push(`d_${n}Old`);
    this.propFix = new FixPropertyAtom(sys, `${id}_SPECIES`, group, [...props, 'ghost', 'yes']);
  }

  /** Stoichiometry in the sparse form: the reactants first, then the products, slots of each reaction (initSparse). */
  private buildSparse(): Sparse {
    const nr = this.nreactions, ns = this.nspecies;
    let maxReactants = 0, maxProducts = 0;
    for (let r = 0; r < nr; r++) {
      let nre = 0, npr = 0;
      for (let k = 0; k < ns; k++) {
        if (this.stoichR[r * ns + k] > 0.0) nre++;
        if (this.stoichP[r * ns + k] > 0.0) npr++;
      }
      maxReactants = Math.max(maxReactants, nre);
      maxProducts = Math.max(maxProducts, npr);
    }
    const maxSpecies = maxReactants + maxProducts;
    const nu = new Float64Array(nr * maxSpecies);
    const nuk = new Int32Array(nr * maxSpecies).fill(INVALID);
    const inu = new Int32Array(nr * maxSpecies);
    const integral = new Uint8Array(nr);
    for (let r = 0; r < nr; r++) {
      let nre = 0, npr = 0;
      let isIntegral = true;
      for (let k = 0; k < ns; k++) {
        if (this.stoichR[r * ns + k] > 0.0) {
          const idx = nre;
          nu[r * maxSpecies + idx] = this.stoichR[r * ns + k];
          nuk[r * maxSpecies + idx] = k;
          isIntegral = isIntegral && (this.stoichR[r * ns + k] % 1.0 === 0.0);
          inu[r * maxSpecies + idx] = Math.trunc(nu[r * maxSpecies + idx]);
          nre++;
        }
        if (this.stoichP[r * ns + k] > 0.0) {
          const idx = maxReactants + npr;
          nu[r * maxSpecies + idx] = this.stoichP[r * ns + k];
          nuk[r * maxSpecies + idx] = k;
          isIntegral = isIntegral && (nu[r * maxSpecies + idx] % 1.0 === 0.0);
          inu[r * maxSpecies + idx] = Math.trunc(nu[r * maxSpecies + idx]);
          npr++;
        }
      }
      integral[r] = isIntegral ? 1 : 0;
    }
    return { maxReactants, maxSpecies, nu, nuk, inu, integral };
  }

  /** The per-atom vectors of the species of a state (d_NAME), in index order. */
  speciesData(s: SimState): Float64Array[] {
    return this.speciesNames.map((n) => {
      const c = s.custom.get(n);
      if (!c) throw new StyleError(`fix rx: species ${n} has no per-atom vector`);
      return c.data;
    });
  }

  /** Index of a species by name (the species order of this fix), or undefined. */
  speciesIndexOf(name: string): number | undefined {
    return this.speciesIndex.get(name);
  }

  attachState(s: SimState): void {
    this.propFix.attachState(s);
  }

  init(): void {
    const sys = this.sys;
    const p = sys.ff.pair;
    if (!pairMatch(p, 'dpd/fdt/energy')) throw new StyleError('Must use pair_style dpd/fdt/energy with fix rx');
    if (!sys.fixes.some((f) => f.style === 'eos/table/rx')) throw new StyleError('fix rx requires fix eos/table/rx to be specified');
  }

  setupPreForce(): void {
    const s = this.sys.state;
    this.bind(s);
    // fix_rx.cpp setup_pre_force: the old species are the current ones; no reaction occurs at step 0 (rates zero)
    for (let i = 0; i < s.n; i++) for (let k = 0; k < this.nspecies; k++) this.old[k][i] = this.cur[k][i];
    this.kFor.fill(0);
    for (let i = 0; i < s.n; i++) {
      if (s.mask[i] & this.groupBit) this.solve(i);
    }
  }

  preForce(): void {
    if (this.skipChemistry) return;
    const s = this.sys.state;
    this.bind(s);
    const theta = s.custom.get(DPD_THETA)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const th = theta[i];
      // fix_rx.cpp pre_force: kFor = Arr T^nArr exp(-Ea/(kB T))
      for (let r = 0; r < this.nreactions; r++) {
        this.kFor[r] = this.arr[r] * Math.pow(th, this.nArr[r]) * Math.exp(-this.ea[r] / this.boltz / th);
      }
      this.solve(i);
    }
  }

  /** Per-atom solve of the species (rk4 or rkf45, fix_rx.cpp pre_force). */
  private solve(id: number): void {
    if (this.rk4Solver) this.rk4(id);
    else this.rkf45(id);
  }

  /** Binds the species vectors and the box/units of the state the entry point uses. */
  private bind(s: SimState): void {
    this.cur = this.speciesData(s);
    this.old = this.speciesNames.map((n) => s.custom.get(`${n}Old`)!.data);
    const b = s.box;
    const xprd = b.hi[0] - b.lo[0], yprd = b.hi[1] - b.lo[1], zprd = b.hi[2] - b.lo[2];
    // fix_rx.cpp rhs: VDPD = xprd * yprd * zprd / atom->natoms
    this.vdpd = (xprd * yprd * zprd) / s.n;
    this.dt = s.dt;
    this.boltz = s.units.boltz;
  }

  /** fix_rx.cpp rk4: Old = current, then minSteps classic steps of size dt/minSteps. */
  private rk4(id: number): void {
    const ns = this.nspecies, y = this.y, yp = this.yp;
    const [k1, k2, k3, k4] = this.k;
    for (let k = 0; k < ns; k++) {
      const tmp = this.cur[k][id];
      this.old[k][id] = tmp;
      y[k] = tmp;
    }
    const h = this.dt / this.minSteps;
    for (let step = 0; step < this.minSteps; step++) {
      this.rhs(y, k1);
      for (let k = 0; k < ns; k++) yp[k] = y[k] + 0.5 * h * k1[k];
      this.rhs(yp, k2);
      for (let k = 0; k < ns; k++) yp[k] = y[k] + 0.5 * h * k2[k];
      this.rhs(yp, k3);
      for (let k = 0; k < ns; k++) yp[k] = y[k] + h * k3[k];
      this.rhs(yp, k4);
      for (let k = 0; k < ns; k++) y[k] += h * (k1[k] / 6.0 + k2[k] / 3.0 + k3[k] / 3.0 + k4[k] / 6.0);
    }
    for (let k = 0; k < ns; k++) {
      if (y[k] < -MY_EPSILON) throw new StyleError('Computed concentration in RK4 solver is < -10*DBL_EPSILON');
      else if (y[k] < MY_EPSILON) y[k] = 0.0;
      this.cur[k][id] = y[k];
    }
  }

  /**
   * fix_rx.cpp rkf45: Runge-Kutta-Fehlberg with the 4th-order solution as the update and an error estimate from the
   * 5th-order one; the step is adapted by hsafe and bounded by a factor of 4. Atoms with failed solves are counted and the
   * run goes on with the last trial value (native warns once per step).
   */
  private rkf45(id: number): void {
    const ns = this.nspecies, y = this.y;
    for (let k = 0; k < ns; k++) {
      const tmp = this.cur[k][id];
      this.old[k][id] = tmp;
      y[k] = tmp;
    }
    const tStop = this.dt;
    const hsafe = 0.840896415;
    const tround = tStop * UROUND;
    const hMin = 100.0 * tround;
    const hMax = this.minSteps > 0 ? tStop / this.minSteps : tStop;
    let h = this.minSteps > 0 ? tStop / this.minSteps : 0.0;
    let t = 0.0;
    if (h < hMin) h = this.rkf45H0(ns, hMin, hMax);
    let nit = 0;
    const yout = this.yout, eout = this.eout;
    while (Math.abs(t - tStop) > tround) {
      this.rkf45Step(ns, h, y, yout, eout);
      let err2 = 0.0;
      for (let k = 0; k < ns; k++) {
        const wterr = eout[k] / (this.relTol * Math.abs(y[k]) + this.absTol);
        err2 += wterr * wterr;
      }
      const err = Math.max(UROUND, Math.sqrt(err2 / ns));
      if (err <= 1.0 || h <= hMin) {
        t += h;
        for (let k = 0; k < ns; k++) y[k] = yout[k];
      }
      let hfac = hsafe * Math.sqrt(Math.sqrt(1.0 / err));
      hfac = Math.max(hfac, 1.0 / 4.0);
      hfac = Math.min(hfac, 4.0);
      h *= hfac;
      h = Math.min(h, hMax);
      h = Math.max(h, hMin);
      if (err <= 1.0 && t + 1.05 * h > tStop) h = tStop - t;
      if (t + h > tStop) h = tStop - t;
      nit++;
      if (this.maxIters && nit > this.maxIters) break;
    }
    for (let k = 0; k < ns; k++) {
      if (y[k] < -1.0e-10) throw new StyleError('Computed concentration in RKF45 solver is < -1.0e-10');
      else if (y[k] < MY_EPSILON) y[k] = 0.0;
      this.cur[k][id] = y[k];
    }
  }

  /** fix_rx.cpp rkf45_step: one Fehlberg step; yout gets the 4th-order solution, eout the |5th - 4th| error. */
  private rkf45Step(neq: number, h: number, y: Float64Array, yout: Float64Array, eout: Float64Array): void {
    const [f1, f2, f3, f4, f5, f6] = this.fs;
    const c21 = 0.25;
    const c31 = 0.09375, c32 = 0.28125;
    const c41 = 0.87938097405553, c42 = -3.2771961766045, c43 = 3.3208921256258;
    const c51 = 2.0324074074074, c52 = -8.0, c53 = 7.1734892787524, c54 = -0.20589668615984;
    const c61 = -0.2962962962963, c62 = 2.0, c63 = -1.3816764132554, c64 = 0.45297270955166, c65 = -0.275;
    const a1 = 0.11574074074074, a3 = 0.54892787524366, a4 = 0.5353313840156, a5 = -0.2;
    const b1 = 0.11851851851852, b3 = 0.51898635477583, b4 = 0.50613149034201, b5 = -0.18, b6 = 0.036363636363636;
    const ytmp = yout;
    this.rhs(y, f1);
    for (let k = 0; k < neq; k++) { f1[k] *= h; ytmp[k] = y[k] + c21 * f1[k]; }
    this.rhs(ytmp, f2);
    for (let k = 0; k < neq; k++) { f2[k] *= h; ytmp[k] = y[k] + c31 * f1[k] + c32 * f2[k]; }
    this.rhs(ytmp, f3);
    for (let k = 0; k < neq; k++) { f3[k] *= h; ytmp[k] = y[k] + c41 * f1[k] + c42 * f2[k] + c43 * f3[k]; }
    this.rhs(ytmp, f4);
    for (let k = 0; k < neq; k++) { f4[k] *= h; ytmp[k] = y[k] + c51 * f1[k] + c52 * f2[k] + c53 * f3[k] + c54 * f4[k]; }
    this.rhs(ytmp, f5);
    for (let k = 0; k < neq; k++) { f5[k] *= h; ytmp[k] = y[k] + c61 * f1[k] + c62 * f2[k] + c63 * f3[k] + c64 * f4[k] + c65 * f5[k]; }
    this.rhs(ytmp, f6);
    for (let k = 0; k < neq; k++) {
      f6[k] *= h;
      const r5 = b1 * f1[k] + b3 * f3[k] + b4 * f4[k] + b5 * f5[k] + b6 * f6[k];
      const r4 = a1 * f1[k] + a3 * f3[k] + a4 * f4[k] + a5 * f5[k];
      eout[k] = Math.abs(r5 - r4);
      yout[k] = y[k] + r4;
    }
  }

  /** fix_rx.cpp rkf45_h0: the initial step from the second derivative (WRMS norm = 1); returns h0. */
  private rkf45H0(neq: number, hMin: number, hMax: number): number {
    let hg = Math.sqrt(hMin * hMax);
    const ydot = this.h0ydot, y1 = this.h0y1, ydot1 = this.h0ydot1, y = this.y;
    const maxIt = 10;
    let hnewOk = false;
    let hnew: number;
    let iter = 0;
    this.rhs(y, ydot);
    for (;;) {
      for (let k = 0; k < neq; k++) y1[k] = y[k] + hg * ydot[k];
      this.rhs(y1, ydot1);
      let yddnrm = 0.0;
      for (let k = 0; k < neq; k++) {
        const ydd = (ydot1[k] - ydot[k]) / hg;
        const wterr = ydd / (this.relTol * Math.abs(y[k]) + this.absTol);
        yddnrm += wterr * wterr;
      }
      yddnrm = Math.sqrt(yddnrm / neq);
      if (hnewOk || iter === maxIt) {
        hnew = hg;
        break;
      }
      hnew = yddnrm * hMax * hMax > 2.0 ? Math.sqrt(2.0 / yddnrm) : Math.sqrt(hg * hMax);
      const hrat = hnew / hg;
      if (hrat > 0.5 && hrat < 2.0) hnewOk = true;
      if (iter > 1 && hrat > 2.0) {
        hnew = hg;
        hnewOk = true;
      }
      hg = hnew;
      iter++;
    }
    let h0 = hnew * 0.5;
    h0 = Math.max(h0, hMin);
    h0 = Math.min(h0, hMax);
    return h0;
  }

  /** Right-hand side: the rates of the species at concentrations y (dense or sparse form). */
  private rhs(y: Float64Array, dydt: Float64Array): void {
    if (this.sparse) this.rhsSparse(y, dydt, this.sparse);
    else this.rhsDense(y, dydt);
  }

  /** fix_rx.cpp rhs_dense. */
  private rhsDense(y: Float64Array, dydt: Float64Array): void {
    const ns = this.nspecies, nr = this.nreactions, vdpd = this.vdpd;
    for (let k = 0; k < ns; k++) dydt[k] = 0.0;
    for (let r = 0; r < nr; r++) {
      let rate = this.kFor[r];
      for (let k = 0; k < ns; k++) {
        const conc = y[k] / vdpd;
        rate *= Math.pow(conc, this.stoichR[r * ns + k]);
      }
      this.rateLaw[r] = rate;
    }
    for (let k = 0; k < ns; k++) {
      for (let r = 0; r < nr; r++) dydt[k] += this.stoich[r * ns + k] * vdpd * this.rateLaw[r];
    }
  }

  /** fix_rx.cpp rhs_sparse (powint for the reactions whose coefficients are all integers). */
  private rhsSparse(y: Float64Array, dydt: Float64Array, sp: Sparse): void {
    const ns = this.nspecies, nr = this.nreactions, vdpd = this.vdpd;
    const { maxReactants, maxSpecies, nu, nuk, inu, integral } = sp;
    const conc = dydt;
    for (let k = 0; k < ns; k++) conc[k] = y[k] / vdpd;
    for (let i = 0; i < nr; i++) {
      const base = i * maxSpecies;
      let rate: number;
      if (integral[i]) {
        rate = this.kFor[i] * powint(conc[nuk[base]], inu[base]);
        for (let kk = 1; kk < maxReactants; kk++) {
          const k = nuk[base + kk];
          if (k === INVALID) break;
          rate *= powint(conc[k], inu[base + kk]);
        }
      } else {
        rate = this.kFor[i] * Math.pow(conc[nuk[base]], nu[base]);
        for (let kk = 1; kk < maxReactants; kk++) {
          const k = nuk[base + kk];
          if (k === INVALID) break;
          rate *= Math.pow(conc[k], nu[base + kk]);
        }
      }
      this.rateLaw[i] = rate;
    }
    for (let k = 0; k < ns; k++) dydt[k] = 0.0;
    for (let i = 0; i < nr; i++) {
      const base = i * maxSpecies;
      const rate = this.rateLaw[i];
      dydt[nuk[base]] -= nu[base] * rate;
      for (let kk = 1; kk < maxReactants; kk++) {
        const k = nuk[base + kk];
        if (k === INVALID) break;
        dydt[k] -= nu[base + kk] * rate;
      }
      dydt[nuk[base + maxReactants]] += nu[base + maxReactants] * rate;
      for (let kk = maxReactants + 1; kk < maxSpecies; kk++) {
        const k = nuk[base + kk];
        if (k === INVALID) break;
        dydt[k] += nu[base + kk] * rate;
      }
    }
    for (let k = 0; k < ns; k++) dydt[k] *= vdpd;
  }

  /** fix_modify rx_chemistry on|off (fix_rx.cpp modify_param); other keywords go to the base class. */
  modify(key: string, values: string[]): number {
    if (key !== 'rx_chemistry') return super.modify(key, values);
    const w = values[0];
    if (w === undefined) throw new StyleError('Illegal fix_modify rx_chemistry command');
    const on = ['yes', 'on', 'true', '1'].includes(w);
    const off = ['no', 'off', 'false', '0'].includes(w);
    if (!on && !off) throw new StyleError(`Expected boolean parameter instead of '${w}' for fix_modify rx_chemistry`);
    this.skipChemistry = !on;
    return 1;
  }
}

/** A keyword-argument integer (utils::inumeric). */
const intArg = (w: string): number => {
  if (!/^[-+]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}'`);
  return Number(w);
};

/** A keyword-argument number (utils::numeric). */
const numArg = (w: string): number => {
  const v = Number(w);
  if (w === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating-point parameter instead of '${w}'`);
  return v;
};
