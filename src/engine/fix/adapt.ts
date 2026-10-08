import { hasChargeStyle, isSphereStyle } from '../atoms';
import { Fix } from './fix';
import { Pair, PairParams, StyleError, typeBounds } from '../force/types';
import type { System } from '../system';
import type { SimState } from '../types';

/*
 * fix adapt — docs.lammps.org/fix_adapt.html. Implemented from that page only;
 * the measured rules are marked Measured with native LAMMPS (black box).
 *
 * Syntax (fix_adapt.rst): "fix ID group-ID adapt N attribute args ... keyword value ..."
 *   "N = adapt simulation settings every this many timesteps"
 *   "If :math:`N` is specified as 0, the specified attributes are only changed
 *   once, before the simulation begins."
 *   "*pair* args = pstyle pparam I J v_name"
 *   "*atom* args = atomparam v_name"; "charge = charge on particle";
 *   "diameter or diameter/disc = diameter of particle"
 *   "*scale* value = *no* or *yes*": "*no* = the variable value is the new setting",
 *   "*yes* = the variable value multiplies the original setting"
 *   "*reset* value = *no* or *yes*": "*yes* = reset altered values to their original
 *   values at the end of a run"
 *   "*mass* value = *no* or *yes*": "*no* = mass is not altered by changes in diameter"
 *   "The option defaults are scale = no, reset = no, mass = yes."
 *
 * Pair parameters: "Pair_coeff settings must be made **explicitly** in order for fix
 * adapt to be able to change them.  Settings inferred from mixing are not suitable."
 * "The *v_name* argument for keyword *pair* is the name of an" equal-style variable
 * "which will be evaluated each time" this fix is invoked "to set the parameter to a
 * new value."
 *
 * Atom attributes: "The new value is assigned to the corresponding attribute for all
 * atoms in the fix group."
 * Diameter and mass: "If the atom parameter is *diameter* and per-atom density and
 * per-atom mass are defined for particles" (atom_style granular in the doc; this engine's
 * atom_style sphere keeps the per-atom mass), "then the mass of each particle is, by
 * default, also changed when the diameter changes. The mass is set from the particle
 * volume for 3d systems (density is assumed to stay constant)."
 *
 * Measured with native LAMMPS (black box), 4x4x4 sc lattice, lj/cut:
 * - pre_force timing: the value is applied at setup (also N = 0; measured for the first run)
 *   and then at every step whose number is a multiple of N, before the forces of that step.
 * - reset yes restores the values the run started with (after run 2 with epsilon set to
 *   2, unfix and run 0 give the epsilon = 1 energy); reset no keeps the last value.
 * - scale yes multiplies the value the parameter had when the run started (a pair_coeff
 *   change before the run is the new original).
 * - a mixed (not pair_coeff-set) type pair targeted by adapt has no effect: the energy is
 *   unchanged; adapting the diagonal epsilon(1,1) re-mixes epsilon(1,2) from the new value
 *   (the same energy as pair_coeff 1 1 2.0 1.0 followed by the mixed 1-2).
 * - atom charge: adapt charge 2.0 on atoms of charge 1 with pair coul/cut gives pe 4x (q^2).
 * - atom diameter 1 -> 2 with atom_style sphere: radius 1, mass 0.5236 -> 4.1888 (d^3);
 *   mass no leaves the mass unchanged.
 * - atom_style sphere without the dynamic flag errors in native LAMMPS; this engine does
 *   not track that flag and accepts it.
 *
 * Not supported (StyleError): bond, angle, dihedral, improper, kspace attributes; pair
 * styles and parameters other than those listed in ADAPTABLE (a cutoff is never adaptable:
 * the neighbor list would not follow it); hybrid pair styles and pstyle:N; mixed type
 * pairs; diameter/disc; hybrid sub-style selection.
 *
 */

/** Pair styles and parameters (lower-case) this engine can adapt; "cut" names are not adaptable. */
const ADAPTABLE: Record<string, string[]> = {
  'lj/cut': ['epsilon', 'sigma'],
  'lj/cut/coul/cut': ['epsilon', 'sigma'],
  soft: ['a'],
  morse: ['d0', 'alpha', 'r0'],
};

const USAGE = 'usage: fix ID group-ID adapt N attribute args ... keyword value ...';

interface PairTarget {
  pstyle: string;
  /** Lower-case parameter name from the input, and the engine's name for it. */
  pparam: string;
  name: string;
  vname: string;
  /** Type words as given (resolved against ntypes at run start). */
  iw: string;
  jw: string;
  /** Explicit type pairs (i <= j) and their original values (captured at run start). */
  pairs: Array<[number, number]>;
  orig: Float64Array;
  table: PairParams;
  pair: Pair;
}

type AtomKind = 'charge' | 'diameter';
interface AtomTarget {
  kind: AtomKind;
  vname: string;
  /** Original per-atom values by atom ID (charge, or radius and mass for diameter). */
  origQ: Map<number, number>;
  origR: Map<number, number>;
  origM: Map<number, number>;
}

const yesNo = (key: string, w: string | undefined): boolean => {
  if (w !== 'yes' && w !== 'no') throw new StyleError(`fix adapt keyword ${key} must be yes or no, got '${w ?? ''}' (${USAGE})`);
  return w === 'yes';
};

const TYPE_WORD = /^(\*|\d+|\d*\*\d*)$/;

export class FixAdapt extends Fix {
  readonly style: string = 'adapt';
  private pairs: PairTarget[] = [];
  private readonly atoms: AtomTarget[] = [];
  private scale = false;
  private reset = false;
  private massFlag = true;
  /** True while init() applies values (before the force field is initialized). */
  private inInit = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const n = Number(args[0]);
    if (args[0] === undefined || !Number.isInteger(n) || n < 0) {
      throw new StyleError(`fix ${id} (adapt): N must be a non-negative integer, got '${args[0] ?? ''}' (${USAGE})`);
    }
    this.nevery = n;
    let k = 1;
    let nattr = 0;
    while (k < args.length) {
      const w = args[k];
      if (w === 'scale' || w === 'reset' || w === 'mass') {
        const v = yesNo(w, args[k + 1]);
        if (w === 'scale') this.scale = v;
        else if (w === 'reset') this.reset = v;
        else this.massFlag = v;
        k += 2;
        continue;
      }
      if (w === 'pair') {
        const pstyle = args[k + 1], pparam = args[k + 2], iw = args[k + 3], jw = args[k + 4], vw = args[k + 5];
        if (vw === undefined) throw new StyleError(`fix ${id} (adapt): pair needs pstyle pparam I J v_name (${USAGE})`);
        this.addPair(id, pstyle, pparam, iw, jw, vw);
        k += 6; nattr++;
        continue;
      }
      if (w === 'atom') {
        const kind = args[k + 1], vw = args[k + 2];
        if (vw === undefined) throw new StyleError(`fix ${id} (adapt): atom needs atomparam v_name (${USAGE})`);
        if (kind === 'diameter/disc') throw new StyleError(`fix ${id} (adapt): atom diameter/disc is not supported by this engine (2d disc particles)`);
        if (kind !== 'charge' && kind !== 'diameter') throw new StyleError(`fix ${id} (adapt): atom parameter '${kind}' is not supported (charge, diameter)`);
        this.checkVariable(id, vw);
        this.atoms.push({ kind, vname: vw.slice(2), origQ: new Map(), origR: new Map(), origM: new Map() });
        k += 3; nattr++;
        continue;
      }
      if (w === 'bond' || w === 'angle' || w === 'dihedral' || w === 'improper' || w === 'kspace') {
        throw new StyleError(`fix ${id} (adapt): attribute ${w} is not supported by this engine (pair and atom are)`);
      }
      throw new StyleError(`fix ${id} (adapt): unknown attribute or keyword '${w}' (${USAGE})`);
    }
    if (nattr === 0) throw new StyleError(`fix ${id} (adapt): no attribute given (${USAGE})`);
    // fix_adapt.rst: "The option defaults are scale = no, reset = no, mass = yes."
  }

  private checkVariable(id: string, vw: string): void {
    if (!vw.startsWith('v_')) throw new StyleError(`fix ${id} (adapt): expected v_name, got '${vw}'`);
    const name = vw.slice(2);
    const v = this.sys.vars.get(name);
    if (!v) throw new StyleError(`fix ${id} (adapt): variable ${name} does not exist`);
    if (v.style === 'atom' || v.style === 'atomfile' || v.style === 'vector') {
      throw new StyleError(`fix ${id} (adapt): variable ${name} must be equal-style (atom-style variables are not supported by this engine)`);
    }
  }

  private addPair(id: string, pstyle: string | undefined, pparam: string | undefined, iw: string | undefined, jw: string | undefined, vw: string): void {
    if (pstyle === undefined || pparam === undefined || iw === undefined || jw === undefined) {
      throw new StyleError(`fix ${id} (adapt): pair needs pstyle pparam I J v_name (${USAGE})`);
    }
    if (pstyle.includes(':')) throw new StyleError(`fix ${id} (adapt): pair sub-style selection '${pstyle}' (style:N) requires hybrid, which fix adapt does not support in this engine`);
    if (!ADAPTABLE[pstyle]) {
      throw new StyleError(`fix ${id} (adapt): pair style '${pstyle}' is not supported by fix adapt in this engine (supported: ${Object.keys(ADAPTABLE).join(', ')})`);
    }
    const lname = pparam.toLowerCase();
    if (!ADAPTABLE[pstyle].includes(lname)) {
      throw new StyleError(`fix ${id} (adapt): pair style ${pstyle} parameter '${pparam}' cannot be adapted (supported: ${ADAPTABLE[pstyle].join(', ')})`);
    }
    if (!TYPE_WORD.test(iw) || !TYPE_WORD.test(jw)) throw new StyleError(`fix ${id} (adapt): invalid type pair '${iw} ${jw}' (${USAGE})`);
    this.checkVariable(id, vw);
    // the type ranges and the pair table are resolved at run start (resolve())
    this.pending.push({ pstyle, pparam: lname, vname: vw.slice(2), iw, jw });
  }

  /** Pair attributes as given; resolve() turns them into PairTarget. */
  private readonly pending: Array<{ pstyle: string; pparam: string; vname: string; iw: string; jw: string }> = [];

  /** Resolves pair targets against the defined pair style and the box (run start). */
  private resolve(): void {
    const s = this.sys.state;
    const pair = this.sys.ff.pair;
    this.pairs = [];
    for (const p of this.pending) {
      if (!pair) throw new StyleError(`fix ${this.id} (adapt): no pair style is defined (pair ${p.pstyle})`);
      if (pair.name !== p.pstyle) throw new StyleError(`fix ${this.id} (adapt): pair style ${p.pstyle} is not the defined pair style (${pair.name})`);
      const table = Object.values(pair).find((v): v is PairParams => v instanceof PairParams && v.names.some((n) => n.toLowerCase() === p.pparam));
      if (!table) throw new StyleError(`fix ${this.id} (adapt): pair style ${p.pstyle} has no parameter ${p.pparam}`);
      const name = table.names.find((n) => n.toLowerCase() === p.pparam)!;
      const [ilo, ihi] = typeBounds(p.iw, s.ntypes);
      const [jlo, jhi] = typeBounds(p.jw, s.ntypes);
      const pairs: Array<[number, number]> = [];
      for (let i = ilo; i <= ihi; i++) for (let j = Math.max(jlo, i); j <= jhi; j++) pairs.push([i, j]);
      if (!pairs.length) throw new StyleError(`fix ${this.id} (adapt): no type pairs with I <= J in ${p.iw} ${p.jw}`);
      for (const [i, j] of pairs) {
        if (!table.isSet(i, j)) {
          throw new StyleError(`fix ${this.id} (adapt): type pair ${i} ${j} was not set by pair_coeff; fix adapt changes only explicit coefficients, not mixed ones`);
        }
      }
      this.pairs.push({
        pstyle: p.pstyle, pparam: p.pparam, name, vname: p.vname, iw: p.iw, jw: p.jw,
        pairs, orig: new Float64Array(pairs.length), table, pair,
      });
    }
    for (const a of this.atoms) {
      if (a.kind === 'charge' && !hasChargeStyle(s.atomStyle)) {
        throw new StyleError(`fix ${this.id} (adapt): atom charge needs atom_style charge or full (atom_style is ${s.atomStyle})`);
      }
      if (a.kind === 'diameter' && (!isSphereStyle(s.atomStyle) || !s.radius || !s.rmass)) {
        throw new StyleError(`fix ${this.id} (adapt): atom diameter needs atom_style sphere`);
      }
    }
  }

  /** Stores the original values (run start) and applies the variables for this run's setup. */
  init(): void {
    this.resolve();
    const s = this.sys.state;
    for (const t of this.pairs) {
      t.pairs.forEach(([i, j], k) => { t.orig[k] = t.table.get(t.name, i, j); });
    }
    for (const a of this.atoms) {
      a.origQ.clear(); a.origR.clear(); a.origM.clear();
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const id = s.id[i];
        if (a.kind === 'charge') a.origQ.set(id, s.q[i]);
        else {
          a.origR.set(id, s.radius![i]);
          a.origM.set(id, s.rmass![i]);
        }
      }
    }
    this.inInit = true;
    try {
      this.apply(false);
    } finally {
      this.inInit = false;
    }
  }

  /** Writes the variable values into the pair tables and the atoms (scale yes multiplies the originals). */
  private apply(reinitPair: boolean): void {
    const sys = this.sys;
    const s = sys.state;
    let pairChanged = false;
    for (const t of this.pairs) {
      const v = sys.equalVariable(t.vname);
      const arr = t.table.p(t.name);
      t.pairs.forEach(([i, j], k) => {
        const val = this.scale ? t.orig[k] * v : v;
        arr[t.table.idx(i, j)] = val;
        arr[t.table.idx(j, i)] = val;
      });
      pairChanged = true;
    }
    if (pairChanged && reinitPair) {
      // the style re-derives its tables and re-mixes the non-explicit pairs from the new values
      const pair = this.pairs[0].pair;
      pair.init(sys.styleContext());
      if (pair.tail) sys.ff.updateTail(s);
    }
    // copies of the style in force threads (cpu/pairThreads.ts) must be refreshed
    if (pairChanged) this.pairs[0].pair.version++;
    for (const a of this.atoms) {
      const v = sys.equalVariable(a.vname);
      if (a.kind === 'charge') this.setCharges(a, v, s);
      else this.setDiameter(a, v, s);
    }
    if (!this.inInit && this.atoms.some((a) => a.kind === 'charge')) sys.nb.refreshCharges(s);
  }

  private setCharges(a: AtomTarget, v: number, s: SimState): void {
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      s.q[i] = this.scale ? (a.origQ.get(s.id[i]) ?? s.q[i]) * v : v;
    }
  }

  private setDiameter(a: AtomTarget, v: number, s: SimState): void {
    const radius = s.radius!, rmass = s.rmass!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const d0 = 2 * radius[i];
      const d = this.scale ? 2 * (a.origR.get(s.id[i]) ?? radius[i]) * v : v;
      if (this.massFlag) {
        if (!(d0 > 0)) throw new StyleError(`fix ${this.id} (adapt): diameter change with mass yes needs particles with a nonzero diameter (atom ${s.id[i]}); use mass no`);
        // fix_adapt.rst: "The mass is set from the particle volume for 3d systems (density is assumed to stay constant)."
        rmass[i] *= (d / d0) ** 3;
      }
      radius[i] = 0.5 * d;
    }
  }

  preForce(): void {
    const s = this.sys.state;
    if (this.nevery > 0 && s.step % this.nevery === 0) this.apply(true);
  }

  /**
   * "pre_exchange" is a no-op. The accelerated backends refuse runs with fixes that have
   * neighbor-time hooks (run/accel.ts), so a run with fix adapt stays on the CPU engine,
   * whose force loop reads the adapted tables. (The reason text there mentions neighbor lists.)
   */
  preExchange(): void {}

  /** reset yes: the values the run started with are restored at the end of the run. */
  postRun(): void {
    if (!this.reset) return;
    const s = this.sys.state;
    let pairChanged = false;
    for (const t of this.pairs) {
      const arr = t.table.p(t.name);
      t.pairs.forEach(([i, j], k) => {
        arr[t.table.idx(i, j)] = t.orig[k];
        arr[t.table.idx(j, i)] = t.orig[k];
      });
      pairChanged = true;
    }
    if (pairChanged) {
      const pair = this.pairs[0].pair;
      pair.init(this.sys.styleContext());
      if (pair.tail) this.sys.ff.updateTail(s);
      pair.version++;
    }
    for (const a of this.atoms) {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const id = s.id[i];
        if (a.kind === 'charge') {
          const q = a.origQ.get(id);
          if (q !== undefined) s.q[i] = q;
        } else {
          const r = a.origR.get(id), m = a.origM.get(id);
          if (r !== undefined) s.radius![i] = r;
          if (m !== undefined) s.rmass![i] = m;
        }
      }
    }
    if (this.atoms.some((a) => a.kind === 'charge')) this.sys.nb.refreshCharges(s);
  }
}
