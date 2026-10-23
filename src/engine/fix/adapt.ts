import { hasChargeStyle, isSphereStyle } from '../atoms';
import { Fix } from './fix';
import { Pair, PairParams, StyleError, typeBounds, type Bonded } from '../force/types';
import { PairHybrid } from '../force/pair/hybrid';
import { SimpleBonded } from '../force/bonded_util';
import { bounds, inumeric, missingArgs } from '../force/util';
import { isNumericTypeWord, typeWord } from '../labelmap';
import { BondedHybrid } from '../force/bonded_hybrid';
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
 * Sub-style selection (fix_adapt.rst): "If there are multiple sub-styles using the same pair
 * style, then *pstyle* should be specified as "style:N", where *N* is which instance of the
 * pair style you wish to adapt (e.g., the first or second)." The engine numbers the instances
 * in pair_style order, as the hybrid style does.
 *
 * Measured with native LAMMPS (black box), 2x2x2 fcc Ni (32 atoms, random displacements), pair_style
 * hybrid/overlay snap snap with the bundled Ni_Zuo_JPCA2020 linear and quadratic files:
 * - snap:1 scale * * 0.3 gives the energy 0.3 E1 + E2 (E1 = the linear sub-style alone, E2 = the
 *   quadratic sub-style alone), and snap:2 scale * * 0.7 gives E1 + 0.7 E2; the pressure follows the
 *   same factors (the virial of the sub-style scales with its energy).
 * - two types, scale 1 1 0.5 scales the energy of the type-1 atoms only; scale 2 2 0.5 the type-2
 *   atoms only; scale * * 0.5 both; scale 1 2 0.5 has no effect at all (energy, pressure and
 *   off-diagonal pressure are unchanged). So only the diagonal entry of the central atom's type is used.
 *
 * Bonded attributes — fix_adapt.rst: "*bond* args = bstyle bparam I v_name", "*angle* args = astyle aparam I
 * v_name", and the same for *dihedral* and *improper*; "Note that internally, theta0 is stored in radians, so the
 * variable should compute theta0 in radians" (likewise phi0, chi0). As native LAMMPS does (src/fix_adapt.cpp at
 * b332f28): the style's extract() names the per-type array (BONDED_ADAPT gives native's names, the engine's array
 * and native's stored value per engine value: dihedral opls stores 0.5 K, improper inversion/harmonic K/3), the
 * values of types I are set at setup and every N steps (scale yes multiplies the values the run started with), the
 * style is re-initialized, and reset yes restores them after the run. Native writes doubles through integer arrays
 * (dihedral charmm n and d, improper cvff d and n): those parameters are refused, as the doc table lists only k.
 *
 * kspace v_name (fix_adapt.rst): "The *kspace* keyword used the specified variable as a scale factor on the energy,
 * forces, virial calculated by whatever k-space solver is defined by the kspace_style command." It sets KSpace.scale.
 *
 * Not supported (StyleError): bonded hybrid styles (their sub-styles are rebuilt from the
 * coefficient lines at every init here); bonded styles outside BONDED_ADAPT; pair
 * styles and parameters other than those listed in ADAPTABLE (a cutoff is never adaptable:
 * the neighbor list would not follow it); mixed type pairs; diameter/disc; sub-style
 * selection for a sub-style name that is not in the hybrid style.
 *
 */

/**
 * Pair styles and parameters (lower-case, as the fix_adapt.rst table names them) this engine can
 * adapt; "cut" names are not adaptable. The engine's table name is matched case-insensitively
 * (a -> A, c -> C, d -> D). reinit: the style derives tables from the parameter at init() and must
 * be re-initialized after a change; false when compute() reads the parameter table directly.
 */
interface Adaptable {
  params: string[];
  reinit: boolean;
  /** Parameters that are listed by the doc table but refused, with the reason (measured below). */
  refused?: Record<string, string>;
}
const ADAPTABLE: Record<string, Adaptable> = {
  'lj/cut': { params: ['epsilon', 'sigma'], reinit: true },
  'lj/cut/coul/cut': { params: ['epsilon', 'sigma'], reinit: true },
  'lj/class2': { params: ['epsilon', 'sigma'], reinit: true },
  'lj/expand': {
    params: ['epsilon', 'sigma', 'delta'],
    reinit: true,
    // Measured with native LAMMPS (black box): with pair_coeff 1 1 1.0 1.0 0.5 and fix adapt delta 1 1 0.14
    // the pe is 92.7189 (the energy uses delta 0.14 but the force cutoff stays cut + 0.5 = 3.0); with
    // pair_coeff delta 0.1 and an adapted 0.14 it is 92.7497 (cutoff 2.6); static delta 0.14 gives 92.7420.
    // The cutoff cut + delta is frozen at the value the run started with, which this engine does not model.
    refused: { delta: 'changes the force cutoff cut + delta, which native LAMMPS keeps at its run-start value (not modelled by this engine)' },
  },
  buck: { params: ['a', 'c'], reinit: true },
  born: { params: ['a', 'c', 'd'], reinit: true },
  soft: { params: ['a'], reinit: true },
  morse: { params: ['d0', 'alpha', 'r0'], reinit: true },
  snap: { params: ['scale'], reinit: false },
};

const USAGE = 'usage: fix ID group-ID adapt N attribute args ... keyword value ...';

/** Words of each attribute keyword, as native's first argument loop counts them (src/fix_adapt.cpp at b332f28). */
const ATTRIBUTE_WORDS: Record<string, number> = { pair: 6, kspace: 2, atom: 3, bond: 5, angle: 5, dihedral: 5, improper: 5 };

type BondedKind = 'bond' | 'angle' | 'dihedral' | 'improper';
const KIND_LABEL: Record<BondedKind, string> = { bond: 'Bond', angle: 'Angle', dihedral: 'Dihedral', improper: 'Improper' };

/**
 * Bonded styles fix adapt can change: native extract() name -> [the engine's parameter, native's stored value per
 * engine value]. Read from the extract() and coeff() functions of each style (src/*.cpp at b332f28).
 */
const BONDED_ADAPT: Record<string, Record<string, [string, number]>> = {
  'bond harmonic': { k: ['K', 1], r0: ['r0', 1] },
  'bond morse': { d0: ['D', 1], alpha: ['alpha', 1], r0: ['r0', 1] },
  'bond fene': { k: ['K', 1], r0: ['R0', 1] },
  'bond fene/expand': { k: ['K', 1], r0: ['R0', 1], epsilon: ['epsilon', 1], sigma: ['sigma', 1], shift: ['delta', 1] },
  'bond gromos': { k: ['K', 1], r0: ['r0', 1] },
  'bond nonlinear': { lamda: ['lam', 1], epsilon: ['epsilon', 1], r0: ['r0', 1] },
  'bond class2': { k2: ['K2', 1], k3: ['K3', 1], k4: ['K4', 1], r0: ['r0', 1] },
  'angle harmonic': { k: ['K', 1], theta0: ['theta0', 1] },
  'angle charmm': { k: ['K', 1], theta0: ['theta0', 1] },
  'angle cosine': { k: ['K', 1] },
  'angle cosine/squared': { k: ['K', 1], theta0: ['theta0', 1] },
  'angle cosine/squared/restricted': { k: ['K', 1], theta0: ['theta0', 1] },
  'angle fourier': { k: ['K', 1], C0: ['C0', 1], C1: ['C1', 1], C2: ['C2', 1] },
  'angle fourier/simple': { k: ['K', 1], C: ['c', 1], N: ['n', 1] },
  'angle quartic': { k2: ['K2', 1], k3: ['K3', 1], k4: ['K4', 1], theta0: ['theta0', 1] },
  'angle mm3': { k2: ['K', 1], theta0: ['theta0', 1] },
  'dihedral charmm': { k: ['K', 1] },
  'dihedral opls': { k1: ['K1', 0.5], k2: ['K2', 0.5], k3: ['K3', 0.5], k4: ['K4', 0.5] },
  'dihedral multi/harmonic': { a1: ['A1', 1], a2: ['A2', 1], a3: ['A3', 1], a4: ['A4', 1], a5: ['A5', 1] },
  'dihedral quadratic': { k: ['K', 1], phi0: ['phi0', 1] },
  'dihedral helix': { a: ['A', 1], b: ['B', 1], c: ['C', 1] },
  'dihedral cosine/squared/restricted': { k: ['K', 1], phi0: ['phi0', 1] },
  'improper harmonic': { k: ['K', 1], chi0: ['chi0', 1] },
  'improper cvff': { k: ['K', 1] },
  'improper fourier': { k: ['K', 1], C0: ['C0', 1], C1: ['C1', 1], C2: ['C2', 1] },
  'improper cossq': { k: ['K', 1], chi0: ['chi0', 1] },
  'improper distance': { k2: ['K2', 1], k4: ['K4', 1] },
  'improper distharm': { k: ['K', 1], d0: ['d0', 1] },
  'improper sqdistharm': { k: ['K', 1] },
  'improper inversion/harmonic': { k: ['K', 1 / 3], w0: ['omega0', 1] },
};

interface BondedTarget {
  kind: BondedKind;
  bstyle: string;
  bparam: string;
  iw: string;
  vname: string;
  /** Resolved at run start: the style, its array, native's value per engine value, the types and their originals. */
  style: Bonded | null;
  arr: Float64Array | null;
  ratio: number;
  lo: number;
  hi: number;
  orig: Float64Array;
}

interface PairTarget {
  /** Sub-style name (or the pair style name), as given without the ":N" instance. */
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
  /** The (sub-)style that owns the table. */
  style: Pair;
  /** The defined pair style: a hybrid style when the sub-style is in one; init() runs on it. */
  owner: Pair;
  reinit: boolean;
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

/** True when the style or one of its hybrid sub-styles applies a tail correction. */
const hasTail = (p: Pair): boolean => {
  if (p.tail) return true;
  return p instanceof PairHybrid && p.subs.some((x) => x.style.tail);
};

export class FixAdapt extends Fix {
  readonly style: string = 'adapt';
  private pairs: PairTarget[] = [];
  private readonly bonded: BondedTarget[] = [];
  private readonly atoms: AtomTarget[] = [];
  /** kspace v_name: the variables that set the kspace scale factor (native KSpace::scale) */
  private readonly kspaces: string[] = [];
  private scale = false;
  private reset = false;
  private massFlag = true;
  /** True while init() applies values (before the force field is initialized). */
  private inInit = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::FixAdapt at b332f28): utils::inumeric reads the every value,
    // and a negative one is refused with "Illegal fix adapt every value".
    const n = inumeric(args[0]);
    if (n < 0) throw new StyleError(`Illegal fix adapt every value ${n}`);
    this.nevery = n;
    // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::FixAdapt at b332f28): the leading attribute keywords are
    // counted first, each checked for its own number of arguments ("Illegal fix adapt pair command: missing argument(s)")
    for (let p = 1; p < args.length; ) {
      const need = ATTRIBUTE_WORDS[args[p]];
      if (typeof need !== 'number') break;
      if (p + need > args.length) throw missingArgs(`fix adapt ${args[p]}`);
      p += need;
    }
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
      if (w === 'bond' || w === 'angle' || w === 'dihedral' || w === 'improper') {
        const bstyle = args[k + 1], bparam = args[k + 2], iw = args[k + 3], vw = args[k + 4];
        if (vw === undefined) throw new StyleError(`fix ${id} (adapt): ${w} needs ${w[0]}style ${w[0]}param I v_name (${USAGE})`);
        // native: "Argument must be variable not X"
        if (!vw.startsWith('v_')) throw new StyleError(`Argument must be variable not ${vw}`);
        this.checkVariable(id, vw);
        this.bonded.push({ kind: w, bstyle, bparam, iw, vname: vw.slice(2), style: null, arr: null, ratio: 1, lo: 0, hi: -1, orig: new Float64Array(0) });
        k += 5; nattr++;
        continue;
      }
      if (w === 'kspace') {
        // fix_adapt.cpp at b332f28: "Argument must be variable not {}"
        const vw = args[k + 1];
        if (vw === undefined) throw new StyleError(`fix ${id} (adapt): kspace needs v_name (${USAGE})`);
        if (!vw.startsWith('v_')) throw new StyleError(`Argument must be variable not ${vw}`);
        this.checkVariable(id, vw);
        this.kspaces.push(vw.slice(2));
        k += 2; nattr++;
        continue;
      }
      // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::FixAdapt at b332f28): a word that is neither an attribute nor a keyword reaches the keyword loop's "Unknown fix adapt keyword {}"
      throw new StyleError(`Unknown fix adapt keyword ${w}`);
    }
    if (nattr === 0) throw new StyleError(`fix ${id} (adapt): no attribute given (${USAGE})`);
    // fix_adapt.rst: "The option defaults are scale = no, reset = no, mass = yes."
  }

  private checkVariable(id: string, vw: string): void {
    // native FixAdapt::FixAdapt (src/fix_adapt.cpp at b332f28), for every attribute
    if (!vw.startsWith('v_')) throw new StyleError(`Argument must be variable not ${vw}`);
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
    // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::FixAdapt at b332f28): the two type ranges
    // (utils::bounds_typelabel) and the v_ reference are checked when the fix is defined; the pair style and its
    // parameter when the run starts (the engine checks them next)
    const sys = this.sys;
    for (const w of [iw, jw]) {
      if (sys.labels.defined && !isNumericTypeWord(w)) inumeric(typeWord(sys.labels, 'atom', w));
      else bounds(w, 1, sys.state.ntypes);
    }
    if (!vw.startsWith('v_')) throw new StyleError(`Argument must be variable not ${vw}`);
    const m = /^([^:]+)(?::(\d+))?$/.exec(pstyle);
    if (!m) throw new StyleError(`fix ${id} (adapt): pair style selection '${pstyle}' must be style or style:N (${USAGE})`);
    const base = m[1];
    const inst = m[2] === undefined ? undefined : Number(m[2]);
    if (inst !== undefined && (!Number.isInteger(inst) || inst < 1)) {
      throw new StyleError(`fix ${id} (adapt): instance N of pair style selection '${pstyle}' must be 1 or more`);
    }
    const entry = ADAPTABLE[base];
    if (!entry) {
      throw new StyleError(`fix ${id} (adapt): pair style '${base}' is not supported by fix adapt in this engine (supported: ${Object.keys(ADAPTABLE).join(', ')})`);
    }
    const lname = pparam.toLowerCase();
    if (entry.refused?.[lname]) throw new StyleError(`fix ${id} (adapt): pair style ${base} parameter '${pparam}' is not supported: ${entry.refused[lname]}`);
    if (!entry.params.includes(lname)) {
      throw new StyleError(`fix ${id} (adapt): pair style ${base} parameter '${pparam}' cannot be adapted (supported: ${entry.params.join(', ')})`);
    }
    this.checkVariable(id, vw);
    // the type ranges and the pair table are resolved at run start (resolve())
    this.pending.push({ pstyle: base, inst, pparam: lname, vname: vw.slice(2), iw, jw, reinit: entry.reinit });
  }

  /** Pair attributes as given; resolve() turns them into PairTarget. */
  private readonly pending: Array<{ pstyle: string; inst: number | undefined; pparam: string; vname: string; iw: string; jw: string; reinit: boolean }> = [];

  /** Resolves pair targets against the defined pair style (or its hybrid sub-styles) and the box (run start). */
  private resolve(): void {
    const s = this.sys.state;
    const owner = this.sys.ff.pair;
    this.pairs = [];
    for (const p of this.pending) {
      if (!owner) throw new StyleError(`fix ${this.id} (adapt): no pair style is defined (pair ${p.pstyle})`);
      let style: Pair;
      if (owner instanceof PairHybrid) {
        const subs = owner.subs.filter((x) => x.name === p.pstyle);
        if (!subs.length) throw new StyleError(`fix ${this.id} (adapt): pair sub-style ${p.pstyle} is not in pair_style ${owner.name}`);
        if (p.inst === undefined && subs.length > 1) {
          throw new StyleError(`fix ${this.id} (adapt): pair sub-style ${p.pstyle} is used ${subs.length} times; select one with ${p.pstyle}:N`);
        }
        const sub = p.inst === undefined ? subs[0] : subs.find((x) => x.instance === p.inst);
        if (!sub) throw new StyleError(`fix ${this.id} (adapt): pair sub-style ${p.pstyle}:${p.inst} does not exist (it is used ${subs.length} times)`);
        style = sub.style;
      } else {
        if (owner.name !== p.pstyle) throw new StyleError(`fix ${this.id} (adapt): pair style ${p.pstyle} is not the defined pair style (${owner.name})`);
        if (p.inst !== undefined && p.inst !== 1) throw new StyleError(`fix ${this.id} (adapt): pair style ${p.pstyle}:${p.inst} does not exist (the pair style is not a hybrid style)`);
        style = owner;
      }
      const table = Object.values(style).find((v): v is PairParams => v instanceof PairParams && v.names.some((n) => n.toLowerCase() === p.pparam));
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
        pairs, orig: new Float64Array(pairs.length), table, style, owner, reinit: p.reinit,
      });
    }
    for (const b of this.bonded) this.resolveBonded(b);
    if (this.kspaces.length && !this.sys.ff.kspace) throw new StyleError('Fix adapt expected a kspace style but none was defined');
    // As native LAMMPS does (src/atom_vec_sphere.cpp init at b332f28): changing diameters needs atom_style sphere 1
    if (this.atoms.some((a) => a.kind !== 'charge') && isSphereStyle(s.atomStyle) && !this.sys.radvary) {
      throw new StyleError('Fix adapt changes atom radii but atom_style sphere is not dynamic');
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

  /** A bonded attribute against the defined style (native FixAdapt::init with Force::bond_match and extract()). */
  private resolveBonded(b: BondedTarget): void {
    const style = this.sys.ff[b.kind] as Bonded | null;
    const label = KIND_LABEL[b.kind].toLowerCase();
    if (style instanceof BondedHybrid) {
      throw new StyleError(`fix ${this.id} (adapt): ${b.kind} with ${b.kind}_style hybrid is not supported by this engine (its sub-styles are rebuilt from the coefficient lines at every init)`);
    }
    if (!style || style.name !== b.bstyle) throw new StyleError(`Fix adapt ${label} style ${b.bstyle} does not exist`);
    const entry = BONDED_ADAPT[`${b.kind} ${b.bstyle}`];
    if (!entry) throw new StyleError(`fix ${this.id} (adapt): ${b.kind} style ${b.bstyle} is not supported by fix adapt in this engine`);
    const m = entry[b.bparam];
    if (!m || !(style instanceof SimpleBonded)) throw new StyleError(`Fix adapt ${label} style parameter ${b.bparam} not supported`);
    const ntypes = style.ntypes;
    const [lo, hi] = typeBounds(b.iw, ntypes);
    b.style = style;
    b.arr = style.params.p(m[0]);
    b.ratio = m[1];
    b.lo = lo;
    b.hi = hi;
    b.orig = new Float64Array(ntypes + 1);
  }

  /** Stores the original values (run start) and applies the variables for this run's setup. */
  init(): void {
    // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::init at b332f28): "allow a dynamic group only if ATOM attribute
    // not used"
    if (this.atoms.length && this.sys.groups.isDynamic(this.groupBit)) throw new StyleError(`Cannot use dynamic group ${this.group} with fix adapt atom`);
    this.resolve();
    const s = this.sys.state;
    for (const t of this.pairs) {
      t.pairs.forEach(([i, j], k) => { t.orig[k] = t.table.get(t.name, i, j); });
    }
    for (const b of this.bonded) for (let i = b.lo; i <= b.hi; i++) b.orig[i] = b.arr![i];
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
      this.apply(false, false);
    } finally {
      this.inInit = false;
    }
    // Measured with native LAMMPS (black box): the kspace style chooses g_ewald and its grid or kmax from the
    // charges present at its init, and fix adapt changes the charges only at setup, after that init. The
    // charges are therefore written by afterForceInit, once the force field is initialized.
    this.chargesPending = this.atoms.some((a) => a.kind === 'charge');
  }

  /** Charges that init() has not written yet (see afterForceInit). */
  private chargesPending = false;

  /** Setup: the charges of this run's setup are written after the force field's init (kspace init). */
  afterForceInit(): void {
    if (!this.chargesPending) return;
    this.chargesPending = false;
    const s = this.sys.state;
    for (const a of this.atoms) {
      if (a.kind === 'charge') this.setCharges(a, this.sys.equalVariable(a.vname), s);
    }
    this.sys.nb.refreshCharges(s);
  }

  /**
   * Writes the variable values into the pair tables, the diameters and (when charges is set) the charges
   * (scale yes multiplies the originals).
   */
  private apply(reinitPair: boolean, charges = true): void {
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
    if (pairChanged) {
      this.refreshPairs(reinitPair);
      sys.ff.pairStale = true;
    }
    for (const b of this.bonded) {
      // native: vector[i] = scale ? value * vector_orig[i] : value, in native's stored units (see BONDED_ADAPT)
      const v = sys.equalVariable(b.vname);
      for (let i = b.lo; i <= b.hi; i++) b.arr![i] = this.scale ? v * b.orig[i] : v / b.ratio;
    }
    this.reinitBonded();
    // fix_adapt.cpp change_settings: "*kspace_scale = value" (scale yes does not apply to it)
    for (const vname of this.kspaces) this.sys.ff.kspace!.scale = sys.equalVariable(vname);
    for (const a of this.atoms) {
      if (a.kind === 'charge') {
        if (charges) this.setCharges(a, sys.equalVariable(a.vname), s);
      } else this.setDiameter(a, sys.equalVariable(a.vname), s);
    }
    if (!this.inInit && charges && this.atoms.some((a) => a.kind === 'charge')) sys.nb.refreshCharges(s);
  }

  /**
   * After the tables changed: re-initializes each defined pair style that derives tables from the
   * changed parameters (the style re-mixes its non-explicit pairs), updates the tail corrections,
   * and bumps the version so that copies of the style in force threads are refreshed.
   */
  private refreshPairs(reinitPair: boolean): void {
    const s = this.sys.state;
    const owners = new Set<Pair>();
    for (const t of this.pairs) owners.add(t.owner);
    if (reinitPair) {
      const reinit = new Set<Pair>();
      for (const t of this.pairs) if (t.reinit) reinit.add(t.owner);
      for (const owner of reinit) {
        owner.init(this.sys.styleContext());
        if (hasTail(owner)) this.sys.ff.reinitTail(s);
      }
    }
    for (const owner of owners) owner.version++;
  }

  /** native bond->reinit() (and angle, dihedral, improper) after the values changed: the style's init(). */
  private reinitBonded(): void {
    const done = new Set<Bonded>();
    for (const b of this.bonded) {
      if (!b.style || done.has(b.style)) continue;
      done.add(b.style);
      b.style.init(this.sys.styleContext());
    }
    if (this.bonded.length) this.sys.bump();
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
    if (pairChanged) this.refreshPairs(true);
    for (const b of this.bonded) for (let i = b.lo; i <= b.hi; i++) b.arr![i] = b.orig[i];
    this.reinitBonded();
    // fix_adapt.cpp restore_settings: "*kspace_scale = 1.0"
    if (this.kspaces.length && this.sys.ff.kspace) this.sys.ff.kspace.scale = 1.0;
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
