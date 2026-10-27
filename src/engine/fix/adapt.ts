import { hasChargeStyle, isSphereStyle } from '../atoms';
import { Fix } from './fix';
import { Pair, StyleError, typeBounds, type AdaptAccess, type Bonded } from '../force/types';
import { ACCEL_ALIASES } from '../accelAliases';
import type { AdaptPairsHook } from './adaptPairs';
import { gplHook } from '../gplHooks';
import { PairHybrid } from '../force/pair/hybrid';
import { SimpleBonded } from '../force/bonded_util';
import { bounds, inumeric, missingArgs, logical } from '../force/util';
import { isNumericTypeWord, typeWord } from '../labelmap';
import { EQUAL_STYLES } from '../argInfo';
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
 * As native LAMMPS does (src/fix_adapt.cpp FixAdapt::init and change_settings, src/force.cpp pair_match and
 * src/pair.cpp Pair::reinit at b332f28): pstyle ("style" or "style:N") names the defined pair style or a hybrid
 * sub-style, pparam is looked up in that style's extract() (fix/adaptPairs.ts), the values of the type pairs I <= J of
 * the range are set at setup and every N steps, and the defined pair style is re-initialized (init_one for every type
 * pair, which re-mixes the pairs pair_coeff did not set, so a value written to a mixed pair has no effect; the run's
 * cutoffs stay). The style and parameter are checked when the run starts, after the force field's init.
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
 * coefficient lines at every init here); bonded styles outside BONDED_ADAPT; pair parameters
 * native extract() names but this engine keeps no table for (fix/adaptPairs.ts); diameter/disc.
 *
 */

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

/** A pair attribute as given. */
interface PendingPair { pstyle: string; pparam: string; vname: string; iw: string; jw: string }

interface PairTarget {
  /** pstyle as given ("style" or "style:N") */
  pstyle: string;
  pparam: string;
  vname: string;
  /** The type pairs I <= J of the range (dim 2; native's loops) and their values when the run started. */
  pairs: Array<[number, number]>;
  orig: Float64Array;
  access: AdaptAccess;
  /** The defined pair style (a hybrid style when the parameter is a sub-style's): reinit() runs on it. */
  owner: Pair;
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

/** fix_adapt.cpp: utils::logical. */
const yesNo = (_key: string, w: string | undefined): boolean => logical(w);

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

  /** The variables of the attributes, in the order they were given (checked at init, as native). */
  private readonly varNames: string[] = [];

  private checkVariable(_id: string, vw: string): void {
    // native FixAdapt::FixAdapt (src/fix_adapt.cpp at b332f28), for every attribute: a v_ reference; the variable itself
    // is looked up at init
    if (!vw.startsWith('v_')) throw new StyleError(`Argument must be variable not ${vw}`);
    this.varNames.push(vw.slice(2));
  }

  private addPair(id: string, pstyle: string | undefined, pparam: string | undefined, iw: string | undefined, jw: string | undefined, vw: string): void {
    if (pstyle === undefined || pparam === undefined || iw === undefined || jw === undefined) {
      throw new StyleError(`fix ${id} (adapt): pair needs pstyle pparam I J v_name (${USAGE})`);
    }
    // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::FixAdapt at b332f28): the two type ranges
    // (utils::bounds_typelabel) and the v_ reference are checked when the fix is defined; the pair style and its
    // parameter when the run starts (resolvePairs)
    const sys = this.sys;
    for (const w of [iw, jw]) {
      if (sys.labels.defined && !isNumericTypeWord(w)) inumeric(typeWord(sys.labels, 'atom', w));
      else bounds(w, 1, sys.state.ntypes);
    }
    if (!vw.startsWith('v_')) throw new StyleError(`Argument must be variable not ${vw}`);
    this.checkVariable(id, vw);
    this.pending.push({ pstyle, pparam, vname: vw.slice(2), iw, jw });
  }

  /** Pair attributes as given; resolvePairs() turns them into PairTarget. */
  private readonly pending: PendingPair[] = [];

  /**
   * As native LAMMPS does (src/force.cpp Force::pair_match at b332f28, exact): the defined pair style when its name is
   * pstyle, else the hybrid sub-style of that name (the nsub-th one, or the only one); null when there is none. An
   * accelerator alias (lj/cut/omp) names its base style, as the engine runs it.
   */
  private pairMatch(word: string, nsub: number): { style: Pair; name: string } | null {
    const owner = this.sys.ff.pair;
    if (!owner) return null;
    const base = ACCEL_ALIASES.get(word) ?? word;
    if (owner.name === base) return { style: owner, name: base };
    if (!(owner instanceof PairHybrid)) return null;
    let count = 0, last: Pair | null = null;
    for (const sub of owner.subs) {
      if (sub.name !== base) continue;
      last = sub.style;
      count++;
      if (nsub === count) return { style: sub.style, name: base };
    }
    return count === 1 ? { style: last!, name: base } : null;
  }

  /**
   * The pair attributes against the defined pair style, as native FixAdapt::init does (src/fix_adapt.cpp at b332f28)
   * once the force field is initialized: the style ("Fix adapt pair style {} not found"), the parameter of its
   * extract() (fix/adaptPairs.ts), and with a hybrid pair style the sub-style of every type pair of the range.
   */
  private resolvePairs(): void {
    const owner = this.sys.ff.pair;
    const nt = this.sys.state.ntypes;
    this.pairs = [];
    if (!this.pending.length) return;
    const { pairExtractHas, pairExtractDim, pairAdaptAccess } = gplHook<AdaptPairsHook>('adapt');
    // native's checks of every attribute first (FixAdapt::init), then Pair::reinit's at the setup's first change, then
    // where this engine keeps each value
    const found: Array<{ p: PendingPair; style: Pair; name: string; pairs: Array<[number, number]> }> = [];
    for (const p of this.pending) {
      const c = p.pstyle.indexOf(':');
      const word = c >= 0 ? p.pstyle.slice(0, c) : p.pstyle;
      const nsub = c >= 0 ? inumeric(p.pstyle.slice(c + 1)) : 0;
      const m = this.pairMatch(word, nsub);
      if (!m || !owner) throw new StyleError(`Fix adapt pair style ${word} not found`);
      let style = m.style, name = m.name;
      if (m.style === owner && owner instanceof PairHybrid) {
        // PairHybrid::extract: the first sub-style's value of that name
        const sub = owner.subs.find((q) => pairExtractHas(q.name, p.pparam));
        if (!sub) throw new StyleError(`Fix adapt pair style ${p.pstyle} param ${p.pparam} not supported`);
        style = sub.style; name = sub.name;
      }
      pairExtractDim(name, p.pstyle, p.pparam);
      const [ilo, ihi] = typeBounds(p.iw, nt);
      const [jlo, jhi] = typeBounds(p.jw, nt);
      const pairs: Array<[number, number]> = [];
      for (let i = ilo; i <= ihi; i++) for (let j = Math.max(jlo, i); j <= jhi; j++) pairs.push([i, j]);
      if (owner instanceof PairHybrid) {
        for (const [i, j] of pairs) {
          if (!owner.subsOf(i, j).some((k) => owner.subs[k].name === word)) {
            throw new StyleError(`Fix adapt type pair range is not valid for pair hybrid sub-style ${word}`);
          }
        }
      }
      found.push({ p, style, name, pairs });
    }
    if (!owner || !found.length) return;
    if (!owner.reinitFlag) throw new StyleError(`Fix adapt interface to pair style ${owner.name} is not supported`);
    for (const { p, style, name, pairs } of found) {
      const access: AdaptAccess = pairAdaptAccess(style, name, p.pstyle, p.pparam);
      this.pairs.push({
        pstyle: p.pstyle, pparam: p.pparam, vname: p.vname,
        pairs: access.dim === 2 ? pairs : [], orig: new Float64Array(access.dim === 2 ? pairs.length : 1), access, owner,
      });
    }
  }

  /** Resolves the bonded, kspace and atom attributes (run start; the pair ones follow the force field's init). */
  private resolve(): void {
    const s = this.sys.state;
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
    // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::init at b332f28): each attribute's variable exists and is
    // equal-style; "allow a dynamic group only if ATOM attribute not used"
    for (const name of this.varNames) {
      const v = this.sys.vars.get(name);
      if (!v) throw new StyleError(`Variable name ${name} for fix adapt does not exist`);
      if (!EQUAL_STYLES.has(v.style)) throw new StyleError(`Variable ${name} for fix adapt is invalid style`);
    }
    if (this.atoms.length && this.sys.groups.isDynamic(this.groupBit)) throw new StyleError(`Cannot use dynamic group ${this.group} with fix adapt atom`);
    this.resolve();
    // the pair attributes are resolved and set once the force field is initialized (afterForceInit)
    this.pairs = [];
    const s = this.sys.state;
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
    // As native LAMMPS does (src/fix_adapt.cpp at b332f28): FixAdapt::init after the force field's init stores the
    // values the run starts with, and setup_pre_force's change_settings sets them and re-initializes the pair style
    this.resolvePairs();
    for (const t of this.pairs) {
      if (t.access.dim === 2) {
        const a = t.access;
        t.pairs.forEach(([i, j], k) => { t.orig[k] = a.get(i, j); });
      } else t.orig[0] = t.access.get();
    }
    if (this.pairs.length) this.applyPairs(true);
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
    if (this.pairs.length) this.applyPairs(reinitPair);
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
   * native change_settings for the pair attributes: each value (scale yes: times the run's starting value) into the
   * type pairs of its range, then the defined pair style re-initialized (Pair::reinit).
   */
  private applyPairs(reinitPair: boolean): void {
    for (const t of this.pairs) {
      const v = this.sys.equalVariable(t.vname);
      if (t.access.dim === 0) t.access.set(this.scale ? v * t.orig[0] : v);
      else {
        const a = t.access;
        t.pairs.forEach(([i, j], k) => a.set(i, j, this.scale ? v * t.orig[k] : v));
      }
    }
    this.refreshPairs(reinitPair);
    this.sys.ff.pairStale = true;
  }

  /**
   * After the values changed: the defined pair style re-initialized as native's Pair::reinit (re-derived coefficients,
   * re-mixed pairs, the run's cutoffs kept), the tail corrections summed again, and the version bumped so that copies of
   * the style in force threads are refreshed.
   */
  private refreshPairs(reinitPair: boolean): void {
    const s = this.sys.state;
    const owners = new Set<Pair>();
    for (const t of this.pairs) owners.add(t.owner);
    for (const owner of owners) {
      if (reinitPair) {
        owner.reinit();
        if (hasTail(owner)) this.sys.ff.reinitTail(s);
      }
      owner.version++;
    }
  }

  private commitPairs(): void {
    for (const t of this.pairs) {
      if (t.access.dim === 0 && t.access.commit?.()) this.sys.ff.heldPairChange = `${t.pstyle} ${t.pparam}`;
    }
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

  /**
   * reset yes: the values the run started with are restored at the end of the run. Then a pair value native reads only
   * when a run initializes, which the access held during the run, is written to the style (AdaptAccess commit).
   */
  postRun(): void {
    if (!this.reset) {
      this.commitPairs();
      return;
    }
    const s = this.sys.state;
    for (const t of this.pairs) {
      if (t.access.dim === 0) t.access.set(t.orig[0]);
      else {
        const a = t.access;
        t.pairs.forEach(([i, j], k) => a.set(i, j, t.orig[k]));
      }
    }
    if (this.pairs.length) this.refreshPairs(true);
    this.commitPairs();
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
