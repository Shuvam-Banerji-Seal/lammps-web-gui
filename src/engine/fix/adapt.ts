import { hasChargeStyle, isSphereStyle } from '../atoms';
import { Fix } from './fix';
import { Pair, StyleError, typeBounds, type AdaptAccess, type Bonded, type BondedAdaptAccess } from '../force/types';
import { ACCEL_ALIASES } from '../accelAliases';
import type { AdaptPairsHook } from './adaptPairs';
import { gplHook } from '../gplHooks';
import { PairHybrid } from '../force/pair/hybrid';
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
 * - reset yes restores the pair and bonded values the run started with (after run 2 with epsilon
 *   set to 2, unfix and run 0 give the epsilon = 1 energy); reset no keeps the last value. The
 *   radii and charges go back to those of the fix's definition (restore_settings, below).
 * - scale yes multiplies the value the parameter had when the run started (a pair_coeff
 *   change before the run is the new original).
 * - a mixed (not pair_coeff-set) type pair targeted by adapt has no effect: the energy is
 *   unchanged; adapting the diagonal epsilon(1,1) re-mixes epsilon(1,2) from the new value
 *   (the same energy as pair_coeff 1 1 2.0 1.0 followed by the mixed 1-2).
 * - atom charge: adapt charge 2.0 on atoms of charge 1 with pair coul/cut gives pe 4x (q^2).
 * - atom diameter 1 -> 2 with atom_style sphere: radius 1, mass 0.5236 -> 4.1888 (d^3);
 *   mass no leaves the mass unchanged.
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
 * b332f28): the style (or hybrid sub-style) extract() names the per-type vector, in native's stored units
 * (fix/adaptBonded.ts), the values of types I are set at setup and every N steps (scale yes multiplies the values the
 * run started with), the defined style is re-initialized, and reset yes restores them after the run. Native writes
 * doubles through integer arrays (angle cosine/periodic, cosine/buck6d multiplicity, dihedral charmm n and d, improper
 * cvff d and n): those parameters are refused.
 *
 * kspace v_name (fix_adapt.rst): "The *kspace* keyword used the specified variable as a scale factor on the energy,
 * forces, virial calculated by whatever k-space solver is defined by the kspace_style command." It sets KSpace.scale.
 *
 * Not supported (StyleError): pair parameters native extract() names but this engine keeps no table for
 * (fix/adaptPairs.ts).
 *
 */

const USAGE = 'usage: fix ID group-ID adapt N attribute args ... keyword value ...';

/** Words of each attribute keyword, as native's first argument loop counts them (src/fix_adapt.cpp at b332f28). */
const ATTRIBUTE_WORDS: Record<string, number> = { pair: 6, kspace: 2, atom: 3, bond: 5, angle: 5, dihedral: 5, improper: 5 };

type BondedKind = 'bond' | 'angle' | 'dihedral' | 'improper';

interface BondedTarget {
  kind: BondedKind;
  bstyle: string;
  bparam: string;
  iw: string;
  vname: string;
  /**
   * Resolved at run start: the defined style (a hybrid style when the parameter is a sub-style's; native reinit()s
   * force->bond), native's per-type vector of the parameter, the types and their values when the run started.
   */
  owner: Bonded | null;
  access: BondedAdaptAccess | null;
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
  /** atom diameter/disc: the mass follows the disc area, d^2 (native discflag, as the last diameter attribute set it). */
  private disc = false;
  /**
   * reset yes: the radii and charges of the group's atoms when the fix was defined, by atom ID (native post_constructor's
   * STORE/ATOM fixes: 0 for atoms outside the group), which restore_settings writes back after each run.
   */
  private readonly defR = new Map<number, number>();
  private readonly defQ = new Map<number, number>();
  /** native previous_diam_scale and previous_chg_scale: the last scale yes value of the run (1 at each init). */
  private prevDiam = 1;
  private prevChg = 1;
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
        // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::FixAdapt at b332f28): diameter and diameter/disc set
        // discflag, any other name is "Unsupported per-atom property X for fix adapt"
        if (kind === 'diameter' || kind === 'diameter/disc') this.disc = kind === 'diameter/disc';
        else if (kind !== 'charge') throw new StyleError(`Unsupported per-atom property ${kind} for fix adapt`);
        this.checkVariable(id, vw);
        this.atoms.push({ kind: kind === 'charge' ? 'charge' : 'diameter', vname: vw.slice(2) });
        k += 3; nattr++;
        continue;
      }
      if (w === 'bond' || w === 'angle' || w === 'dihedral' || w === 'improper') {
        const bstyle = args[k + 1], bparam = args[k + 2], iw = args[k + 3], vw = args[k + 4];
        if (vw === undefined) throw new StyleError(`fix ${id} (adapt): ${w} needs ${w[0]}style ${w[0]}param I v_name (${USAGE})`);
        // native: "Argument must be variable not X"
        if (!vw.startsWith('v_')) throw new StyleError(`Argument must be variable not ${vw}`);
        // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::FixAdapt at b332f28): the type range
        // (utils::bounds_typelabel against the number of types of the kind) when the fix is defined
        const topo = this.sys.state.topo;
        const ntypes = w === 'bond' ? topo.nbondtypes : w === 'angle' ? topo.nangletypes : w === 'dihedral' ? topo.ndihedraltypes : topo.nimpropertypes;
        if (this.sys.labels.defined && !isNumericTypeWord(iw)) inumeric(typeWord(this.sys.labels, w, iw));
        else bounds(iw, 1, ntypes);
        this.checkVariable(id, vw);
        this.bonded.push({ kind: w, bstyle, bparam, iw, vname: vw.slice(2), owner: null, access: null, lo: 0, hi: -1, orig: new Float64Array(0) });
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
    // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::post_constructor at b332f28): with reset yes the radii and
    // charges of the atoms when the fix is defined are stored (0 outside the group) for the restore after each run
    if (this.reset) {
      const s = sys.state;
      for (let i = 0; i < s.n; i++) {
        const inGroup = (s.mask[i] & this.groupBit) !== 0;
        if (this.atoms.some((a) => a.kind === 'diameter') && s.radius) this.defR.set(s.id[i], inGroup ? s.radius[i] : 0);
        if (this.atoms.some((a) => a.kind === 'charge') && hasChargeStyle(s.atomStyle)) this.defQ.set(s.id[i], inGroup ? s.q[i] : 0);
      }
    }
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
    // As native LAMMPS does (src/fix_adapt.cpp FixAdapt::init at b332f28): the per-atom attributes the fix changes,
    // a 2d box for diameter/disc, and previous_diam_scale and previous_chg_scale back to 1
    for (const a of this.atoms) {
      if (a.kind === 'diameter') {
        if (!s.radius) throw new StyleError('Fix adapt requires atom attribute diameter');
        if (!s.rmass) throw new StyleError('Fix adapt requires atom attribute mass');
        if (this.disc && s.dimension !== 2) throw new StyleError('Fix adapt requires 2d simulation');
        this.prevDiam = 1;
      } else {
        if (!hasChargeStyle(s.atomStyle)) throw new StyleError('Fix adapt requires atom attribute charge');
        this.prevChg = 1;
      }
    }
  }

  /**
   * A bonded attribute against the defined style, as native FixAdapt::init does (src/fix_adapt.cpp at b332f28): the
   * style or hybrid sub-style of that name (Force::bond_match and the others: "Fix adapt bond style X does not exist"),
   * a name its extract() knows ("Fix adapt bond style parameter X not supported"), and with a hybrid style the
   * sub-style of every type of the range (check_itype). Where this engine keeps the value: fix/adaptBonded.ts.
   */
  private resolveBonded(b: BondedTarget): void {
    const defined = this.sys.ff[b.kind] as Bonded | null;
    const word = ACCEL_ALIASES.get(b.bstyle) ?? b.bstyle;
    const style = defined instanceof BondedHybrid ? defined.subStyle(word) : defined?.name === word ? defined : null;
    if (!style || !defined) throw new StyleError(`Fix adapt ${b.kind} style ${b.bstyle} does not exist`);
    const { bondedExtractHas, bondedAdaptAccess } = gplHook<AdaptPairsHook>('adapt');
    if (!bondedExtractHas(b.kind, style.name, b.bparam)) throw new StyleError(`Fix adapt ${b.kind} style parameter ${b.bparam} not supported`);
    const ntypes = style.ntypes;
    const [lo, hi] = typeBounds(this.sys.labels.defined && !isNumericTypeWord(b.iw) ? typeWord(this.sys.labels, b.kind, b.iw) : b.iw, ntypes);
    if (defined instanceof BondedHybrid) {
      for (let i = lo; i <= hi; i++) {
        if (defined.subNameOf(i) !== word) throw new StyleError(`Fix adapt type ${b.kind} range is not valid for ${b.kind} hybrid sub-style ${b.bstyle}`);
      }
    }
    b.owner = defined;
    b.access = bondedAdaptAccess(style, b.kind, b.bparam);
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
    this.inInit = true;
    try {
      this.apply(false, false, false, false);
    } finally {
      this.inInit = false;
    }
    // As native LAMMPS does (src/fix_adapt.cpp at b332f28): setup_pre_force changes the per-atom attributes after the
    // force field's init and the setup's neighbor build, so the kspace style chooses g_ewald and its grid from the
    // charges before the change, and the lists are built with the radii before it. afterForceInit writes them.
    this.atomsPending = this.atoms.length > 0;
  }

  /** Per-atom attributes that init() has not written yet (see afterForceInit). */
  private atomsPending = false;

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
    // the bonded values too: native writes them in the same change_settings, so a bonded style's re-init reads what the
    // pair attributes of this and the fixes before it have written (improper amoeba copies the opbend weights of pair
    // amoeba in its init_style)
    for (const b of this.bonded) for (let i = b.lo; i <= b.hi; i++) b.orig[i] = b.access!.get(i);
    this.applyBonded();
    if (!this.atomsPending) return;
    this.atomsPending = false;
    this.applyAtoms(true);
  }

  /** native change_settings for the atom attributes, in the order given; charges reach the force field's copies. */
  private applyAtoms(charges: boolean): void {
    const s = this.sys.state;
    for (const a of this.atoms) {
      if (a.kind === 'charge') {
        if (charges) this.setCharges(this.sys.equalVariable(a.vname), s);
      } else this.setDiameter(this.sys.equalVariable(a.vname), s);
    }
    if (this.atoms.some((a) => a.kind === 'diameter')) this.sys.bump();
    if (charges && this.atoms.some((a) => a.kind === 'charge')) this.sys.nb.refreshCharges(s);
  }

  /**
   * Writes the variable values into the pair tables, the diameters and (when charges is set) the charges
   * (scale yes multiplies the originals).
   */
  private apply(reinitPair: boolean, charges = true, bonded = true, atoms = true): void {
    const sys = this.sys;
    if (this.pairs.length) this.applyPairs(reinitPair);
    if (bonded) this.applyBonded();
    // fix_adapt.cpp change_settings: "*kspace_scale = value" (scale yes does not apply to it)
    for (const vname of this.kspaces) this.sys.ff.kspace!.scale = sys.equalVariable(vname);
    if (atoms) this.applyAtoms(charges && !this.inInit);
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

  /** native change_settings for the bonded attributes: vector[i] = scale ? value * vector_orig[i] : value, then reinit. */
  private applyBonded(): void {
    for (const b of this.bonded) {
      const v = this.sys.equalVariable(b.vname);
      for (let i = b.lo; i <= b.hi; i++) b.access!.set(i, this.scale ? v * b.orig[i] : v);
    }
    this.reinitBonded();
  }

  /**
   * native bond->reinit() (and angle, dihedral, improper) after the values changed, on the defined style: "Fix adapt
   * interface to this bond style not supported" for a style without reinitflag (src/bond.cpp Bond::reinit at b332f28),
   * else its init().
   */
  private reinitBonded(): void {
    const done = new Set<Bonded>();
    for (const b of this.bonded) {
      if (!b.owner || done.has(b.owner)) continue;
      done.add(b.owner);
      if (!b.owner.reinitFlag) throw new StyleError(`Fix adapt interface to this ${b.kind} style not supported`);
      b.owner.init(this.sys.styleContext());
    }
    if (this.bonded.length) this.sys.bump();
  }

  /**
   * As native LAMMPS does (src/fix_adapt.cpp change_settings at b332f28): the charge of the group's atoms set to the
   * value, or with scale yes multiplied by value / previous_chg_scale ("Fix adapt charge scale factor of 0.0 is not
   * supported").
   */
  private setCharges(v: number, s: SimState): void {
    let scale = 1;
    if (this.scale) {
      if (v === 0 || this.prevChg === 0) throw new StyleError('Fix adapt charge scale factor of 0.0 is not supported');
      scale = v / this.prevChg;
    }
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (this.scale) s.q[i] *= scale;
      else s.q[i] = v;
    }
    if (this.scale) this.prevChg = v;
  }

  /**
   * As native LAMMPS does (src/fix_adapt.cpp change_settings at b332f28): the radius of the group's atoms set to half
   * the value, or with scale yes multiplied by value / previous_diam_scale; with mass yes the mass follows at constant
   * density, scale^3 (scale^2 for diameter/disc), the scale of an atom being 0.5 value / radius without scale yes.
   */
  private setDiameter(v: number, s: SimState): void {
    const radius = s.radius!, rmass = s.rmass!;
    let scale = 1;
    if (this.scale) {
      if (v === 0 || this.prevDiam === 0) throw new StyleError('Fix adapt diameter scale factor of 0.0 is not supported');
      scale = v / this.prevDiam;
    }
    if (this.massFlag && (scale === 0 || v === 0)) throw new StyleError('Fix adapt particle mass has become 0.0');
    if (!this.massFlag && (scale < 0 || v < 0)) throw new StyleError('Fix adapt particle diameter has become negative');
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (this.massFlag) {
        if (!this.scale) scale = 0.5 * v / radius[i];
        if (this.disc) rmass[i] *= scale * scale;
        else rmass[i] *= scale * scale * scale;
      }
      if (this.scale) radius[i] *= scale;
      else radius[i] = 0.5 * v;
    }
    if (this.scale) this.prevDiam = v;
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
    for (const b of this.bonded) for (let i = b.lo; i <= b.hi; i++) b.access!.set(i, b.orig[i]);
    this.reinitBonded();
    // fix_adapt.cpp restore_settings: "*kspace_scale = 1.0"
    if (this.kspaces.length && this.sys.ff.kspace) this.sys.ff.kspace.scale = 1.0;
    // As native LAMMPS does (src/fix_adapt.cpp restore_settings at b332f28): the radii and charges stored when the fix
    // was defined. With mass yes the mass is multiplied by the cube (square for diameter/disc) of previous_diam_scale
    // under scale yes, and of stored radius / radius otherwise: under scale yes it grows again rather than returning.
    if (this.atoms.some((a) => a.kind === 'diameter') && s.radius && s.rmass) {
      let scale = this.prevDiam;
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const r0 = this.defR.get(s.id[i]) ?? 0;
        if (this.massFlag) {
          if (!this.scale) scale = r0 / s.radius[i];
          if (this.disc) s.rmass[i] *= scale * scale;
          else s.rmass[i] *= scale * scale * scale;
        }
        s.radius[i] = r0;
      }
      this.sys.bump();
    }
    if (this.atoms.some((a) => a.kind === 'charge')) {
      for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) s.q[i] = this.defQ.get(s.id[i]) ?? 0;
      this.sys.nb.refreshCharges(s);
    }
  }
}
