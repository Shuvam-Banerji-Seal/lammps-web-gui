import { Bonded, newAccum, StyleError, typeBounds, type Accum, type BondedCompute, type Pair, type StyleContext } from './types';
import type { TopoList } from '../types';

/*
 * bond_style hybrid, angle_style hybrid, dihedral_style hybrid and
 * improper_style hybrid — docs.lammps.org/bond_hybrid.html (and the angle,
 * dihedral and improper pages). "A bond style is assigned to each bond type."
 * Each *_coeff line names its sub-style after the type (bond_coeff.html).
 *
 * Sub-styles are instances of the registry's styles. Each one is handed only
 * the topology entries whose type it owns (a shallow copy of the state with
 * that kind's list replaced), so every sub-style keeps its own loop unchanged.
 * Energy is tallied per sub-style into a scratch accumulator and added to the
 * total; the per-sub-style values feed compute bond, angle, dihedral and improper.
 *
 * Measured with native LAMMPS (black box):
 *  - a sub-style may not repeat (the message names the kind, e.g. Bond style
 *    hybrid cannot use same bond style twice; the other kinds follow the same pattern);
 *  - a settings word that is not a style name is refused: harmonic 1.0 gives
 *    Illegal bond_style hybrid argument: 1.0 (also for the other kinds);
 *  - a coeff line naming a style that is not listed gives Expected hybrid
 *    sub-style instead of foo in bond_coeff command (the kind word differs for
 *    the other kinds);
 *  - a listed sub-style that no type uses stops the run with Bond hybrid
 *    sub-style fene is not used (likewise Angle, Dihedral and Improper);
 *  - a type left without coefficients stops the run with All bond coeffs are not set;
 *  - none is accepted in coeff lines for every kind; for angle, dihedral and
 *    improper the type then gets no energy; for bond a none type stops every run
 *    with Invoked bond equil distance on bond style none, whether or not bonds of
 *    that type exist;
 *  - skip in an angle, dihedral or improper coeff line leaves the type's earlier
 *    setting in place (the type stays unset when there was none); bond refuses it;
 *  - re-issuing bond_style hybrid clears every coefficient, as a new style does.
 */

export type HybridKind = 'bond' | 'angle' | 'dihedral' | 'improper';
/** The pair style and special_bonds weights a bonded style may need (Bonded.linkForceField). */
type Link = { pair: Pair | null; special: { lj: readonly number[]; coul: readonly number[] } };
type TopoKey = 'bonds' | 'angles' | 'dihedrals' | 'impropers';
type EnergyKey = 'ebond' | 'eangle' | 'edihed' | 'eimp';

const LABEL: Record<HybridKind, string> = { bond: 'Bond', angle: 'Angle', dihedral: 'Dihedral', improper: 'Improper' };
const LIST_KEY: Record<HybridKind, TopoKey> = { bond: 'bonds', angle: 'angles', dihedral: 'dihedrals', improper: 'impropers' };
const ENERGY_KEY: Record<HybridKind, EnergyKey> = { bond: 'ebond', angle: 'eangle', dihedral: 'edihed', improper: 'eimp' };

/** Per-type marks in the assignment table (values >= 0 are sub-style indices). */
const UNSET = -1;
const NONE = -2;

const NO_CONTEXT: StyleContext = {
  s: null,
  readFile: () => { throw new StyleError('hybrid sub-style: no file access in this context'); },
  log: () => {},
};

/** Sub-style names that look like style names (letters first): unknown ones are reported as not supported. */
const looksLikeStyle = (w: string): boolean => /^[a-zA-Z]/.test(w);

export class BondedHybrid extends Bonded {
  readonly name = 'hybrid';
  readonly kind: HybridKind;
  /** Sub-style names in the order they were listed. */
  subNames: string[] = [];
  /** Per type: sub-style index, UNSET or NONE. */
  private assign = new Int32Array(0);
  /** Per type: the words after the sub-style name in its last coeff line (null for none). */
  private words: (string[] | null)[] = [];
  /** Sub-style instances built from the current coefficients (null when they must be rebuilt). */
  private subs: Bonded[] | null = null;
  /** Energy of each sub-style at the last force evaluation. */
  private energy = new Float64Array(0);
  private ctx: StyleContext | null = null;
  private link: Link | null = null;
  private readonly registry: Record<string, () => Bonded>;
  private readonly scratch: Accum = newAccum();

  constructor(kind: HybridKind, registry: Record<string, () => Bonded>) {
    super();
    this.kind = kind;
    this.registry = registry;
  }

  settings(args: string[], ctx?: StyleContext): void {
    if (ctx) this.ctx = ctx;
    const kind = this.kind;
    if (!args.length) throw new StyleError(`Illegal ${kind}_style hybrid command: missing argument(s)`);
    const names: string[] = [];
    args.forEach((a, i) => {
      if (a === 'hybrid') throw new StyleError(`${LABEL[kind]} style hybrid cannot have hybrid as an argument`);
      if (a === 'none') throw new StyleError(`${LABEL[kind]} style hybrid cannot have none as an argument`);
      if (names.includes(a)) throw new StyleError(`${LABEL[kind]} style hybrid cannot use same ${kind} style twice`);
      if (!this.registry[a]) {
        if (i > 0 && !looksLikeStyle(a)) throw new StyleError(`Illegal ${kind}_style hybrid argument: ${a}`);
        throw new StyleError(`${kind}_style hybrid sub-style '${a}' is not supported by the browser engine`);
      }
      names.push(a);
    });
    // a sub-style with required settings (e.g. lepton expressions) cannot sit in a hybrid list
    for (const n of names) this.make(n, this.ctx);
    this.subNames = names;
    this.energy = new Float64Array(names.length);
    this.subs = null;
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.assign = new Int32Array(ntypes + 1).fill(UNSET);
    this.words = new Array<string[] | null>(ntypes + 1).fill(null);
    this.subs = null;
  }

  /** The sub-style of each *_coeff line: N style args, or N none, or N skip (angle, dihedral, improper). */
  coeff(args: string[], ctx?: StyleContext): void {
    const kind = this.kind;
    if (ctx) this.ctx = ctx;
    if (args.length < 2) throw new StyleError(`usage: ${kind}_coeff N style coefficients (hybrid sub-style)`);
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const name = args[1];
    const rest = args.slice(2);
    if (name === 'none') {
      if (rest.length) throw new StyleError(`${kind}_coeff ${args[0]} none takes no coefficients`);
      for (let t = lo; t <= hi; t++) { this.assign[t] = NONE; this.words[t] = null; }
    } else if (name === 'skip' && kind !== 'bond') {
      // measured: a skip line changes nothing for its types (see the header comment)
      if (rest.length) throw new StyleError(`${kind}_coeff ${args[0]} skip takes no coefficients`);
      return;
    } else {
      const k = this.subNames.indexOf(name);
      if (k < 0) throw new StyleError(`Expected hybrid sub-style instead of ${name} in ${kind}_coeff command`);
      // the words are validated by a scratch instance of the sub-style
      const probe = this.make(name, this.ctx);
      probe.allocate(this.ntypes);
      probe.coeff([args[0], ...rest], this.ctx ?? NO_CONTEXT);
      for (let t = lo; t <= hi; t++) { this.assign[t] = k; this.words[t] = rest; }
    }
    this.subs = null;
  }

  linkForceField(link: Link): void {
    this.link = link;
    if (this.subs) for (const st of this.subs) st.linkForceField?.(link);
  }

  /** Setup: every type needs coefficients, every sub-style must be used, and bond none types stop the run. */
  init(ctx?: StyleContext): void {
    if (ctx) this.ctx = ctx;
    const kind = this.kind;
    for (let t = 1; t <= this.ntypes; t++) {
      if (this.assign[t] === UNSET) throw new StyleError(`All ${kind} coeffs are not set (type ${t})`);
    }
    for (let k = 0; k < this.subNames.length; k++) {
      if (!this.usedBy(k)) throw new StyleError(`${LABEL[kind]} hybrid sub-style ${this.subNames[k]} is not used`);
    }
    if (kind === 'bond' && this.hasNone()) throw new StyleError('Invoked bond equil distance on bond style none');
    this.subs = null;
    const subs = this.build();
    const c = this.ctx ?? NO_CONTEXT;
    for (const st of subs) st.init(c);
  }

  compute(bc: BondedCompute): void {
    const subs = this.build();
    const key = ENERGY_KEY[this.kind];
    const listKey = LIST_KEY[this.kind];
    const buckets = this.bucket(bc.s.topo[listKey]);
    for (let k = 0; k < subs.length; k++) {
      this.energy[k] = 0;
      const list = buckets[k];
      if (list.n === 0) continue;
      this.scratch[key] = 0;
      const state = { ...bc.s, topo: { ...bc.s.topo, [listKey]: list } };
      subs[k].compute({ ...bc, s: state, acc: this.scratch });
      const e = this.scratch[key];
      this.energy[k] = e;
      bc.acc[key] += e;
    }
  }

  /** Energy of each sub-style at the last force evaluation (indices 1..N in compute bond, angle, ...). */
  subEnergies(): Float64Array { return this.energy; }

  equilibrium(type: number): number {
    const a = type >= 0 && type < this.assign.length ? this.assign[type] : UNSET;
    if (a === NONE) {
      if (this.kind === 'bond') throw new StyleError('Invoked bond equil distance on bond style none');
      if (this.kind === 'angle') throw new StyleError('Invoked angle equil angle on angle style none');
      return Number.NaN;
    }
    if (a < 0) return Number.NaN;
    return this.build()[a].equilibrium(type);
  }

  private usedBy(k: number): boolean {
    for (let t = 1; t <= this.ntypes; t++) if (this.assign[t] === k) return true;
    return false;
  }

  private hasNone(): boolean {
    for (let t = 1; t <= this.ntypes; t++) if (this.assign[t] === NONE) return true;
    return false;
  }

  private make(name: string, ctx: StyleContext | null): Bonded {
    const f = this.registry[name];
    if (!f || name === 'hybrid') throw new StyleError(`${this.kind}_style hybrid sub-style '${name}' is not supported by the browser engine`);
    const st = f();
    st.settings([], ctx ?? NO_CONTEXT);
    return st;
  }

  /**
   * Builds one instance per sub-style holding the coefficients of its own types. Every type the
   * sub-style does not own gets a copy of its first coefficient line: the sub-style's init() checks
   * every type, and the copies are never read because compute() only sees the owned types.
   */
  private build(): Bonded[] {
    if (this.subs) return this.subs;
    if (this.assign.length !== this.ntypes + 1) this.allocate(this.ntypes);
    const c = this.ctx ?? NO_CONTEXT;
    const subs = this.subNames.map((n) => this.make(n, this.ctx));
    for (const st of subs) {
      st.allocate(this.ntypes);
      if (this.link) st.linkForceField?.(this.link);
    }
    const donor: (string[] | null)[] = new Array<string[] | null>(subs.length).fill(null);
    for (let t = 1; t <= this.ntypes; t++) {
      const a = this.assign[t];
      if (a < 0) continue;
      const w = this.words[t]!;
      subs[a].coeff([String(t), ...w], c);
      donor[a] ??= w;
    }
    for (let k = 0; k < subs.length; k++) {
      const w = donor[k];
      if (!w) continue;
      for (let t = 1; t <= this.ntypes; t++) {
        if (this.assign[t] !== k) subs[k].coeff([String(t), ...w], c);
      }
    }
    this.subs = subs;
    return subs;
  }

  /** Splits a topology list into one list per sub-style (types with no sub-style are dropped). */
  private bucket(list: TopoList): TopoList[] {
    const nsub = this.subNames.length;
    const width = list.width;
    const owner = (t: number): number => (t >= 0 && t < this.assign.length && this.assign[t] >= 0 ? this.assign[t] : -1);
    const counts = new Int32Array(nsub);
    for (let e = 0; e < list.n; e++) {
      const k = owner(list.type[e]);
      if (k >= 0) counts[k]++;
    }
    const out: TopoList[] = [];
    for (let k = 0; k < nsub; k++) {
      out.push({ n: counts[k], width, type: new Int32Array(counts[k]), atoms: new Int32Array(counts[k] * width) });
    }
    const fill = new Int32Array(nsub);
    for (let e = 0; e < list.n; e++) {
      const k = owner(list.type[e]);
      if (k < 0) continue;
      const j = fill[k]++;
      const dst = out[k];
      dst.type[j] = list.type[e];
      for (let c = 0; c < width; c++) dst.atoms[j * width + c] = list.atoms[e * width + c];
    }
    return out;
  }
}
