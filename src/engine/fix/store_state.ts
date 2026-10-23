import { Fix } from './fix';
import { errorUrl } from '../formula';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { expandWildcards, parseRef, peratomValues, type Ref } from '../refs';
import { CUSTOM_ATTR, hasCharge, hasMolecule, massOf } from '../atoms';
import { fillPeratomAttr, peratomAttrProblem } from '../compute/peratom';
import { inumeric, missingArgs, numeric } from '../force/util';

/*
 * fix ID group-ID store/state N input1 input2 ... [keyword value ...] — docs.lammps.org/fix_store_state.html:
 * "N = store atom attributes every N steps, N = 0 for initial store only"; "If N > 0, then the stored attributes
 * will be updated once every N steps. If N = 0, then the attributes are stored when the fix is defined and never
 * changed"; "The requested values are stored in a per-atom vector or array as discussed below. Zeroes are stored for
 * atoms not in the specified group."; "Only atom attributes specified by keywords like xu or vy or radius are
 * initially stored immediately at the point in your input script when the fix is defined. Attributes specified by a
 * compute, fix, or variable are not initially stored until the first run following the fix definition begins."
 * "The list of possible attributes is the same as that used by the dump custom command, which describes their meaning."
 * Measured with native LAMMPS (black box, lj fcc, fix store/state 5 x y z, dump of f_s[1..3] each step): the values
 * hold the definition-time positions on steps 0 to 4, and change to the positions of steps 5 and 10 at those steps.
 *
 * Keywords (native src/fix_store_state.cpp at b332f28, the constructor's optional-argument loop):
 * "com" (docs: "If the com keyword is set to yes then the xu, yu, and zu inputs store the position of each atom
 * relative to the center-of-mass of the group of atoms"; native subtracts the group's mass-weighted centre of the
 * unwrapped positions from those three attributes only, not from the scaled ones); "thresh" (docs: "If the thresh
 * keyword is used, data is only stored on steps where also the threshold condition following the keyword is met.
 * The first argument must be a variable reference to an equal-style or compatible variable"; "The thresh keyword may
 * be used exactly once"; operators "<", "<=", ">", ">=", "==", "!=" and "|^", the last meaning exclusive or of the
 * zero tests); "history" (docs: "history values = Nevery Nrepeat Nfreq"; "If the optional keyword history is used,
 * then the N keyword must be specified as 0, and attributes are stored for multiple timesteps"; "In both cases,
 * attributes are stored Nrepeat times, once every Nevery steps"; "If Nfreq = 0, then the Nrepeat attributes are updated
 * continuously and the attributes are accessible on any timestep which is a multiple of Nevery"; "If Nfreq is
 * non-zero, it must be a multiple of Nevery, and the attributes are accessible only on timesteps which are a multiple
 * of Nfreq"; "If the history keyword is used, no per-atom vector or array is produced. Instead, the fix provides access
 * to its multiple timesteps of stored per-atom history", read by compute property/atom history[I][J]).
 *
 * History store, as native LAMMPS does (src/fix_store_state.cpp at b332f28): the argument checks are N = 0, Nevery > 0,
 * Nrepeat > 0, Nfreq >= 0 and a multiple of Nevery; nothing is stored at the definition; a run setup on a multiple of
 * Nevery and every later multiple of Nevery store all inputs as one frame, in a ring of Nrepeat frames, except that
 * with Nfreq > Nevery Nrepeat a step whose next multiple of Nfreq is Nrepeat or more Nevery-intervals away is skipped;
 * a frame more than Nevery steps after the previous one restarts the count of available frames. The frames are per
 * atom in the engine's atom order; a change of the atom count clears them.
 *
 * Native timing (src/fix_store_state.cpp end_of_step, setup): the constructor stores the keyword attributes only; the
 * first run's setup stores the compute, fix, variable and custom-property inputs only; each step that is a multiple
 * of N stores everything. The threshold is tested first, and a failing step stores nothing.
 */

/** Attribute words of the docs list that the engine stores (the dump custom attributes that have per-atom data). */
const ATTRS = new Set([
  'id', 'mol', 'type', 'mass', 'x', 'y', 'z', 'xs', 'ys', 'zs', 'xu', 'yu', 'zu', 'xsu', 'ysu', 'zsu', 'ix', 'iy', 'iz',
  'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'q', 'mux', 'muy', 'muz', 'mu', 'radius', 'diameter',
  'omegax', 'omegay', 'omegaz', 'angmomx', 'angmomy', 'angmomz', 'tqx', 'tqy', 'tqz',
]);
const COM_ATTRS = ['xu', 'yu', 'zu'];
const KEYWORDS = new Set(['com', 'history', 'thresh']);
/** A word that can be an input: an attribute, or a compute, fix, variable or custom reference (prefix c_, f_, v_, d_, i_). */
const isInputWord = (w: string): boolean => ATTRS.has(w) || /^[cfvid]2?_/.test(w);
const OPS = new Set(['<', '<=', '>', '>=', '==', '!=', '|^']);
/** utils::logical (src/utils.cpp at b332f28): yes, on, true, 1 and no, off, false, 0. */
const TRUE_WORDS = new Set(['yes', 'on', 'true', '1']);
const FALSE_WORDS = new Set(['no', 'off', 'false', '0']);

/**
 * A stored input: 'keyword' attributes are packed at the constructor and on each stored step; 'custom' (i_name,
 * d2_name[I], ...) and 'ref' (c_, f_, v_) inputs wait for the first run, as the native cfv branch does.
 */
interface Input {
  kind: 'keyword' | 'custom' | 'ref';
  text: string;
  ref?: Ref;
}

const LT = '<', LE = '<=', GT = '>', GE = '>=', EQ = '==', NEQ = '!=', XOR = '|^';

export class FixStoreState extends Fix {
  readonly style = 'store/state';
  peratomFlag = true;
  private readonly nStore: number;
  private readonly inputs: Input[];
  private readonly nvals: number;
  private readonly comflag: boolean;
  private thresh: { name: string; op: string; value: number } | null = null;
  private firstSetup = true;
  /** The history keyword: Nevery, Nrepeat, Nfreq; null without it. */
  history: { nevery: number; nrepeat: number; nfreq: number } | null = null;
  /** Ring of stored frames (nvals values per atom each), the newest at mostRecentIndex; countHistory are valid. */
  private frames: (Float64Array | null)[] = [];
  countHistory = 0;
  mostRecentIndex = -1;
  private mostRecentStep = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `fix ${id} (store/state)`;
    if (args.length < 2) throw new StyleError(`usage: fix ${id} group-ID store/state N input1 input2 ... [keyword value ...]`);
    // As native LAMMPS does (src/fix_store_state.cpp FixStoreState::FixStoreState at b332f28): N is read with utils::inumeric
    this.nStore = inumeric(args[0]);
    if (this.nStore < 0) throw new StyleError(`Invalid fix store/state N value ${this.nStore}`);
    let k = 1;
    const words: string[] = [];
    // As native LAMMPS does (src/fix_store_state.cpp at b332f28): the input loop ends at the first word that is neither an
    // attribute nor a c_, f_, v_, d_ or i_ reference, and that word is then read as a keyword
    while (k < args.length && !KEYWORDS.has(args[k]) && isInputWord(args[k])) words.push(args[k++]);
    if (!words.length) {
      if (k < args.length && !KEYWORDS.has(args[k])) throw new StyleError(`Unknown fix store/state keyword: ${args[k]}`);
      throw new StyleError(`${what}: no input attributes given`);
    }
    this.inputs = expandInputs(sys, words).map((w) => this.parseInput(w));
    this.nvals = this.inputs.length;
    this.sizePeratomCols = this.nvals === 1 ? 0 : this.nvals;
    let comflag = false;
    while (k < args.length) {
      const kw = args[k];
      if (kw === 'com') {
        if (k + 2 > args.length) throw new StyleError(`${what}: keyword com needs a value`);
        const v = args[k + 1];
        if (!TRUE_WORDS.has(v) && !FALSE_WORDS.has(v)) throw new StyleError(`${what}: com must be yes or no, not '${v}'`);
        comflag = TRUE_WORDS.has(v);
        k += 2;
      } else if (kw === 'history') {
        if (k + 4 > args.length) throw new StyleError('Illegal fix store/state history command: missing argument(s)');
        const v = args.slice(k + 1, k + 4).map((w) => inumeric(w));
        this.history = { nevery: v[0], nrepeat: v[1], nfreq: v[2] };
        k += 4;
      } else if (kw === 'thresh') {
        // As native LAMMPS does (src/fix_store_state.cpp FixStoreState::FixStoreState at b332f28): the word count, one
        // use only, the v_ reference and its variable, the operator, then the value (utils::numeric)
        if (k + 4 > args.length) throw missingArgs('fix store/state thresh');
        if (this.thresh) throw new StyleError('The fix store/state thresh keyword may be used only once');
        const name = args[k + 1];
        if (!name.startsWith('v_')) throw new StyleError('No variable reference as first argument to thresh keyword');
        this.thresh = { name: name.slice(2), op: '', value: 0 };
        this.checkThreshVariable();
        const op = args[k + 2];
        if (!OPS.has(op)) throw new StyleError(`Unsupported fix store/state thresh comparison operator ${op}`);
        this.thresh = { name: name.slice(2), op, value: numeric(args[k + 3]) };
        k += 4;
      } else {
        throw new StyleError(`Unknown fix store/state keyword: ${kw}`);
      }
    }
    this.comflag = comflag;
    this.nevery = 1;
    const h = this.history;
    if (h) {
      if (this.nStore) throw new StyleError('Fix store/state N = 0 required with history keyword');
      if (h.nevery <= 0) throw new StyleError(`Invalid fix store/state history Nevery ${h.nevery}`);
      if (h.nrepeat <= 0) throw new StyleError(`Invalid fix store/state history Nrepeat ${h.nrepeat}`);
      if (h.nfreq < 0 || h.nfreq % h.nevery) throw new StyleError(`Invalid fix store/state history Nfreq ${h.nfreq}`);
      this.nevery = h.nevery;
      // no per-atom vector or array: the frames are read through compute property/atom history[I][J]
      this.peratomFlag = false;
      this.sizePeratomCols = 0;
      this.frames = new Array(h.nrepeat).fill(null);
    }
    this.checkRefs();
    this.alloc();
    // native constructor: end_of_step with kflag = 1 and cfv_flag = 0 (keyword attributes only), not with history
    if (!h) this.stepStore(true, false);
  }

  /** Number of stored attributes per atom (the J range of compute property/atom history[I][J]). */
  get nattribute(): number {
    return this.nvals;
  }

  /**
   * Frame I (1 = most recent) of attribute J (1-based) for each atom: the stored value for the atoms of groupBit, zero
   * for the others, and zero for all when fewer than I frames are stored (native pack_history).
   */
  historyColumn(frame: number, col: number, groupBit: number, out: Float64Array, stride: number, k: number): void {
    const h = this.history!;
    const s = this.sys.state;
    let idx = this.mostRecentIndex - (frame - 1);
    if (idx < 0) idx += h.nrepeat;
    const data = this.frames[idx];
    const avail = frame <= this.countHistory && data && data.length === this.nvals * s.n;
    for (let i = 0; i < s.n; i++) {
      out[stride * i + k] = avail && (s.mask[i] & groupBit) ? data![this.nvals * i + col - 1] : 0;
    }
  }

  private parseInput(w: string): Input {
    if (ATTRS.has(w)) {
      const s = this.sys.state;
      const flagMissing = (w === 'mol' && !hasMolecule(s)) || (w === 'q' && !hasCharge(s));
      if (flagMissing || peratomAttrProblem(this.sys, w)) throw new StyleError(`Cannot use fix store/state ${w} for atom style ${this.sys.atomStyle}`);
      return { kind: 'keyword', text: w };
    }
    if (CUSTOM_ATTR.test(w)) {
      const name = /^[id]2?_([A-Za-z0-9_]+)/.exec(w)![1];
      if (!this.sys.state.custom.has(name)) throw new StyleError(`Custom vector/array ${name} for fix store/state does not exist`);
      return { kind: 'custom', text: w };
    }
    return { kind: 'ref', text: w, ref: parseRef(w, false) };
  }

  /**
   * The c_ID, f_ID and v_name inputs, checked after the keywords as native LAMMPS does (src/fix_store_state.cpp
   * FixStoreState::FixStoreState "error check" at b332f28): each exists and has per-atom data of the shape the word asks
   * for; a variable is atom-style.
   */
  private checkRefs(): void {
    for (const inp of this.inputs) {
      if (inp.kind !== 'ref' || !inp.ref) continue;
      const r = inp.ref;
      if (r.kind === 'v') {
        const v = this.sys.vars.get(r.id);
        if (!v) throw new StyleError(`Variable name ${r.id} for fix store/state does not exist`);
        if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`Fix store/state variable ${r.id} is not atom-style variable`);
        continue;
      }
      if (r.kind !== 'c' && r.kind !== 'f') continue;
      const what = r.kind === 'c' ? 'compute' : 'fix';
      const src = r.kind === 'c' ? this.sys.compute(r.id, `Compute ID ${r.id} for fix store/state does not exist`) : this.sys.fix(r.id, `Fix ID ${r.id} for fix store/state does not exist`);
      const k = r.index ?? 0;
      if (!src.peratomFlag) throw new StyleError(`Fix store/state ${what} ${r.id} does not calculate per-atom values`);
      if (k === 0 && src.sizePeratomCols !== 0) throw new StyleError(`Fix store/state ${what} ${r.id} does not calculate per-atom vector`);
      if (k > 0 && src.sizePeratomCols === 0) throw new StyleError(`Fix store/state ${what} ${r.id} does not calculate per-atom array`);
      if (k > 0 && k > src.sizePeratomCols) {
        throw new StyleError(r.kind === 'c' ? `Fix store/state compute array ${r.id} is accessed out-of-range${errorUrl(20)}` : `Fix store/state fix ${r.id} array is accessed out-of-range${errorUrl(20)}`);
      }
    }
  }

  private checkThreshVariable(): void {
    const t = this.thresh!;
    const v = this.sys.vars.get(t.name);
    if (!v) throw new StyleError(`Variable name ${t.name} for fix store/state thresh does not exist`);
    if (v.style !== 'equal') throw new StyleError(`Fix store/state thresh variable ${t.name} has incompatible style`);
  }

  /** The per-atom storage follows the atom count (zeros for the atoms not yet stored). */
  private alloc(): void {
    const need = this.nvals * this.sys.state.n;
    if (this.history) {
      if (this.arrayAtom.length !== need) {
        this.arrayAtom = new Float64Array(need);
        this.frames.fill(null);
        this.countHistory = 0;
      }
      return;
    }
    if (this.nvals === 1) {
      if (this.vectorAtom.length !== this.sys.state.n) this.vectorAtom = new Float64Array(this.sys.state.n);
    } else if (this.arrayAtom.length !== need) this.arrayAtom = new Float64Array(need);
  }

  /** The thresh test of the native end_of_step; true when there is no thresh keyword. */
  private threshMet(): boolean {
    if (!this.thresh) return true;
    const val = this.sys.equalVariable(this.thresh.name);
    const t = this.thresh.value;
    switch (this.thresh.op) {
      case LT: return val < t;
      case LE: return val <= t;
      case GT: return val > t;
      case GE: return val >= t;
      case EQ: return val === t;
      case NEQ: return val !== t;
      case XOR: return (val === 0) !== (t === 0);
      default: return true;
    }
  }

  /** Mass-weighted centre of the group's unwrapped positions (native group->xcm), for the com keyword. */
  private groupCentre(): number[] {
    const s = this.sys.state;
    const u = [0, 0, 0];
    const c = [0, 0, 0];
    let mt = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      const m = massOf(s, i);
      mt += m;
      for (let d = 0; d < 3; d++) c[d] += m * u[d];
    }
    return mt > 0 ? c.map((x) => x / mt) : c;
  }

  /**
   * Stores the keyword inputs and/or the compute, fix, variable and custom inputs, when the threshold is met; false
   * when it is not.
   */
  private stepStore(keywords: boolean, cfv: boolean): boolean {
    if (!this.threshMet()) return false;
    this.alloc();
    const s = this.sys.state;
    const n = s.n;
    const single = this.nvals === 1 && !this.history;
    const dest = single ? this.vectorAtom : this.arrayAtom;
    const stride = single ? 1 : this.nvals;
    const cm = this.comflag ? this.groupCentre() : null;
    for (let k = 0; k < this.nvals; k++) {
      const r = this.inputs[k];
      if (r.kind === 'keyword' ? !keywords : !cfv) continue;
      // atoms outside the group keep the zero that is stored for them
      for (let i = 0; i < n; i++) dest[stride * i + k] = 0;
      if (r.kind !== 'ref') {
        fillPeratomAttr(this.sys, r.text, this.groupBit, dest, stride, k);
        if (cm && COM_ATTRS.includes(r.text)) {
          const d = 'xyz'.indexOf(r.text[0]);
          for (let i = 0; i < n; i++) if (s.mask[i] & this.groupBit) dest[stride * i + k] -= cm[d];
        }
      } else {
        const vals = peratomValues(this.sys, r.ref!);
        for (let i = 0; i < n; i++) if (s.mask[i] & this.groupBit) dest[stride * i + k] = vals[i];
      }
    }
    return true;
  }

  /** One history frame (native end_of_step with historyflag): the Nfreq window test, then the store and the ring. */
  private historyStep(): void {
    const h = this.history!;
    const step = this.sys.state.step;
    if (h.nfreq > h.nevery * h.nrepeat && step % h.nfreq) {
      const nfreqNext = Math.floor(step / h.nfreq) * h.nfreq + h.nfreq;
      const nprevious = Math.floor((nfreqNext - step) / h.nevery);
      if (nprevious >= h.nrepeat) return;
    }
    if (!this.stepStore(true, true)) return;
    if (step - this.mostRecentStep > h.nevery) this.countHistory = 0;
    this.mostRecentStep = step;
    if (this.countHistory < h.nrepeat) this.countHistory++;
    this.mostRecentIndex++;
    if (this.mostRecentIndex === h.nrepeat) this.mostRecentIndex = 0;
    this.frames[this.mostRecentIndex] = Float64Array.from(this.arrayAtom);
  }

  /** The first run stores the inputs that could not be read at the definition (native setup with cfv_flag). */
  setup(): void {
    if (!this.firstSetup) return;
    this.firstSetup = false;
    if (!this.history) this.stepStore(false, true);
    else if (this.sys.state.step % this.history.nevery === 0) this.historyStep();
  }

  init(): void {
    this.alloc();
  }

  endOfStep(): void {
    if (this.history) {
      if (this.sys.state.step % this.history.nevery === 0) this.historyStep();
      return;
    }
    if (this.nStore > 0 && this.sys.state.step % this.nStore === 0) this.stepStore(true, true);
  }
}

/**
 * Expands the input words: custom d2_name[*], d2_name[m*n] (native expand_args on custom arrays) and compute, fix
 * and variable wildcards (refs.ts expandWildcards).
 */
const expandInputs = (sys: System, words: string[]): string[] => {
  const out: string[] = [];
  for (const w of words) {
    const m = /^([id]2_[A-Za-z0-9_]+)\[(\d*)\*(\d*)\]$/.exec(w);
    if (!m) { out.push(...expandWildcards(sys, [w], 'peratom')); continue; }
    const name = m[1].slice(3);
    const cp = sys.state.custom.get(name);
    if (!cp) throw new StyleError(`Custom vector/array ${name} for fix store/state does not exist`);
    const lo = m[2] === '' ? 1 : Number(m[2]);
    const hi = m[3] === '' ? cp.cols : Number(m[3]);
    for (let c = lo; c <= Math.min(hi, cp.cols); c++) out.push(`${m[1]}[${c}]`);
  }
  return out;
};
