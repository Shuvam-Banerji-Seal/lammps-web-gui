import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { expandWildcards, parseRef, peratomValues, type Ref } from '../refs';
import { CUSTOM_ATTR, hasCharge, hasMolecule, massOf } from '../atoms';
import { fillPeratomAttr, peratomAttrProblem } from '../compute/peratom';

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
 * zero tests). "history" is refused by name (a StyleError) since it needs the per-step history store.
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

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `fix ${id} (store/state)`;
    if (args.length < 2) throw new StyleError(`usage: fix ${id} group-ID store/state N input1 input2 ... [keyword value ...]`);
    this.nStore = Number(args[0]);
    if (!Number.isInteger(this.nStore) || this.nStore < 0) throw new StyleError(`${what}: N must be an integer >= 0`);
    let k = 1;
    const words: string[] = [];
    while (k < args.length && !KEYWORDS.has(args[k])) words.push(args[k++]);
    if (!words.length) throw new StyleError(`${what}: no input attributes given`);
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
        throw new StyleError(`${what}: keyword 'history' is not supported by the browser engine`);
      } else if (kw === 'thresh') {
        if (k + 4 > args.length) throw new StyleError(`${what}: keyword thresh needs three values`);
        if (this.thresh) throw new StyleError(`${what}: the thresh keyword may be used only once`);
        const name = args[k + 1];
        if (!name.startsWith('v_')) throw new StyleError(`${what}: no variable reference as first argument to thresh keyword`);
        const op = args[k + 2];
        if (!OPS.has(op)) throw new StyleError(`${what}: unsupported thresh comparison operator ${op}`);
        const value = Number(args[k + 3]);
        if (args[k + 3].trim() === '' || !Number.isFinite(value)) throw new StyleError(`${what}: invalid thresh value '${args[k + 3]}'`);
        this.thresh = { name: name.slice(2), op, value };
        this.checkThreshVariable();
        k += 4;
      } else {
        throw new StyleError(`Unknown fix store/state keyword: ${kw}`);
      }
    }
    this.comflag = comflag;
    this.nevery = 1;
    this.alloc();
    // native constructor: end_of_step with kflag = 1 and cfv_flag = 0 (keyword attributes only)
    this.stepStore(true, false);
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
    const r = parseRef(w, false);
    if (r.kind === 'v') {
      const v = this.sys.vars.get(r.id);
      if (!v) throw new StyleError(`fix store/state: variable ${r.id} does not exist`);
      if (v.style !== 'atom') throw new StyleError(`fix store/state: variable ${r.id} must be atom-style`);
    }
    return { kind: 'ref', text: w, ref: r };
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

  /** Stores the keyword inputs and/or the compute, fix, variable and custom inputs, when the threshold is met. */
  private stepStore(keywords: boolean, cfv: boolean): void {
    if (!this.threshMet()) return;
    this.alloc();
    const s = this.sys.state;
    const n = s.n;
    const dest = this.nvals === 1 ? this.vectorAtom : this.arrayAtom;
    const stride = this.nvals === 1 ? 1 : this.nvals;
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
  }

  /** The first run stores the inputs that could not be read at the definition (native setup with cfv_flag). */
  setup(): void {
    if (!this.firstSetup) return;
    this.firstSetup = false;
    this.stepStore(false, true);
  }

  init(): void {
    this.alloc();
  }

  endOfStep(): void {
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
