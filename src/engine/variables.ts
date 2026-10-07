import { StyleError } from './force/types';
import { evaluate, type FormulaEnv, type Mode, type Value } from './formula';
import { formatNumber } from './script';

/*
 * The variable store — docs.lammps.org/variable.html.
 *
 * "When a variable command is encountered in the input script and the
 * variable name has already been specified, the command is ignored ...
 * There are two exceptions to this rule. First, variables of style string,
 * getenv, internal, equal, vector, atom, and python ARE redefined each time
 * the command is encountered. ... Second, ... if a variable is iterated on to
 * the end of its list of strings via the next command, it is removed from the
 * list of active variables, and is thus available to be re-defined".
 * index: "Initially, the first string is assigned to the variable. Each time
 * a next command is used with the variable name, the next string is
 * assigned." loop: "the strings are the integers from 1 to N inclusive ...
 * can also be specified with two arguments N1 and N2"; "pad = all values
 * will be same length, e.g. 001, 002, ..., 100". world/universe/uloop with
 * one partition behave like index/loop on the single world. string:
 * "performs variable substitution even if the string parameter is quoted".
 * format: "vname fstr" formats an equal-style variable. "Vector-style variables
 * only can be initialized with a special syntax, instead of using a formula.
 * The syntax is a bracketed, comma-separated syntax like ... [1,3.5,7,10.2]".
 * Substitution text (measured with native LAMMPS): $x of an equal-style
 * variable prints %.15g; a vector prints "[a,b,c]" with each element in
 * shortest round-trip form. equal/vector/atom:
 * "a formula that will be evaluated afresh each time the variable is used".
 * "Variables are not deleted by the clear command with the exception of
 * atomfile-style variables."
 */

export type VarStyle =
  | 'index' | 'loop' | 'world' | 'universe' | 'uloop' | 'string' | 'getenv' | 'internal'
  | 'equal' | 'vector' | 'atom' | 'format' | 'file' | 'atomfile' | 'timer';

export interface Variable {
  style: VarStyle;
  /** index/loop/universe/...: the list; string/getenv/file: one value. */
  values: string[];
  which: number;
  /** equal/vector/atom formula; format: [vname, fmt]. */
  formula?: string;
  format?: [string, string];
  /** internal value. */
  num?: number;
  /** file/atomfile: remaining lines; atomfile: per-atom values by id. */
  lines?: string[];
  atomValues?: Map<number, number>;
  /** timer: wall-clock reference. */
  t0?: number;
  /** vector initialized from a [a,b,c] list. */
  vec?: Float64Array;
}

const REDEFINABLE = new Set<VarStyle>(['string', 'getenv', 'internal', 'equal', 'vector', 'atom']);

export class Variables {
  readonly vars = new Map<string, Variable>();
  /** Guards against self-referencing formulas. */
  private depth = 0;

  constructor(private readFile: (name: string) => string) {}

  has(name: string): boolean { return this.vars.has(name); }
  get(name: string): Variable | undefined { return this.vars.get(name); }

  /** variable name style args — returns false if ignored (already defined). */
  define(name: string, style: string, args: string[]): boolean {
    if (!/^[A-Za-z0-9_]+$/.test(name)) throw new StyleError(`variable name '${name}' must be alphanumeric or underscore`);
    if (style === 'delete') {
      if (args.length) throw new StyleError('variable delete takes no arguments');
      this.vars.delete(name);
      return true;
    }
    const existing = this.vars.get(name);
    if (existing && !REDEFINABLE.has(style as VarStyle)) return false;
    if (existing && existing.style !== style && REDEFINABLE.has(style as VarStyle) && !REDEFINABLE.has(existing.style)) {
      throw new StyleError(`cannot redefine variable ${name} as a different style`);
    }
    const one = () => {
      if (args.length !== 1) throw new StyleError(`variable ${style} takes exactly one argument (quote text that contains spaces)`);
      return args[0];
    };
    let v: Variable;
    switch (style) {
      case 'index': case 'world': case 'universe':
        if (args.length < 1) throw new StyleError(`variable ${style} needs at least one value`);
        v = { style, values: [...args], which: 0 };
        break;
      case 'loop': case 'uloop': {
        let pad = false;
        const a = [...args];
        if (a[a.length - 1] === 'pad') { pad = true; a.pop(); }
        const ints = a.map((w) => {
          if (!/^-?\d+$/.test(w)) throw new StyleError(`variable ${style}: '${w}' is not an integer`);
          return Number(w);
        });
        let n1: number, n2: number;
        if (ints.length === 1) { n1 = 1; n2 = ints[0]; }
        else if (ints.length === 2 && style === 'loop') { [n1, n2] = ints; }
        else throw new StyleError(`usage: variable name ${style} N [pad]${style === 'loop' ? ' or N1 N2 [pad]' : ''}`);
        if (n2 < 0 || n1 > n2) throw new StyleError(`variable ${style}: need N1 <= N2 and N2 >= 0`);
        const width = String(n2).length;
        const values: string[] = [];
        for (let k = n1; k <= n2; k++) values.push(pad ? String(k).padStart(width, '0') : String(k));
        if (values.length === 0) throw new StyleError(`variable ${style} has no values`);
        v = { style, values, which: 0 };
        break;
      }
      case 'string': v = { style, values: [one()], which: 0 }; break;
      // a browser has no process environment: every name is unset (empty string)
      case 'getenv': one(); v = { style, values: [''], which: 0 }; break;
      case 'internal': {
        const n = Number(one());
        if (!Number.isFinite(n)) throw new StyleError('variable internal needs a numeric value');
        v = { style, values: [], which: 0, num: n };
        break;
      }
      case 'equal': case 'vector': case 'atom': {
        const f = one();
        v = { style, values: [], which: 0, formula: f };
        const list = /^\s*\[(.*)\]\s*$/.exec(f);
        if (style === 'vector' && list) {
          const nums = list[1].split(',').map((w) => {
            const x = Number(w);
            if (w.trim() === '' || !Number.isFinite(x)) throw new StyleError(`vector initializer: '${w}' is not a number`);
            return x;
          });
          v.vec = Float64Array.from(nums);
        }
        break;
      }
      case 'format':
        if (args.length !== 2) throw new StyleError('usage: variable name format vname fstr');
        v = { style, values: [], which: 0, format: [args[0], args[1]] };
        break;
      case 'file': {
        const lines = this.fileLines(one());
        if (!lines.length) throw new StyleError(`file-style variable ${name}: file has no values`);
        v = { style, values: [lines.shift()!], which: 0, lines };
        break;
      }
      case 'atomfile': {
        const lines = this.fileLines(one());
        v = { style, values: [], which: 0, lines };
        this.nextAtomfile(v);
        break;
      }
      case 'timer': v = { style, values: [], which: 0, t0: performance.now() }; break;
      case 'python': throw new StyleError('variable python is not available in the browser engine (no Python interpreter)');
      default:
        throw new StyleError(`unknown variable style '${style}'`);
    }
    this.vars.set(name, v);
    return true;
  }

  /** Non-comment, non-blank lines of a file. */
  private fileLines(file: string): string[] {
    return this.readFile(file).split('\n').map((l) => l.replace(/#.*/, '').trim()).filter((l) => l.length > 0);
  }

  /** atomfile: "N" line then N lines "ID value". */
  private nextAtomfile(v: Variable): boolean {
    const lines = v.lines!;
    if (!lines.length) return false;
    const n = Number(lines.shift());
    if (!Number.isInteger(n) || n < 0) throw new StyleError('atomfile variable: expected a count line');
    const m = new Map<number, number>();
    for (let k = 0; k < n; k++) {
      const w = (lines.shift() ?? '').split(/\s+/);
      m.set(Number(w[0]), Number(w[1]));
    }
    v.atomValues = m;
    return true;
  }

  /**
   * next name1 name2 ... — docs.lammps.org/next.html. Returns true if any
   * variable was exhausted (then the next jump is skipped).
   */
  next(names: string[]): boolean {
    let exhausted = false;
    const vs = names.map((nm) => {
      const v = this.vars.get(nm);
      if (!v) throw new StyleError(`invalid variable '${nm}' in next command`);
      return [nm, v] as const;
    });
    const style = vs[0][1].style;
    for (const [nm, v] of vs) {
      if (v.style !== style && !(['index', 'loop'].includes(v.style) && ['index', 'loop'].includes(style))) {
        throw new StyleError('all variables in a next command must be the same style');
      }
      if (['index', 'loop', 'world', 'universe', 'uloop'].includes(v.style)) {
        v.which++;
        if (v.which >= v.values.length) { exhausted = true; this.vars.delete(nm); }
      } else if (v.style === 'file') {
        const line = v.lines!.shift();
        if (line === undefined) { exhausted = true; this.vars.delete(nm); } else v.values = [line];
      } else if (v.style === 'atomfile') {
        if (!this.nextAtomfile(v)) { exhausted = true; this.vars.delete(nm); }
      } else {
        throw new StyleError(`variable ${nm} (style ${v.style}) cannot be used with next`);
      }
    }
    return exhausted;
  }

  /** Text of a variable for $name / ${name} substitution. */
  text(name: string, env: FormulaEnv): string {
    const v = this.vars.get(name);
    if (!v) throw new StyleError(`substitution for illegal variable ${name}`);
    switch (v.style) {
      case 'equal': return formatNumber(this.scalar(name, env), '%.15g');
      case 'internal': return formatNumber(v.num!, '%.15g');
      case 'timer': return formatNumber((performance.now() - v.t0!) / 1000, '%.15g');
      case 'format': {
        const [vn, fmt] = v.format!;
        return formatNumber(this.scalar(vn, env), fmt);
      }
      case 'vector': {
        const a = this.evalVector(name, env);
        return `[${Array.from(a, (x) => String(x)).join(',')}]`;
      }
      case 'atom': throw new StyleError(`atom-style variable ${name} cannot be substituted with $ (it has one value per atom)`);
      case 'atomfile': throw new StyleError(`atomfile-style variable ${name} cannot be substituted with $`);
      default: return v.values[v.which];
    }
  }

  /** Numeric value of an equal-compatible variable. */
  scalar(name: string, env: FormulaEnv): number {
    const v = this.vars.get(name);
    if (!v) throw new StyleError(`variable ${name} is not defined`);
    switch (v.style) {
      case 'equal': {
        const r = this.guard(name, () => evaluate(v.formula!, env, 'equal'));
        if (typeof r !== 'number') throw new StyleError(`variable ${name} does not give a single value`);
        return r;
      }
      case 'internal': return v.num!;
      case 'timer': return (performance.now() - v.t0!) / 1000;
      case 'format': return this.scalar(v.format![0], env);
      case 'vector': case 'atom': case 'atomfile':
        throw new StyleError(`variable ${name} (style ${v.style}) does not give a single value`);
      default: {
        const s = v.values[v.which];
        const n = Number(s);
        if (s.trim() === '' || !Number.isFinite(n)) throw new StyleError(`variable ${name} = '${s}' is not a number`);
        return n;
      }
    }
  }

  /** A vector-style variable's values. */
  evalVector(name: string, env: FormulaEnv): Float64Array {
    const v = this.vars.get(name);
    if (!v || v.style !== 'vector') throw new StyleError(`variable ${name} is not a vector-style variable`);
    if (v.vec) return v.vec;
    const r = this.guard(name, () => evaluate(v.formula!, env, 'vector'));
    return typeof r === 'number' ? Float64Array.of(r) : r;
  }

  /** Per-atom values of an atom-style (or atomfile-style) variable, given the atom IDs. */
  evalAtom(name: string, env: FormulaEnv, ids: Int32Array, n: number): Float64Array {
    const v = this.vars.get(name);
    if (!v) throw new StyleError(`variable ${name} is not defined`);
    if (v.style === 'atomfile') {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[i] = v.atomValues?.get(ids[i]) ?? 0;
      return out;
    }
    if (v.style !== 'atom') {
      const x = this.scalar(name, env);
      return new Float64Array(n).fill(x);
    }
    const r = this.guard(name, () => evaluate(v.formula!, env, 'atom'));
    return typeof r === 'number' ? new Float64Array(n).fill(r) : r;
  }

  /** v_name in a formula in the given mode. */
  value(name: string, env: FormulaEnv, mode: Mode, ids: Int32Array, n: number): Value {
    const v = this.vars.get(name);
    if (!v) throw new StyleError(`variable ${name} in a formula is not defined`);
    if (v.style === 'atom' || v.style === 'atomfile') {
      if (mode !== 'atom') throw new StyleError(`atom-style variable ${name} can only be used in an atom-style variable`);
      return this.evalAtom(name, env, ids, n);
    }
    if (v.style === 'vector') {
      if (mode === 'equal') throw new StyleError(`vector-style variable ${name} needs an index v_${name}[i] in an equal-style formula`);
      return this.evalVector(name, env);
    }
    return this.scalar(name, env);
  }

  private guard<T>(name: string, f: () => T): T {
    if (this.depth > 64) throw new StyleError(`variable ${name} refers to itself (directly or through other variables)`);
    this.depth++;
    try { return f(); } finally { this.depth--; }
  }

  /** clear: "Variables are not deleted ... with the exception of atomfile-style variables." */
  clear(): void {
    for (const [k, v] of this.vars) if (v.style === 'atomfile') this.vars.delete(k);
  }

  /** Sets an internal-style variable (creating it if needed). */
  setInternal(name: string, value: number): void {
    const v = this.vars.get(name);
    if (v && v.style !== 'internal') throw new StyleError(`variable ${name} is not internal-style`);
    this.vars.set(name, { style: 'internal', values: [], which: 0, num: value });
  }
}
