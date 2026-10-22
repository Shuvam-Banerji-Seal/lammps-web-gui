import { StyleError } from './force/types';
import { inumeric, numeric } from './force/util';
import { checkCFormat } from './cformat';
import { evaluate, FormulaError, roundUnderflow, type FormulaEnv, type Mode, type Value } from './formula';
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
 * will be same length, e.g. 001, 002, ..., 100". world/universe/uloop take the value of the partition (the
 * first one on a single partition). As native LAMMPS does (src/variable.cpp at b332f28): a world variable has one
 * value per partition ("World variable count {} doesn't match # of partitions {}"), and a universe or uloop
 * variable at least as many ("{} style variable count {} < # of partitions {}"). string:
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

/** A number as native utils::is_double reads one: sign, digits with one period, optional exponent. */
const IS_DOUBLE = /^[+-]?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?$/;

/** The variable styles (native's varstyles). */
const ALL_STYLES = new Set<string>(['index', 'loop', 'world', 'universe', 'uloop', 'string', 'getenv', 'file', 'atomfile', 'format', 'equal', 'atom', 'vector', 'python', 'timer', 'internal']);

export class Variables {
  readonly vars = new Map<string, Variable>();
  /** The variables being evaluated now (native eval_in_progress): a formula reaching one again is circular. */
  private inProgress = new Set<string>();

  /**
   * The partition (replica) this table belongs to. A world-style variable gives each partition one of its N
   * values (variable.rst); uloop and universe values are handed out per partition as well. A single-partition
   * run uses partition 0, which is the first value (as before).
   */
  partition = 0;
  /** Partitions of the session (lammps -partition N); the world-style count check reads it. */
  nworlds = 1;

  /** warn: prints a WARNING line (the session's log). */
  /** getenv: the page's environment (System.env, set by shell putenv); undefined when the name is not set. */
  constructor(private readFile: (name: string) => string, private warn: (text: string) => void = () => undefined,
    private getenv: (name: string) => string | undefined = () => undefined) {}

  has(name: string): boolean { return this.vars.has(name); }
  get(name: string): Variable | undefined { return this.vars.get(name); }

  /**
   * variable name style args — returns false if ignored (already defined). As native LAMMPS does (src/variable.cpp
   * Variable::set at b332f28): index, loop, world, universe, uloop, file and atomfile keep an existing variable (of
   * any style); the other styles replace one of the same style and refuse one of another style ("Cannot redefine ...").
   */
  define(name: string, style: string, args: string[]): boolean {
    if (!/^[A-Za-z0-9_]+$/.test(name)) throw new StyleError(`Variable name '${name}' must have only letters, numbers, or underscores`);
    const narg = 2 + args.length;
    const url3 = '\nFor more information see https://docs.lammps.org/err0003';
    const expect = (n: number) => {
      if (narg !== n) throw new StyleError(`Illegal variable command: expected ${n} arguments but found ${narg}${url3}`);
    };
    if (style === 'delete') {
      if (narg > 2) throw new StyleError(`Illegal variable delete command: expected 2 arguments but found ${narg}${url3}`);
      this.vars.delete(name);
      return true;
    }
    const existing = this.vars.get(name);
    const KEEP = ['index', 'loop', 'world', 'universe', 'uloop', 'file', 'atomfile'];
    if (!KEEP.includes(style) && existing && existing.style !== style && ALL_STYLES.has(style)) {
      throw new StyleError(`Cannot redefine ${existing.style} style variable ${name} as ${style} style variable`);
    }
    const one = () => args[0];
    let v: Variable;
    switch (style) {
      case 'index': case 'world': case 'universe':
        if (args.length < 1) throw new StyleError(`Illegal variable ${style} command: missing argument(s)`);
        if (existing) return false;
        if (style === 'world' && args.length !== this.nworlds) {
          throw new StyleError(`World variable count ${args.length} doesn't match # of partitions ${this.nworlds}`);
        }
        if (style === 'universe' && args.length < this.nworlds) {
          throw new StyleError(`universe style variable count ${args.length} < # of partitions ${this.nworlds}`);
        }
        // world/universe: partition k takes value k (the first partition takes the first value)
        v = { style, values: [...args], which: style === 'index' ? 0 : Math.min(this.partition, args.length - 1) };
        break;
      case 'loop': case 'uloop': {
        let pad = false;
        let n1: number, n2: number;
        if (style === 'loop') {
          if (args.length < 1) throw new StyleError('Illegal variable loop command: missing argument(s)');
          if (existing) return false;
          if (narg === 3 || (narg === 4 && args[1] === 'pad')) {
            n1 = 1;
            n2 = inumeric(args[0]);
            if (n2 <= 0) throw new StyleError(`Invalid variable loop argument: ${n2}`);
            pad = narg === 4;
          } else if (narg === 4 || (narg === 5 && args[2] === 'pad')) {
            n1 = inumeric(args[0]);
            n2 = inumeric(args[1]);
            if (n1 > n2 || n2 < 0) throw new StyleError(`Illegal variable loop command: ${n1} > ${n2}`);
            pad = narg === 5;
          } else throw new StyleError('Illegal variable loop command: too many arguments');
        } else {
          if (narg < 3 || narg > 4) throw new StyleError(`Illegal variable uloop command: expected 3 or 4 arguments but found ${narg}${url3}`);
          if (narg === 4 && args[1] !== 'pad') throw new StyleError(`Invalid variable uloop argument: ${args[1]}`);
          if (existing) return false;
          n1 = 1;
          n2 = inumeric(args[0]);
          pad = narg === 4;
        }
        const width = String(n2).length;
        const values: string[] = [];
        for (let k = n1; k <= n2; k++) values.push(pad ? String(k).padStart(width, '0') : String(k));
        if (values.length === 0) throw new StyleError(`variable ${style} has no values`);
        if (style === 'uloop' && values.length < this.nworlds) {
          throw new StyleError(`uloop style variable count ${values.length} < # of partitions ${this.nworlds}`);
        }
        v = { style, values, which: style === 'uloop' ? Math.min(this.partition, values.length - 1) : 0 };
        break;
      }
      case 'string': expect(3); v = { style, values: [one()], which: 0 }; break;
      // variable.html: "it returns the value of the environment variable, or an empty string if it not defined". The
      // browser engine runs in a page, which has no process environment: every name reads as not defined (empty)
      case 'getenv': expect(3); v = { style, values: [one()], which: 0 }; break;
      case 'internal': {
        expect(3);
        v = { style, values: [], which: 0, num: numeric(one()) };
        break;
      }
      case 'equal': case 'vector': case 'atom': {
        // variable.html: "If you want spaces in the string, enclose it in double quotes so the parser will treat
        // it as a single argument." Measured with native LAMMPS (black box): unquoted words after an equal,
        // vector or atom style are joined with single spaces (variable a equal 2 +    3 gives 5; atom x + 1 and
        // vector [1,2] * 3 are joined the same way), while string, format, getenv and internal refuse extra words.
        // native names the atom style for a vector-style variable too
        if (args.length < 1) throw new StyleError(`Illegal variable ${style === 'vector' ? 'atom' : style} command: missing argument(s)`);
        const f = args.join(' ');
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
      case 'format': {
        expect(4);
        const src = this.vars.get(args[0]);
        if (!src) throw new StyleError(`Variable ${name}: format variable ${args[0]} does not exist`);
        if (src.style !== 'equal' && src.style !== 'timer' && src.style !== 'internal') {
          throw new StyleError(`Variable ${name}: format variable ${args[0]} has incompatible style`);
        }
        checkCFormat(args[1], 'float', 'Invalid format string for format style variable');
        v = { style, values: [], which: 0, format: [args[0], args[1]] };
        break;
      }
      case 'file': {
        expect(3);
        if (existing) return false;
        const lines = this.fileLines(one(), 'file', name);
        if (!lines.length) throw new StyleError(`file-style variable ${name}: file has no values`);
        v = { style, values: [lines.shift()!], which: 0, lines };
        break;
      }
      case 'atomfile': {
        expect(3);
        if (existing) return false;
        const lines = this.fileLines(one(), 'atomfile', name);
        v = { style, values: [], which: 0, lines };
        this.nextAtomfile(v);
        break;
      }
      case 'timer': expect(2); v = { style, values: [], which: 0, t0: performance.now() }; break;
      // the browser engine has no Python interpreter, as a native binary built without the PYTHON package
      case 'python': expect(3); throw new StyleError('LAMMPS is not built with Python embedded');
      default:
        throw new StyleError(`Unknown variable style: ${style}`);
    }
    this.vars.set(name, v);
    return true;
  }

  /** Non-comment, non-blank lines of a file. */
  private fileLines(file: string, style = 'file', name = ''): string[] {
    let text: string;
    try {
      text = this.readFile(file);
    } catch (e) {
      // native's message (VarReader), then the engine's hint
      if (e instanceof StyleError) throw new StyleError(`Cannot open ${style} variable ${name} file ${file}: No such file or directory (${e.message})`);
      throw e;
    }
    return text.split('\n').map((l) => l.replace(/#.*/, '').trim()).filter((l) => l.length > 0);
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
    // As native LAMMPS does (src/variable.cpp Variable::next at b332f28): every variable exists and has the first one's
    // style (universe and uloop may mix), no name repeats, and the style is one next advances
    const vs: (readonly [string, Variable])[] = [];
    for (const nm of names) {
      const v = this.vars.get(nm);
      if (!v) throw new StyleError(`Invalid variable '${nm}' in next command`);
      if (vs.length) {
        const z = vs[0][1].style;
        const mix = (v.style === 'uloop' && z === 'universe') || (v.style === 'universe' && z === 'uloop');
        if (!mix && v.style !== z) throw new StyleError('All variables in next command must have same style');
      }
      vs.push([nm, v] as const);
    }
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) if (names[i] === names[j]) throw new StyleError(`Duplicate variable '${names[j]}' in next command`);
    }
    const style = vs[0][1].style;
    if (['string', 'equal', 'world', 'getenv', 'atom', 'vector', 'format', 'python', 'timer', 'internal'].includes(style)) {
      throw new StyleError(`Variable ${names[0]} has invalid style ${style} for next command`);
    }
    for (const [nm, v] of vs) {
      if (['index', 'loop', 'world', 'universe', 'uloop'].includes(v.style)) {
        v.which++;
        if (v.which >= v.values.length) { exhausted = true; this.vars.delete(nm); }
      } else if (v.style === 'file') {
        const line = v.lines!.shift();
        if (line === undefined) { exhausted = true; this.autoDelete(nm); } else v.values = [line];
      } else if (v.style === 'atomfile') {
        if (!this.nextAtomfile(v)) { exhausted = true; this.autoDelete(nm); }
      } else {
        throw new StyleError(`variable ${nm} (style ${v.style}) cannot be used with next`);
      }
    }
    return exhausted;
  }

  /**
   * variable.html (versionchanged 4Jul2026): "LAMMPS prints a warning when a file or atomfile style
   * variable is exhausted and auto-deleted." Measured with native LAMMPS (black box): the warning
   * is printed once, by the next command that exhausts the variable, as Auto-deleting variable a;
   * exhausted index and loop variables print nothing. A deleted variable is then undefined here, so
   * a later reference throws.
   */
  private autoDelete(nm: string): void {
    this.warn(`Auto-deleting variable ${nm}`);
    this.vars.delete(nm);
  }

  /** Text of a variable for $name / ${name} substitution. */
  text(name: string, env: FormulaEnv): string {
    const v = this.vars.get(name);
    if (!v) throw new StyleError(`Substitution for illegal variable ${name}`);
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
      case 'getenv': return this.getenv(v.values[0]) ?? '';
      case 'atom': throw new StyleError(`atom-style variable ${name} cannot be substituted with $ (it has one value per atom)`);
      case 'atomfile': throw new StyleError(`atomfile-style variable ${name} cannot be substituted with $`);
      default: return v.values[v.which];
    }
  }

  /**
   * Numeric value of an equal-compatible variable. As native LAMMPS does (Variable::compute_equal and retrieve,
   * variable.cpp at b332f28): an equal-style value below DBL_MIN in size is 0, and the text of an index, loop, string or
   * other text style variable must read as a number (utils::is_double).
   */
  scalar(name: string, env: FormulaEnv): number {
    const v = this.vars.get(name);
    if (!v) throw new StyleError(`variable ${name} is not defined`);
    switch (v.style) {
      case 'equal': {
        const r = this.guard(name, () => evaluate(v.formula!, env, 'equal'));
        if (typeof r !== 'number') throw new StyleError(`variable ${name} does not give a single value`);
        return roundUnderflow(r);
      }
      case 'internal': return v.num!;
      case 'timer': return (performance.now() - v.t0!) / 1000;
      case 'format': return this.scalar(v.format![0], env);
      case 'vector': case 'atom': case 'atomfile':
        throw new StyleError(`variable ${name} (style ${v.style}) does not give a single value`);
      default: {
        const s = v.style === 'getenv' ? this.getenv(v.values[0]) ?? '' : v.values[v.which];
        if (!IS_DOUBLE.test(s)) throw new FormulaError(`Variable ${name}: Non-numeric variable value in variable formula`, true);
        return Number(s);
      }
    }
  }

  /**
   * A vector-style variable's values. As native LAMMPS does (Variable::compute_vector, variable.cpp at b332f28): a formula
   * whose terms hold no vector has zero length, an error.
   */
  evalVector(name: string, env: FormulaEnv): Float64Array {
    const v = this.vars.get(name);
    if (!v || v.style !== 'vector') throw new StyleError(`variable ${name} is not a vector-style variable`);
    if (v.vec) return v.vec;
    return this.guard(name, () => {
      const r = evaluate(v.formula!, env, 'vector');
      if (typeof r === 'number' || r.length === 0) throw new FormulaError('Vector-style variable has zero length');
      return r;
    });
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

  /**
   * v_name in a formula in the given mode. As native LAMMPS does (Variable::evaluate, variable.cpp at b332f28): a
   * vector-, atom- or atomfile-style variable must suit the formula's style, in messages that name the referenced variable.
   */
  value(name: string, env: FormulaEnv, mode: Mode, ids: Int32Array, n: number): Value {
    const v = this.vars.get(name);
    if (!v) throw new StyleError(`Invalid variable reference v_${name} in variable formula`);
    const named = (text: string) => new FormulaError(`Variable ${name}: ${text}`, true);
    if (this.inProgress.has(name)) throw named('has a circular dependency');
    const formula = mode === 'equal' ? 'equal' : mode === 'vector' ? 'vector' : 'atom';
    if (v.style === 'atom' || v.style === 'atomfile') {
      const what = v.style === 'atom' ? 'Atom' : 'Atomfile';
      if (mode !== 'atom') throw named(`${what}-style variable in ${formula}-style variable formula`);
      return this.evalAtom(name, env, ids, n);
    }
    if (v.style === 'vector') {
      if (mode !== 'vector') throw named(`Vector-style variable in ${formula}-style variable formula`);
      return this.evalVector(name, env);
    }
    if (v.style === 'internal') return v.num!;
    // native reads the variable's text (Variable::retrieve: an equal- or timer-style value printed as %.15g, a format-style
    // one with its format) and then the number in it, so a referenced value keeps 15 significant digits
    const text = this.text(name, env);
    if (!IS_DOUBLE.test(text)) throw named('Non-numeric variable value in variable formula');
    return Number(text);
  }

  /**
   * Evaluates the formula of variable `name`. As native LAMMPS does (Variable::print_var_error and eval_in_progress,
   * variable.cpp at b332f28): an error of the formula that names no variable is given this one's name ("Variable NAME:
   * ..."), and reaching a variable again while it is evaluated is a circular dependency.
   */
  private guard<T>(name: string, f: () => T): T {
    if (this.inProgress.has(name)) throw new FormulaError(`Variable ${name}: has a circular dependency`, true);
    this.inProgress.add(name);
    try {
      return f();
    } catch (e) {
      if (e instanceof FormulaError && !e.named) throw new FormulaError(`Variable ${name}: ${e.message}`, true);
      throw e;
    } finally {
      this.inProgress.delete(name);
    }
  }

  /**
   * The integer of a [v_name] index. As native LAMMPS does (Variable::int_between_brackets, variable.cpp at b332f28): the
   * variable must exist and give text (not an atom- or atomfile-style one), which std::stod reads and the index truncates.
   */
  bracketValue(name: string, env: FormulaEnv): number {
    const v = this.vars.get(name);
    if (!v) throw new StyleError(`Invalid variable name ${name} in variable formula`);
    if (v.style === 'atom' || v.style === 'atomfile') throw new StyleError(`Invalid variable evaluation for variable ${name} in variable formula`);
    const m = /^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(this.text(name, env));
    if (!m) throw new StyleError(`Illegal value in brackets: stod(v_${name})`);
    return Math.trunc(Number(m[0]));
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
