import { StyleError } from './force/types';
import { libm } from './libm';

/*
 * Variable formulas — docs.lammps.org/variable.html. One parser builds an
 * AST (cached per formula string); the evaluator runs it in one of three
 * modes:
 *   equal  -> a scalar
 *   vector -> a global vector (Float64Array) or scalar
 *   atom   -> one value per owned atom (Float64Array of length n) or scalar
 *
 * Elements: numbers "0.2, 100, 1.0e20, -15.4"; constants "PI, version, on,
 * off, true, false, yes, no"; thermo keywords; operators "(), -x, x+y, x-y,
 * x*y, x/y, x^y, x%y, x == y, x != y, x < y, x <= y, x > y, x >= y, x && y,
 * x || y, x |^ y, !x" with the documented precedence ("unary minus and unary
 * logical NOT operator "!" have the highest precedence, exponentiation "^"
 * is next; multiplication and division and the modulo operator "%" are
 * next; addition and subtraction are next; the 4 relational operators "<",
 * "<=", ">", and ">=" are next; the two remaining relational operators "=="
 * and "!=" are next; then the logical AND operator "&&"; and finally the
 * logical OR operator "||" and logical XOR (exclusive or) operator "|^" have
 * the lowest precedence"; "-2^2" is 4); math functions; group and region
 * functions; special functions; atom values x[ID] and atom vectors x;
 * compute (c_, C_), fix (f_, F_) and variable (v_) references with [I] and
 * [I][J] indices, where an index may be v_name.
 * "Math functions that operate on scalar values produce a scalar value";
 * applied to per-atom or global vectors they act element by element.
 * ternary: "only the selected argument y or z is evaluated".
 * ramp: "value = x + (y-x) * (timestep-startstep) / (stopstep-startstep)
 * ... If called in between runs or during a run 0 command, the ramp(x,y)
 * function will return the value of x."
 */

export type Mode = 'equal' | 'vector' | 'atom';
export type Value = number | Float64Array;

/** errorurl(N) of native LAMMPS: the line that points at docs.lammps.org/errNNNN. */
export const errorUrl = (n: number): string => `\nFor more information see https://docs.lammps.org/err${String(n).padStart(4, '0')}`;

/**
 * An error of a variable formula. As native LAMMPS does (src/variable.cpp Variable::print_var_error at b332f28), an error
 * found while a named variable is evaluated reads "Variable NAME: text": the variable store adds the name of the variable
 * it evaluates (Variables.guard) unless the error names one already (`named`). Errors native raises without a name
 * (error->all and error->one) are plain StyleErrors.
 */
export class FormulaError extends StyleError {
  constructor(message: string, readonly named = false) { super(message); }
}

const SYNTAX = 'Invalid syntax in variable formula';
const MATH_FN = 'Invalid math function in variable formula';

/**
 * A bracketed index as written. As native LAMMPS does (src/variable.cpp Variable::int_between_brackets at b332f28), the
 * text is a v_name or digits with + - * / signs read as an integer; the checks are raised when the formula is evaluated,
 * after the compute, fix, variable or atom vector in front of the brackets was looked up (native's order).
 */
export type Index = { text: string; error: string | null; n: number; variable: string | null };

type Node =
  | { t: 'num'; v: number }
  | { t: 'name'; name: string }
  | { t: 'un'; op: '-' | '!'; a: Node }
  | { t: 'bin'; op: string; a: Node; b: Node }
  | { t: 'call'; fn: string; args: Node[] }
  | { t: 'raw'; fn: string; args: string[]; index: Index | null }
  | { t: 'ref'; kind: 'c' | 'C' | 'f' | 'F' | 'v'; id: string; i: Index | null; j: Index | null }
  | { t: 'atomval'; name: string; index: Index }
  | { t: 'custom'; name: string; col: Index | null; index: Index | null };

export const ATOM_VECTORS = ['id', 'mass', 'type', 'mol', 'radius', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'q'];

/** Functions whose arguments are IDs or references, not formulas. */
const RAW_FUNCS = new Set([
  'count', 'mass', 'charge', 'xcm', 'vcm', 'fcm', 'bound', 'gyration', 'ke', 'angmom', 'torque', 'inertia', 'omega',
  'sum', 'min', 'max', 'ave', 'trap', 'slope', 'sort', 'rsort', 'gmask', 'rmask', 'grmask', 'next',
  'is_file', 'is_os', 'extract_setting', 'label2type', 'is_typelabel', 'is_timeout',
  'is_available', 'is_active', 'is_defined',
]);

const MATH: Record<string, number> = {
  sqrt: 1, exp: 1, ln: 1, log: 1, abs: 1, sign: 1, sin: 1, cos: 1, tan: 1, asin: 1, acos: 1, atan: 1,
  atan2: 2, random: 3, normal: 3, ceil: 1, floor: 1, round: 1, ternary: 3, ramp: 2, stagger: 2,
  logfreq: 3, logfreq2: 3, logfreq3: 3, stride: 3, stride2: 6, vdisplace: 2, swiggle: 3, cwiggle: 3,
};

/** native's messages for a math function with the wrong number of arguments (math_function, variable.cpp at b332f28) */
const MATH_ARGS: Record<string, string> = {
  vdisplace: 'Invalid vdisplace function in variable formula: must have 2 arguments',
  swiggle: 'Invalid swiggle function in variable formula: must have 3 arguments',
  cwiggle: 'Invalid cwiggle function in variable formula: must have 3 arguments',
};

const CONSTANTS: Record<string, number> = { PI: Math.PI, on: 1, true: 1, yes: 1, off: 0, false: 0, no: 0 };

/** The smallest normal double: native rounds an equal-style result below it to zero (Variable::compute_equal). */
const DBL_MIN = 2.2250738585072014e-308;

type Tok =
  | { k: 'num'; v: number; at: number }
  | { k: 'id'; v: string; at: number }
  | { k: 'op'; v: string; at: number }
  | { k: 'bad'; v: string; at: number; error?: string };
const OPS = ['&&', '||', '|^', '==', '!=', '<=', '>=', '<', '>', '+', '-', '*', '/', '%', '^', '!', '(', ')', ',', '[', ']'];

/**
 * Tokens of a formula; a character that starts no token becomes a 'bad' token, a syntax error where the parser reaches
 * it. As native LAMMPS does (src/variable.cpp Variable::evaluate at b332f28): a number is a run of digits and periods
 * with an optional exponent, read as its longest leading number (std::stod: 1.2.3 is 1.2); a word starts with a letter.
 */
const lex = (src: string): Tok[] => {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (/\s/.test(c)) { i++; continue; }
    const num = /^[0-9.]+(?:[eE][+-]?[0-9]*)?/.exec(src.slice(i));
    if (num) {
      const v = Number.parseFloat(num[0]);
      // std::stod refuses a number past the range of a double, or one that underflows below DBL_MIN; native LAMMPS then
      // stops with "Exception: stod" (measured with native LAMMPS: 1e400, 1e-320)
      const mantissa = /^[0-9.]*/.exec(num[0])![0];
      const range = !Number.isFinite(v) || (/[1-9]/.test(mantissa) && Math.abs(v) < DBL_MIN);
      if (Number.isNaN(v)) out.push({ k: 'bad', v: num[0], at: i });
      else if (range) out.push({ k: 'bad', v: num[0], at: i, error: `Exception: stod (${num[0]} is outside the range of a double)` });
      else out.push({ k: 'num', v, at: i });
      i += num[0].length;
      continue;
    }
    const id = /^[A-Za-z][A-Za-z0-9_]*/.exec(src.slice(i));
    if (id) { out.push({ k: 'id', v: id[0], at: i }); i += id[0].length; continue; }
    const op = OPS.find((o) => src.startsWith(o, i));
    if (op) { out.push({ k: 'op', v: op, at: i }); i += op.length; continue; }
    out.push({ k: 'bad', v: c, at: i });
    i++;
  }
  return out;
};

/**
 * The bracketed index whose text starts at src[start] (after the '['); end is the position of the ']' (or the end of
 * src). varallow: a v_name index is allowed. As native LAMMPS does (Variable::int_between_brackets, variable.cpp at
 * b332f28): a bad character fails first, then a missing ']' and empty brackets; digits are read as std::stoll does.
 */
export const scanBracket = (src: string, start: number, varallow: boolean): { index: Index; end: number } => {
  const isVar = varallow && src.startsWith('v_', start);
  let q = start;
  let error: string | null = null;
  for (; q < src.length && src[q] !== ']'; q++) {
    if (error) continue;
    if (isVar ? !/[A-Za-z0-9_]/.test(src[q]) : !/[0-9+\-*/]/.test(src[q])) {
      error = isVar ? 'Variable name between brackets must be letters, numbers, or underscores' : 'Non digit character between brackets in variable';
    }
  }
  if (!error && q >= src.length) error = 'Mismatched brackets in variable';
  if (!error && q === start) error = 'Empty brackets in variable';
  const text = src.slice(start, q);
  let n = 0;
  if (!error && !isVar) {
    // std::stoll: an optional sign and digits (the rest is ignored); none, or a value past 64 bits, is an error
    const m = /^[+-]?\d+/.exec(text);
    if (!m || BigInt(m[0]) > 9223372036854775807n || BigInt(m[0]) < -9223372036854775808n) error = `Illegal value in brackets: stoll(${text})`;
    else n = Number(m[0]);
  }
  return { index: { text, error, n, variable: isVar ? text.slice(2) : null }, end: q };
};

const cache = new Map<string, Node>();

export const parseFormula = (src: string): Node => {
  const hit = cache.get(src);
  if (hit) return hit;
  const toks = lex(src);
  let p = 0;
  const syntax = () => new FormulaError(SYNTAX);
  const peek = () => toks[p];
  const isOp = (v: string) => toks[p]?.k === 'op' && (toks[p] as { v: string }).v === v;
  const expect = (v: string) => {
    if (!isOp(v)) throw syntax();
    p++;
  };
  /** the position of the ')' that closes the '(' at src[open] (native find_matching_paren counts parentheses only) */
  const closing = (open: number): number => {
    let level = 0;
    for (let q = open + 1; q < src.length; q++) {
      if (src[q] === '(') level++;
      else if (src[q] === ')') { if (level === 0) return q; level--; }
    }
    throw syntax();
  };
  const skipTo = (pos: number) => { while (p < toks.length && toks[p].at <= pos) p++; };
  const LEVELS = [['||', '|^'], ['&&'], ['==', '!='], ['<', '<=', '>', '>='], ['+', '-'], ['*', '/', '%']];

  const binary = (level: number): Node => {
    if (level === LEVELS.length) return power();
    let a = binary(level + 1);
    for (;;) {
      const t = peek();
      if (!t || t.k !== 'op' || !LEVELS[level].includes(t.v)) return a;
      p++;
      a = { t: 'bin', op: t.v, a, b: binary(level + 1) };
    }
  };
  // "^" binds tighter than * /, below unary minus ("-2^2" is 4); left-associative like the other operators
  const power = (): Node => {
    let a = unary();
    while (isOp('^')) { p++; a = { t: 'bin', op: '^', a, b: unary() }; }
    return a;
  };
  // native has no unary plus: a '+' where a value is expected is a syntax error
  const unary = (): Node => {
    if (isOp('-')) { p++; return { t: 'un', op: '-', a: unary() }; }
    if (isOp('!')) { p++; return { t: 'un', op: '!', a: unary() }; }
    return primary();
  };
  /** the brackets at toks[p] (a '[') */
  const bracket = (): Index => {
    const r = scanBracket(src, toks[p].at + 1, true);
    skipTo(r.end);
    return r.index;
  };
  /** the comma-separated arguments of the function whose '(' is at toks[p], as text (native parse_args: trimmed) */
  const rawArgs = (): string[] => {
    const open = toks[p].at;
    const close = closing(open);
    skipTo(close);
    const body = src.slice(open + 1, close);
    const out: string[] = [];
    let level = 0, from = 0;
    for (let q = 0; q < body.length; q++) {
      if (body[q] === '(') level++;
      else if (body[q] === ')') level--;
      else if (body[q] === ',' && level === 0) { out.push(body.slice(from, q).trim()); from = q + 1; }
    }
    out.push(body.slice(from).trim());
    return out;
  };
  const primary = (): Node => {
    const t = peek();
    if (t?.k === 'bad' && t.error) throw new StyleError(t.error);
    if (!t || t.k === 'bad') throw syntax();
    if (t.k === 'num') { p++; return { t: 'num', v: t.v }; }
    if (t.k === 'op' && t.v === '(') { p++; const v = binary(0); expect(')'); return v; }
    if (t.k !== 'id') throw syntax();
    p++;
    const name = t.v;
    if (isOp('(')) {
      if (RAW_FUNCS.has(name)) {
        const args = rawArgs();
        // sort() and rsort() of an equal-style formula take the element in the brackets that follow (special_function)
        const index = (name === 'sort' || name === 'rsort') && isOp('[') ? bracket() : null;
        return { t: 'raw', fn: name, args, index };
      }
      if (!(name in MATH)) {
        closing(toks[p].at);
        if (name.startsWith('py_')) throw new FormulaError('Invalid python function variable name');
        throw new FormulaError(`Invalid math/group/special/feature function '${name}()' in variable formula`);
      }
      p++;
      // an empty argument is an empty formula, a syntax error (native evaluates each argument of parse_args)
      const args: Node[] = [binary(0)];
      while (isOp(',')) { p++; args.push(binary(0)); }
      expect(')');
      if (args.length !== MATH[name]) throw new FormulaError(MATH_ARGS[name] ?? MATH_FN);
      return { t: 'call', fn: name, args };
    }
    const m = /^([cCfFv])_(.+)$/.exec(name);
    if (m) {
      // c_, C_, f_ and F_ take up to two brackets, v_ one (a further bracket is a syntax error)
      let i: Index | null = null, j: Index | null = null;
      if (isOp('[')) i = bracket();
      if (m[1] !== 'v' && i && isOp('[')) j = bracket();
      return { t: 'ref', kind: m[1] as 'c' | 'C' | 'f' | 'F' | 'v', id: m[2], i, j };
    }
    // variable.html: custom atom properties "i_name[I]" (atom ID I), "i2_name[I][J]", and in
    // atom-style variables "i_name" (per-atom vector) and "i2_name[I]" (column of per-atom array).
    // Measured: native LAMMPS (2 Sep 2026) reads J of i2_name[I][J] as 0-based (J = Ncol runs into
    // the next atom's row) and gives inconsistent atom-style column sums; the engine follows the
    // documented 1-based columns.
    if (/^[id]2?_[A-Za-z0-9_]+$/.test(name)) {
      const a1 = isOp('[') ? bracket() : null;
      const a2 = a1 && isOp('[') ? bracket() : null;
      return { t: 'custom', name, col: a2 ? a2 : null, index: a1 };
    }
    if (isOp('[')) return { t: 'atomval', name, index: bracket() };
    return { t: 'name', name };
  };
  const node = binary(0);
  if (p !== toks.length) throw syntax();
  cache.set(src, node);
  return node;
};

/** What a formula can reach; implemented by the session. */
export interface FormulaEnv {
  /** A name that is no constant: an atom vector (per-atom values in atom mode) or a thermo keyword. */
  name(name: string, mode: Mode): Value;
  /** v_name and v_name[I]: equal -> number; atom/vector -> per-atom/global values. */
  variable(name: string, index: Index | null, mode: Mode): Value;
  /** c_/C_/f_/F_ references with their brackets as written. */
  reference(kind: 'c' | 'C' | 'f' | 'F', id: string, i: Index | null, j: Index | null, mode: Mode): Value;
  /** NAME[ID]: the atom vector NAME of the atom with that ID (0.0 for an ID that does not exist). */
  atomValue(name: string, index: Index): number;
  /** i_name, d_name, i2_name and d2_name with their brackets. */
  custom(name: string, b1: Index | null, b2: Index | null, mode: Mode): Value;
  /** The integer of a [v_name] index: the variable's value, truncated. */
  indexVariable(name: string): number;
  /** Group / region / special / feature functions with raw arguments; index: sort()[I] and rsort()[I]. */
  raw(fn: string, args: string[], mode: Mode, index: Index | null): Value;
  /** Current step and run bounds for ramp() etc. */
  timing(): { step: number; startStep: number; stopStep: number; dt: number; inRun: boolean; firstStep: number };
  random(lo: number, hi: number, seed: number, mode: Mode): Value;
  normal(mu: number, sigma: number, seed: number, mode: Mode): Value;
  /** Number of owned atoms (length of atom-mode values). */
  natoms(): number;
  /** A fix of this style is defined (vdisplace, swiggle and cwiggle refuse fix dt/reset). */
  hasFixStyle(style: string): boolean;
}

const isVec = (v: Value): v is Float64Array => typeof v !== 'number';

const map1 = (a: Value, f: (x: number) => number): Value => {
  if (!isVec(a)) return f(a);
  const out = new Float64Array(a.length);
  for (let k = 0; k < a.length; k++) out[k] = f(a[k]);
  return out;
};

const map2 = (a: Value, b: Value, f: (x: number, y: number) => number, mode: Mode): Value => {
  if (!isVec(a) && !isVec(b)) return f(a, b);
  const n = isVec(a) ? a.length : (b as Float64Array).length;
  // native size_tree_vector: the vectors of a vector-style formula have one length
  if (isVec(a) && isVec(b) && a.length !== b.length) {
    throw mode === 'vector' ? new FormulaError('Inconsistent lengths in vector-style variable') : new StyleError('vector lengths do not match in a formula');
  }
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) out[k] = f(isVec(a) ? a[k] : a, isVec(b) ? b[k] : b);
  return out;
};

/**
 * An error of the arithmetic of a formula. As native LAMMPS does (src/variable.cpp at b332f28): an equal-style formula
 * is computed while it is parsed and names the variable (print_var_error); the tree of a vector- or atom-style formula is
 * computed afterwards (collapse_tree, eval_tree), whose errors have no name (error->one).
 */
const arith = (text: string, mode: Mode): StyleError => (mode === 'equal' ? new FormulaError(text) : new StyleError(text));

const truth = (b: boolean) => (b ? 1 : 0);

/**
 * round(x): the nearest integer, with a halfway value going toward +infinity. Measured with native LAMMPS
 * (black box): round(-2.5) gives -2, round(2.5) gives 3, round(-0.5) gives 0, round(-1.4) gives -1.
 * variable.html: "Round() is the nearest integer to its argument." (the halfway rule is not stated there)
 */
const cround = (x: number) => {
  const f = Math.floor(x);
  return x - f >= 0.5 ? f + 1 : f;
};

/**
 * The step-sequence functions take integer arguments: native LAMMPS truncates each one (black box:
 * stagger(5.5,2) starts with 2 as stagger(5,2) does, logfreq(5.5,2,10) at step 0 gives 5, and logfreq3(1.5,3,10)
 * gives the sequence of logfreq3(1,3,10)).
 */
const ints = (a: number[]) => a.map((v) => Math.trunc(v));

/**
 * The value of stride() and stride2() past their last timestep: native's MAXBIGINT_DOUBLE, the largest 64-bit integer
 * less 512 as a double (src/variable.cpp at b332f28), which prints as 9.2233720368547748e+18.
 */
const NO_MORE_STEPS = 9223372036854774784;

export const evaluate = (src: string, env: FormulaEnv, mode: Mode): Value => evalNode(parseFormula(src), env, mode, src);


/** An immediate formula $(...): as native LAMMPS does (Variable::compute_equal(str), variable.cpp at b332f28), a value below DBL_MIN in size is 0. */
export const evaluateScalar = (src: string, env: FormulaEnv): number => {
  const v = evaluate(src, env, 'equal');
  if (isVec(v)) throw new StyleError(`formula "${src}" gives a vector where a single value is needed`);
  return Math.abs(v) < DBL_MIN ? 0 : v;
};

/** native's rounding of an equal-style variable's value (|x| < DBL_MIN gives 0, -0 included) */
export const roundUnderflow = (v: number): number => (Math.abs(v) < DBL_MIN ? 0 : v);

/**
 * The integer of a bracketed index. As native LAMMPS does (Variable::int_between_brackets, variable.cpp at b332f28): a
 * malformed bracket fails first, then a v_name index reads the variable, and the index must be positive.
 */
export const resolveIndex = (ix: Index, env: FormulaEnv): number => {
  if (ix.error) throw new StyleError(ix.error);
  const n = ix.variable !== null ? env.indexVariable(ix.variable) : ix.n;
  if (n <= 0) throw new StyleError('Index between variable brackets must be positive');
  return n;
};

const evalNode = (n: Node, env: FormulaEnv, mode: Mode, src: string): Value => {
  switch (n.t) {
    case 'num': return n.v;
    case 'un': {
      const a = evalNode(n.a, env, mode, src);
      return n.op === '-' ? map1(a, (x) => -x) : map1(a, (x) => truth(x === 0));
    }
    case 'bin': {
      const a = evalNode(n.a, env, mode, src);
      const b = evalNode(n.b, env, mode, src);
      const m2 = (f: (x: number, y: number) => number) => map2(a, b, f, mode);
      switch (n.op) {
        case '+': return m2((x, y) => x + y);
        case '-': return m2((x, y) => x - y);
        case '*': return m2((x, y) => x * y);
        case '/': return m2((x, y) => {
          if (y === 0) throw arith('Divide by 0 in variable formula', mode);
          return x / y;
        });
        case '%': return m2((x, y) => {
          if (y === 0) throw arith('Modulo 0 in variable formula', mode);
          return x % y;
        });
        case '^': return m2((x, y) => {
          if (y === 0) return 1;
          if (x === 0 && y < 0) throw arith('Invalid power expression in variable formula', mode);
          return Math.pow(x, y);
        });
        case '<': return m2((x, y) => truth(x < y));
        case '<=': return m2((x, y) => truth(x <= y));
        case '>': return m2((x, y) => truth(x > y));
        case '>=': return m2((x, y) => truth(x >= y));
        case '==': return m2((x, y) => truth(x === y));
        case '!=': return m2((x, y) => truth(x !== y));
        case '&&': return m2((x, y) => truth(x !== 0 && y !== 0));
        case '||': return m2((x, y) => truth(x !== 0 || y !== 0));
        case '|^': return m2((x, y) => truth((x !== 0) !== (y !== 0)));
      }
      throw new StyleError(`unknown operator ${n.op}`);
    }
    case 'name': {
      if (n.name in CONSTANTS) return CONSTANTS[n.name];
      if (n.name === 'version') return 20260902;
      return env.name(n.name, mode);
    }
    case 'atomval': return env.atomValue(n.name, n.index);
    case 'custom': return env.custom(n.name, n.index, n.col, mode);
    case 'ref':
      if (n.kind === 'v') return env.variable(n.id, n.i, mode);
      return env.reference(n.kind, n.id, n.i, n.j, mode);
    case 'raw': return env.raw(n.fn, n.args, mode, n.index);
    case 'call': return callMath(n, env, mode, src);
  }
};

const callMath = (n: Extract<Node, { t: 'call' }>, env: FormulaEnv, mode: Mode, src: string): Value => {
  const arg = (k: number) => evalNode(n.args[k], env, mode, src);
  const scalarArgs = () => n.args.map((a, k) => {
    const v = evalNode(a, env, mode, src);
    if (isVec(v)) throw new StyleError(`argument ${k + 1} of ${n.fn}() must be a single value`);
    return v;
  });
  switch (n.fn) {
    case 'sqrt': return map1(arg(0), (x) => {
      if (x < 0) throw arith('Sqrt of negative value in variable formula', mode);
      return Math.sqrt(x);
    });
    case 'exp': return map1(arg(0), (x) => libm.exp(x));
    case 'ln': return map1(arg(0), (x) => {
      if (x <= 0) throw arith('Log of zero/negative value in variable formula', mode);
      return libm.log(x);
    });
    case 'log': return map1(arg(0), (x) => {
      if (x <= 0) throw arith('Log of zero/negative value in variable formula', mode);
      return Math.log10(x);
    });
    case 'abs': return map1(arg(0), Math.abs);
    case 'sign': return map1(arg(0), (x) => (x >= 0 ? 1 : -1));
    case 'sin': return map1(arg(0), (x) => libm.sin(x));
    case 'cos': return map1(arg(0), (x) => libm.cos(x));
    case 'tan': return map1(arg(0), Math.tan);
    case 'asin': return map1(arg(0), (x) => {
      if (x < -1 || x > 1) throw arith('Arcsin of invalid value in variable formula', mode);
      return Math.asin(x);
    });
    case 'acos': return map1(arg(0), (x) => {
      if (x < -1 || x > 1) throw arith('Arccos of invalid value in variable formula', mode);
      return libm.acos(x);
    });
    case 'atan': return map1(arg(0), Math.atan);
    case 'atan2': return map2(arg(0), arg(1), Math.atan2, mode);
    case 'ceil': return map1(arg(0), Math.ceil);
    case 'floor': return map1(arg(0), Math.floor);
    case 'round': return map1(arg(0), cround);
    case 'ternary': {
      const c = arg(0);
      if (!isVec(c)) return c !== 0 ? arg(1) : arg(2);
      return map2(map2(c, arg(1), (x, y) => (x !== 0 ? y : Number.NaN), mode), arg(2), (x, z) => (Number.isNaN(x) ? z : x), mode);
    }
    case 'random': { const [lo, hi, seed] = scalarArgs(); return env.random(lo, hi, seed, mode); }
    case 'normal': {
      const [mu, sg, seed] = scalarArgs();
      if (sg < 0) throw arith(MATH_FN, mode);
      return env.normal(mu, sg, seed, mode);
    }
    default: break;
  }
  // native checks fix dt/reset while it parses the call, in every style of formula (math_function)
  if ((n.fn === 'vdisplace' || n.fn === 'swiggle' || n.fn === 'cwiggle') && env.hasFixStyle('dt/reset')) {
    throw new FormulaError(`Must not use ${n.fn}(${n.fn === 'vdisplace' ? 'x,y' : 'x,y,z'}) function with fix dt/reset`);
  }
  const a = scalarArgs();
  const tm = env.timing();
  const step = tm.step;
  switch (n.fn) {
    case 'ramp': {
      if (!tm.inRun || tm.stopStep === tm.startStep) return a[0];
      return a[0] + (a[1] - a[0]) * (step - tm.startStep) / (tm.stopStep - tm.startStep);
    }
    case 'vdisplace': return a[0] + a[1] * (step - tm.startStep) * tm.dt;
    case 'swiggle':
      if (a[2] === 0) throw arith('Invalid swiggle(x,y,z) function in variable formula: z must be > 0', mode);
      return a[0] + a[1] * libm.sin(2 * Math.PI / a[2] * (step - tm.startStep) * tm.dt);
    case 'cwiggle':
      if (a[2] === 0) throw arith('Invalid cwiggle(x,y,z) function in variable formula: z must be > 0', mode);
      return a[0] + a[1] * (1 - libm.cos(2 * Math.PI / a[2] * (step - tm.startStep) * tm.dt));
    case 'stagger': {
      const [x, y] = ints(a);
      if (!(x > 0 && y > 0 && x > y)) throw arith(MATH_FN, mode);
      const lower = Math.floor(step / x) * x;
      const delta = step - lower;
      return delta < y ? lower + y : lower + x;
    }
    case 'logfreq': {
      const [x, y, z] = ints(a);
      if (!(x > 0 && y > 0 && z > 0 && y < z)) throw arith(MATH_FN, mode);
      if (step < x) return x;
      let lower = x;
      while (step >= z * lower) lower *= z;
      const multiple = Math.floor(step / lower);
      return multiple < y ? (multiple + 1) * lower : lower * z;
    }
    case 'logfreq2': {
      // variable.html: "a single logarithmic interval is divided into y equally-spaced timesteps and all of them are
      // output". As native LAMMPS does (math_function, variable.cpp at b332f28): from x the points advance by
      // delta = x*(z-1)/y, delta grows by z after every y points, and the value is the ceiling of the first point above
      // the step, the points summed one by one (in floating point, as native adds them). x, y and z must be positive;
      // with z = 1 native's interval never grows and its loop does not end once the step reaches x, which the engine
      // refuses as well.
      const [x, y, z] = ints(a);
      if (!(x > 0 && y > 0 && z > 1)) throw arith(MATH_FN, mode);
      if (step < x) return x;
      let value = x;
      let delta = x * (z - 1) / y;
      for (let count = 1; step >= value; count++) {
        value += delta;
        if (count % y === 0) delta *= z;
      }
      return Math.ceil(value);
    }
    case 'logfreq3': {
      // variable.html: "generates y points between x and z (inclusive), that are separated by a multiplicative ratio:
      // (z/x)^(1/(y-1))", e.g. "10, 13, 15, 18, 22, 27, 32,...384, 465, 563, 682, 826, 1000". As native LAMMPS does
      // (math_function, variable.cpp at b332f28): the k-th point after x is k + x when that is above x*ratio^k, else
      // c - floor((c - 1)/z) with c = ceil(x*ratio^k), the products taken one by one; the value is the first point above
      // the step. Native gives a point past z at step z (29 for logfreq3(1,3,10), measured with native LAMMPS), and past
      // step z it stops with "Calls to variable exceeded limit".
      const [x, y, z] = ints(a);
      if (!(x > 0 && z > 0 && y > 1 && z - x >= y - 1)) throw arith(MATH_FN, mode);
      let value = x;
      if (step >= x) {
        const factor = Math.pow(z / x, 1 / (y - 1));
        let logsp = x;
        let linsp = x;
        while (step >= value) {
          logsp *= factor;
          linsp++;
          const c = Math.ceil(logsp);
          value = linsp > logsp ? linsp : c - Math.trunc((c - 1) / z);
        }
      }
      if (step > z) throw new StyleError('Calls to variable exceeded limit');
      return value;
    }
    case 'stride': {
      const [x, y, z] = ints(a);
      if (!(x >= 0 && y >= 0 && z > 0 && x <= y)) throw new StyleError(MATH_FN);
      if (step < x) return x;
      if (step >= y) return NO_MORE_STEPS;
      const next = x + (Math.floor((step - x) / z) + 1) * z;
      return next > y ? NO_MORE_STEPS : next;
    }
    case 'stride2': {
      // variable.html: "generates two sets of strided timesteps": x, x+z, ... up to a; then increments of c from a to b;
      // then increments of z again. The documented example stride2(1000,2000,100,1350,1360,1) is reproduced. Measured with
      // native LAMMPS (black box), stride2(2,20,5,7,12,3) at steps 0..19: 2 2 7 7 7 7 7 10 10 10 17 17 17 17 17 17 17 22 22 22,
      // so a fine step that would pass b continues on the z grid above b (17, not 12), and the result is not checked
      // against y (22 at step 19, the limit is reached at step 20).
      const [x, y, z, aa, b, c] = ints(a);
      if (!(x >= 0 && y >= 0 && z > 0 && x <= y && aa >= 0 && b >= 0 && c > 0 && aa < b && aa >= x && b <= y)) {
        throw new StyleError(MATH_FN);
      }
      if (step < x) return x;
      if (step >= y) return NO_MORE_STEPS;
      const gridAbove = (v: number) => x + (Math.floor((v - x) / z) + 1) * z;
      if (step < aa) {
        const next = gridAbove(step);
        return next > aa ? aa : next;
      }
      if (step < b) {
        const fine = aa + (Math.floor((step - aa) / c) + 1) * c;
        return fine <= b ? fine : gridAbove(b);
      }
      return gridAbove(step);
    }
  }
  throw new StyleError(`unknown function ${n.fn}()`);
};

/** Collects v_name references of a formula (for dependency checks). */
export const formulaVariables = (src: string): string[] => {
  const out: string[] = [];
  const walk = (n: Node) => {
    if (n.t === 'ref' && n.kind === 'v') out.push(n.id);
    if (n.t === 'un') walk(n.a);
    if (n.t === 'bin') { walk(n.a); walk(n.b); }
    if (n.t === 'call') n.args.forEach(walk);
  };
  walk(parseFormula(src));
  return out;
};
