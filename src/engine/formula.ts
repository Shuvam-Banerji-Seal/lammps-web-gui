import { StyleError } from './force/types';

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

type Node =
  | { t: 'num'; v: number }
  | { t: 'name'; name: string }
  | { t: 'un'; op: '-' | '!'; a: Node }
  | { t: 'bin'; op: string; a: Node; b: Node }
  | { t: 'call'; fn: string; args: Node[] }
  | { t: 'raw'; fn: string; args: string[] }
  | { t: 'ref'; kind: 'c' | 'C' | 'f' | 'F' | 'v'; id: string; i: Index | null; j: Index | null }
  | { t: 'atomval'; name: string; index: Index };

export type Index = number | { variable: string };

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

const CONSTANTS: Record<string, number> = { PI: Math.PI, on: 1, true: 1, yes: 1, off: 0, false: 0, no: 0 };

type Tok = { k: 'num'; v: number; s: string } | { k: 'id'; v: string } | { k: 'op'; v: string };
const OPS = ['&&', '||', '|^', '==', '!=', '<=', '>=', '<', '>', '+', '-', '*', '/', '%', '^', '!', '(', ')', ',', '[', ']'];

const lex = (src: string): Tok[] => {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i++; continue; }
    const num = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(src.slice(i));
    if (num) { out.push({ k: 'num', v: Number(num[0]), s: num[0] }); i += num[0].length; continue; }
    const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
    if (id) { out.push({ k: 'id', v: id[0] }); i += id[0].length; continue; }
    const op = OPS.find((o) => src.startsWith(o, i));
    if (op) { out.push({ k: 'op', v: op }); i += op.length; continue; }
    throw new StyleError(`invalid character '${c}' in formula "${src}"`);
  }
  return out;
};

const cache = new Map<string, Node>();

export const parseFormula = (src: string): Node => {
  const hit = cache.get(src);
  if (hit) return hit;
  const toks = lex(src);
  let p = 0;
  const peek = () => toks[p];
  const isOp = (v: string) => toks[p]?.k === 'op' && (toks[p] as { v: string }).v === v;
  const expect = (v: string) => {
    if (!isOp(v)) throw new StyleError(`expected '${v}' in formula "${src}"`);
    p++;
  };
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
  const unary = (): Node => {
    if (isOp('-')) { p++; return { t: 'un', op: '-', a: unary() }; }
    if (isOp('!')) { p++; return { t: 'un', op: '!', a: unary() }; }
    if (isOp('+')) { p++; return unary(); }
    return primary();
  };
  const index = (): Index => {
    const t = peek();
    if (t?.k === 'num' && Number.isInteger(t.v)) { p++; return t.v; }
    if (t?.k === 'id' && t.v.startsWith('v_')) { p++; return { variable: t.v.slice(2) }; }
    if (t?.k === 'op' && t.v === '*') throw new StyleError(`wildcard indices are not allowed in formulas ("${src}")`);
    throw new StyleError(`an index must be an integer or v_name in formula "${src}"`);
  };
  const rawArgs = (): string[] => {
    // tokens up to the matching ')', split at depth-0 commas, re-joined as text
    expect('(');
    const out: string[] = [];
    let cur = '';
    let depth = 0;
    for (;;) {
      const t = toks[p];
      if (!t) throw new StyleError(`unbalanced parentheses in formula "${src}"`);
      p++;
      if (t.k === 'op' && (t.v === '(' || t.v === '[')) depth++;
      if (t.k === 'op' && (t.v === ')' || t.v === ']')) {
        if (depth === 0 && t.v === ')') { if (cur || out.length) out.push(cur); break; }
        depth--;
      }
      if (t.k === 'op' && t.v === ',' && depth === 0) { out.push(cur); cur = ''; continue; }
      cur += t.k === 'num' ? t.s : t.v;
    }
    return out;
  };
  const primary = (): Node => {
    const t = peek();
    if (!t) throw new StyleError(`formula "${src}" ends unexpectedly`);
    if (t.k === 'num') { p++; return { t: 'num', v: t.v }; }
    if (t.k === 'op' && t.v === '(') { p++; const v = binary(0); expect(')'); return v; }
    if (t.k !== 'id') throw new StyleError(`unexpected '${t.v}' in formula "${src}"`);
    p++;
    const name = t.v;
    if (isOp('(')) {
      if (RAW_FUNCS.has(name)) return { t: 'raw', fn: name, args: rawArgs() };
      if (!(name in MATH)) throw new StyleError(`unknown function ${name}() in formula "${src}"`);
      p++;
      const args: Node[] = [];
      if (!isOp(')')) {
        args.push(binary(0));
        while (isOp(',')) { p++; args.push(binary(0)); }
      }
      expect(')');
      if (args.length !== MATH[name]) throw new StyleError(`${name}() takes ${MATH[name]} argument(s), got ${args.length}`);
      return { t: 'call', fn: name, args };
    }
    const m = /^([cCfFv])_(.+)$/.exec(name);
    if (m) {
      let i: Index | null = null, j: Index | null = null;
      if (isOp('[')) { p++; i = index(); expect(']'); }
      if (isOp('[')) { p++; j = index(); expect(']'); }
      return { t: 'ref', kind: m[1] as 'c' | 'C' | 'f' | 'F' | 'v', id: m[2], i, j };
    }
    if (ATOM_VECTORS.includes(name) && isOp('[')) {
      p++;
      const ix = index();
      expect(']');
      return { t: 'atomval', name, index: ix };
    }
    // variable.html: custom atom properties "i_name[I]" (atom ID I), "i2_name[I][J]", and in
    // atom-style variables "i_name" (per-atom vector) and "i2_name[I]" (column of per-atom array).
    // Measured: native LAMMPS (2 Sep 2026) reads J of i2_name[I][J] as 0-based (J = Ncol runs into
    // the next atom's row) and gives inconsistent atom-style column sums; the engine follows the
    // documented 1-based columns.
    const cm = /^[id](2?)_[A-Za-z0-9_]+$/.exec(name);
    if (cm) {
      if (!isOp('[')) {
        if (cm[1]) throw new StyleError(`${name} is a per-atom array: give a column, ${name}[J], in formula "${src}"`);
        return { t: 'name', name };
      }
      p++;
      const a1 = index();
      expect(']');
      if (!cm[1]) return { t: 'atomval', name, index: a1 };
      if (!isOp('[')) {
        if (typeof a1 !== 'number') throw new StyleError(`the column of ${name} must be an integer in formula "${src}"`);
        return { t: 'name', name: `${name}[${a1}]` };
      }
      p++;
      const a2 = index();
      expect(']');
      if (typeof a2 !== 'number') throw new StyleError(`the column of ${name} must be an integer in formula "${src}"`);
      return { t: 'atomval', name: `${name}[${a2}]`, index: a1 };
    }
    return { t: 'name', name };
  };
  const node = binary(0);
  if (p !== toks.length) throw new StyleError(`unexpected '${(toks[p] as { v: unknown }).v}' in formula "${src}"`);
  cache.set(src, node);
  return node;
};

/** What a formula can reach; implemented by the session. */
export interface FormulaEnv {
  /** Thermo keyword value, or undefined if `name` is not one. */
  thermo(name: string): number | undefined;
  /** v_name: equal -> number; atom/vector -> per-atom/global values. */
  variable(name: string, mode: Mode): Value;
  /** v_name[i] (vector-style variable element, 1-based). */
  variableElement(name: string, i: number): number;
  /** c_/C_/f_/F_ references. i, j are 1-based or null. */
  reference(kind: 'c' | 'C' | 'f' | 'F', id: string, i: number | null, j: number | null, mode: Mode): Value;
  /** x[ID] etc.; 0.0 for an ID that does not exist. */
  atomValue(name: string, id: number): number;
  /** Per-atom vector (atom mode). */
  atomVector(name: string): Float64Array;
  /** Group / region / special / feature functions with raw arguments. */
  raw(fn: string, args: string[], mode: Mode): Value;
  /** Current step and run bounds for ramp() etc. */
  timing(): { step: number; startStep: number; stopStep: number; dt: number; inRun: boolean; firstStep: number };
  random(lo: number, hi: number, seed: number, mode: Mode): Value;
  normal(mu: number, sigma: number, seed: number, mode: Mode): Value;
  /** Number of owned atoms (length of atom-mode values). */
  natoms(): number;
}

const isVec = (v: Value): v is Float64Array => typeof v !== 'number';

const map1 = (a: Value, f: (x: number) => number): Value => {
  if (!isVec(a)) return f(a);
  const out = new Float64Array(a.length);
  for (let k = 0; k < a.length; k++) out[k] = f(a[k]);
  return out;
};

const map2 = (a: Value, b: Value, f: (x: number, y: number) => number): Value => {
  if (!isVec(a) && !isVec(b)) return f(a, b);
  const n = isVec(a) ? a.length : (b as Float64Array).length;
  if (isVec(a) && isVec(b) && a.length !== b.length) throw new StyleError('vector lengths do not match in a formula');
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) out[k] = f(isVec(a) ? a[k] : a, isVec(b) ? b[k] : b);
  return out;
};

const truth = (b: boolean) => (b ? 1 : 0);

/** C round(): halfway cases away from zero. */
const cround = (x: number) => Math.sign(x) * Math.round(Math.abs(x));

export const evaluate = (src: string, env: FormulaEnv, mode: Mode): Value => evalNode(parseFormula(src), env, mode, src);

export const evaluateScalar = (src: string, env: FormulaEnv): number => {
  const v = evaluate(src, env, 'equal');
  if (isVec(v)) throw new StyleError(`formula "${src}" gives a vector where a single value is needed`);
  return v;
};

const resolveIndex = (ix: Index, env: FormulaEnv): number => {
  if (typeof ix === 'number') return ix;
  const v = env.variable(ix.variable, 'equal');
  if (isVec(v)) throw new StyleError(`index variable ${ix.variable} must give a single value`);
  return Math.trunc(v);
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
      switch (n.op) {
        case '+': return map2(a, b, (x, y) => x + y);
        case '-': return map2(a, b, (x, y) => x - y);
        case '*': return map2(a, b, (x, y) => x * y);
        case '/': return map2(a, b, (x, y) => {
          if (y === 0) throw new StyleError(`divide by 0 in variable formula "${src}"`);
          return x / y;
        });
        case '%': return map2(a, b, (x, y) => {
          if (y === 0) throw new StyleError(`modulo 0 in variable formula "${src}"`);
          return x % y;
        });
        case '^': return map2(a, b, (x, y) => {
          if (y === 0) return 1;
          if (x === 0 && y < 0) throw new StyleError(`invalid power expression in variable formula "${src}"`);
          return Math.pow(x, y);
        });
        case '<': return map2(a, b, (x, y) => truth(x < y));
        case '<=': return map2(a, b, (x, y) => truth(x <= y));
        case '>': return map2(a, b, (x, y) => truth(x > y));
        case '>=': return map2(a, b, (x, y) => truth(x >= y));
        case '==': return map2(a, b, (x, y) => truth(x === y));
        case '!=': return map2(a, b, (x, y) => truth(x !== y));
        case '&&': return map2(a, b, (x, y) => truth(x !== 0 && y !== 0));
        case '||': return map2(a, b, (x, y) => truth(x !== 0 || y !== 0));
        case '|^': return map2(a, b, (x, y) => truth((x !== 0) !== (y !== 0)));
      }
      throw new StyleError(`unknown operator ${n.op}`);
    }
    case 'name': {
      if (n.name in CONSTANTS) return CONSTANTS[n.name];
      if (n.name === 'version') return 20260902;
      if (mode === 'atom' && (ATOM_VECTORS.includes(n.name) || /^[id]2?_/.test(n.name))) return env.atomVector(n.name);
      const th = env.thermo(n.name);
      if (th !== undefined) return th;
      if (ATOM_VECTORS.includes(n.name) || /^[id]2?_/.test(n.name)) throw new StyleError(`atom vector '${n.name}' can only be used in an atom-style variable (use ${n.name}[ID] for one atom)`);
      throw new StyleError(`invalid thermo keyword or name '${n.name}' in variable formula "${src}"`);
    }
    case 'atomval': return env.atomValue(n.name, resolveIndex(n.index, env));
    case 'ref': {
      const i = n.i === null ? null : resolveIndex(n.i, env);
      const j = n.j === null ? null : resolveIndex(n.j, env);
      if (n.kind === 'v') {
        if (j !== null) throw new StyleError(`v_${n.id}[i][j] is not valid`);
        if (i !== null) return env.variableElement(n.id, i);
        return env.variable(n.id, mode);
      }
      return env.reference(n.kind, n.id, i, j, mode);
    }
    case 'raw': return env.raw(n.fn, n.args, mode);
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
      if (x < 0) throw new StyleError(`sqrt of negative value in variable formula "${src}"`);
      return Math.sqrt(x);
    });
    case 'exp': return map1(arg(0), Math.exp);
    case 'ln': return map1(arg(0), (x) => {
      if (x <= 0) throw new StyleError(`log of zero/negative value in variable formula "${src}"`);
      return Math.log(x);
    });
    case 'log': return map1(arg(0), (x) => {
      if (x <= 0) throw new StyleError(`log of zero/negative value in variable formula "${src}"`);
      return Math.log10(x);
    });
    case 'abs': return map1(arg(0), Math.abs);
    case 'sign': return map1(arg(0), (x) => (x >= 0 ? 1 : -1));
    case 'sin': return map1(arg(0), Math.sin);
    case 'cos': return map1(arg(0), Math.cos);
    case 'tan': return map1(arg(0), Math.tan);
    case 'asin': return map1(arg(0), (x) => {
      if (x < -1 || x > 1) throw new StyleError(`arcsin of invalid value in variable formula "${src}"`);
      return Math.asin(x);
    });
    case 'acos': return map1(arg(0), (x) => {
      if (x < -1 || x > 1) throw new StyleError(`arccos of invalid value in variable formula "${src}"`);
      return Math.acos(x);
    });
    case 'atan': return map1(arg(0), Math.atan);
    case 'atan2': return map2(arg(0), arg(1), Math.atan2);
    case 'ceil': return map1(arg(0), Math.ceil);
    case 'floor': return map1(arg(0), Math.floor);
    case 'round': return map1(arg(0), cround);
    case 'ternary': {
      const c = arg(0);
      if (!isVec(c)) return c !== 0 ? arg(1) : arg(2);
      return map2(map2(c, arg(1), (x, y) => (x !== 0 ? y : Number.NaN)), arg(2), (x, z) => (Number.isNaN(x) ? z : x));
    }
    case 'random': { const [lo, hi, seed] = scalarArgs(); return env.random(lo, hi, seed, mode); }
    case 'normal': { const [mu, sg, seed] = scalarArgs(); return env.normal(mu, sg, seed, mode); }
    default: break;
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
    case 'swiggle': return a[0] + a[1] * Math.sin(2 * Math.PI / a[2] * (step - tm.startStep) * tm.dt);
    case 'cwiggle': return a[0] + a[1] * (1 - Math.cos(2 * Math.PI / a[2] * (step - tm.startStep) * tm.dt));
    case 'stagger': {
      const [x, y] = a;
      if (!(x > 0 && y > 0 && x > y)) throw new StyleError('invalid stagger() arguments: need x > y > 0');
      const lower = Math.floor(step / x) * x;
      const delta = step - lower;
      return delta < y ? lower + y : lower + x;
    }
    case 'logfreq': {
      const [x, y, z] = a;
      if (!(x > 0 && y > 0 && z > 0 && y < z)) throw new StyleError('invalid logfreq() arguments');
      if (step < x) return x;
      let lower = x;
      while (step >= z * lower) lower *= z;
      const multiple = Math.floor(step / lower);
      return multiple < y ? (multiple + 1) * lower : lower * z;
    }
    case 'logfreq2': {
      const [x, y, z] = a;
      if (!(x > 0 && y > 0 && z > 0)) throw new StyleError('invalid logfreq2() arguments');
      if (step < x) return x;
      let lower = x;
      while (step >= z * lower) lower *= z;
      const upper = z * lower;
      const delta = (upper - lower) / y;
      return Math.round(lower + delta * (Math.floor((step - lower) / delta) + 1));
    }
    case 'logfreq3': {
      const [x, y, z] = a;
      if (!(x > 0 && z > 0 && y > 1 && z - x >= y - 1)) throw new StyleError('invalid logfreq3() arguments');
      if (step < x) return x;
      const r = Math.pow(z / x, 1 / (y - 1));
      let cur = x;
      let prev = x;
      for (let k = 1; k < y; k++) {
        let next = Math.round(x * Math.pow(r, k));
        if (next <= prev) next = prev + 1;
        prev = next;
        cur = next;
        if (cur > step) return cur;
      }
      return 1e20;
    }
    case 'stride': {
      const [x, y, z] = a;
      if (!(x >= 0 && y >= 0 && z > 0 && x <= y)) throw new StyleError('invalid stride() arguments');
      if (step < x) return x;
      if (step >= y) return 1e20;
      const next = x + (Math.floor((step - x) / z) + 1) * z;
      return next > y ? 1e20 : next;
    }
    case 'stride2': {
      const [x, y, z, aa, b, c] = a;
      if (!(x >= 0 && y >= 0 && z > 0 && x <= y && aa >= 0 && b >= 0 && c > 0 && aa < b && aa >= x && b <= y)) {
        throw new StyleError('invalid stride2() arguments');
      }
      if (step < x) return x;
      if (step >= aa && step < b) return Math.min(b, aa + (Math.floor((step - aa) / c) + 1) * c);
      let next = x + (Math.floor((step - x) / z) + 1) * z;
      if (step < aa && next > aa) next = aa;
      return next > y ? 1e20 : next;
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
