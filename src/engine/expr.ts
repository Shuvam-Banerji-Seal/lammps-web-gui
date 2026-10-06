import { Rng } from './rng';

/*
 * Equal-style variable formulas (docs.lammps.org/variable.html), evaluated
 * by a small recursive-descent parser — never `eval`.
 *
 *   numbers "0.0, 100, -5.4, 2.8e-4"; constants "PI, version, on, off,
 *   true, false, yes, no"; thermo keywords "from thermo_style"; variable
 *   references "v_name".
 *   "Operators are evaluated left to right and have the usual C-style
 *   precedence: unary minus and unary logical NOT operator “!” have the
 *   highest precedence, exponentiation “^” is next; multiplication and
 *   division and the modulo operator “%” are next; addition and subtraction
 *   are next; the 4 relational operators “<”, “<=”, “>”, and “>=” are next;
 *   the two remaining relational operators “==” and “!=” are next; then the
 *   logical AND operator “&&”; and finally the logical OR operator “||” and
 *   logical XOR (exclusive or) operator “|^” have the lowest precedence."
 *   "the formula “-2^2” will evaluate to 4, not -4."
 *   "The ln() is the natural log; log() is the base 10 log."
 *   "The sign(x) function returns 1.0 if the value is greater than or equal
 *   to 0.0, and -1.0 otherwise."
 *   random(lo,hi,seed) / normal(mu,sigma,seed): "the seed is used the first
 *   time the internal random number generator is invoked, to initialize it".
 *   ternary(x,y,z) returns y if x is non-zero, else z.
 */

export interface ExprContext {
  /** Value of a thermo keyword, or undefined if the name is not one. */
  thermo(name: string): number | undefined;
  /** Numeric value of v_name. */
  variable(name: string): number;
  /** Shared generator for random()/normal(), created on first use. */
  rng(seed: number): Rng;
}

type Tok = { k: 'num'; v: number } | { k: 'id'; v: string } | { k: 'op'; v: string };

const OPS = ['&&', '||', '|^', '==', '!=', '<=', '>=', '<', '>', '+', '-', '*', '/', '%', '^', '!', '(', ')', ','];

const lex = (src: string): Tok[] => {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (/\s/.test(c)) { i++; continue; }
    const num = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(src.slice(i));
    if (num) { out.push({ k: 'num', v: Number(num[0]) }); i += num[0].length; continue; }
    const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
    if (id) { out.push({ k: 'id', v: id[0] }); i += id[0].length; continue; }
    const op = OPS.find((o) => src.startsWith(o, i));
    if (op) { out.push({ k: 'op', v: op }); i += op.length; continue; }
    throw new Error(`unexpected character '${c}' in formula '${src}'`);
  }
  return out;
};

const CONSTANTS: Record<string, number> = {
  PI: Math.PI, on: 1, true: 1, yes: 1, off: 0, false: 0, no: 0,
};

const FUNCS: Record<string, { n: number; f?: (...a: number[]) => number }> = {
  sqrt: { n: 1, f: Math.sqrt }, exp: { n: 1, f: Math.exp }, ln: { n: 1, f: Math.log },
  log: { n: 1, f: Math.log10 }, abs: { n: 1, f: Math.abs },
  sign: { n: 1, f: (x) => (x >= 0 ? 1 : -1) },
  sin: { n: 1, f: Math.sin }, cos: { n: 1, f: Math.cos }, tan: { n: 1, f: Math.tan },
  asin: { n: 1, f: Math.asin }, acos: { n: 1, f: Math.acos }, atan: { n: 1, f: Math.atan },
  atan2: { n: 2, f: Math.atan2 },
  ceil: { n: 1, f: Math.ceil }, floor: { n: 1, f: Math.floor },
  // C round(): halfway cases away from zero
  round: { n: 1, f: (x) => Math.sign(x) * Math.round(Math.abs(x)) },
  ternary: { n: 3 }, random: { n: 3 }, normal: { n: 3 },
};

export const evaluateFormula = (src: string, ctx: ExprContext): number => {
  const toks = lex(src);
  let p = 0;
  const peek = () => toks[p];
  const isOp = (v: string) => toks[p]?.k === 'op' && toks[p].v === v;
  const expect = (v: string) => {
    if (!isOp(v)) throw new Error(`expected '${v}' in formula '${src}'`);
    p++;
  };
  const bool = (b: boolean) => (b ? 1 : 0);

  const levels: { ops: string[]; apply: (op: string, a: number, b: number) => number }[] = [
    { ops: ['||', '|^'], apply: (o, a, b) => (o === '||' ? bool(a !== 0 || b !== 0) : bool((a !== 0) !== (b !== 0))) },
    { ops: ['&&'], apply: (_o, a, b) => bool(a !== 0 && b !== 0) },
    { ops: ['==', '!='], apply: (o, a, b) => bool(o === '==' ? a === b : a !== b) },
    { ops: ['<', '<=', '>', '>='], apply: (o, a, b) => bool(o === '<' ? a < b : o === '<=' ? a <= b : o === '>' ? a > b : a >= b) },
    { ops: ['+', '-'], apply: (o, a, b) => (o === '+' ? a + b : a - b) },
    { ops: ['*', '/', '%'], apply: (o, a, b) => {
      if (o === '*') return a * b;
      if (b === 0) throw new Error(`division by zero in formula '${src}'`);
      return o === '/' ? a / b : a % b;
    } },
    { ops: ['^'], apply: (_o, a, b) => Math.pow(a, b) },
  ];

  const binary = (level: number): number => {
    if (level === levels.length) return unary();
    let a = binary(level + 1);
    for (;;) {
      const t = peek();
      if (!t || t.k !== 'op' || !levels[level].ops.includes(t.v)) return a;
      p++;
      const b = binary(level + 1);
      a = levels[level].apply(t.v, a, b);
    }
  };

  const unary = (): number => {
    if (isOp('-')) { p++; return -unary(); }
    if (isOp('!')) { p++; return unary() === 0 ? 1 : 0; }
    return primary();
  };

  const args = (): number[] => {
    expect('(');
    const out: number[] = [];
    if (!isOp(')')) {
      out.push(binary(0));
      while (isOp(',')) { p++; out.push(binary(0)); }
    }
    expect(')');
    return out;
  };

  const primary = (): number => {
    const t = peek();
    if (!t) throw new Error(`formula '${src}' ends unexpectedly`);
    if (t.k === 'num') { p++; return t.v; }
    if (t.k === 'op' && t.v === '(') { p++; const v = binary(0); expect(')'); return v; }
    if (t.k === 'id') {
      p++;
      const fn = FUNCS[t.v];
      if (fn && isOp('(')) {
        const a = args();
        if (a.length !== fn.n) throw new Error(`${t.v}() takes ${fn.n} argument(s), got ${a.length}`);
        if (t.v === 'ternary') return a[0] !== 0 ? a[1] : a[2];
        if (t.v === 'random') return a[0] + (a[1] - a[0]) * ctx.rng(a[2]).uniform();
        if (t.v === 'normal') return a[0] + a[1] * ctx.rng(a[2]).gaussian();
        return fn.f!(...a);
      }
      if (t.v.startsWith('v_')) return ctx.variable(t.v.slice(2));
      if (t.v in CONSTANTS) return CONSTANTS[t.v];
      const th = ctx.thermo(t.v);
      if (th !== undefined) return th;
      throw new Error(`unknown name '${t.v}' in formula '${src}' (supported: numbers, PI, v_name, thermo keywords, math functions)`);
    }
    throw new Error(`unexpected '${t.v}' in formula '${src}'`);
  };

  const v = binary(0);
  if (p !== toks.length) throw new Error(`unexpected '${(toks[p] as Tok).v}' in formula '${src}'`);
  return v;
};
