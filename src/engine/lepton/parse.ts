import { StyleError } from '../force/types';

/*
 * Lepton expression parser (docs.lammps.org/lepton_expression.html).
 *
 * Operators "+ - * / ^" and functions as listed on that page. Measured with
 * native LAMMPS (black box, pair style lepton): unary minus binds looser than
 * "^" ("-2^2" is -(2^2)); "^" is right-associative ("2^3^2" is 2^(3^2));
 * "+" is binary only (a leading unary plus is a parse error); "-" may follow
 * another operator ("2*-r"; "2^-1" is 0.5); whitespace is removed before
 * parsing ("2 3*r" is 23*r). Those rules are the grammar below.
 *
 * Quoted text: "Whitespace and quotation characters ('\'' and '"') are
 * ignored." (lepton_expression.rst). Definitions: "An expression may be
 * followed by definitions for intermediate values that appear in the
 * expression. A semicolon ";" is used as a delimiter between value
 * definitions." Names are resolved in resolve.ts.
 *
 * Numbers: "Numbers may be given in either decimal or exponential form.  All of
 * the following are valid numbers: `5`, `-3.1`, `1e6`, and `3.12e-2`."
 * Variables: "v_name" is replaced by the value of the LAMMPS equal-style
 * variable "name" before evaluation.
 */

export type Node =
  | { t: 'num'; v: number }
  | { t: 'name'; name: string }
  | { t: 'vref'; name: string }
  | { t: 'neg'; a: Node }
  | { t: 'bin'; op: '+' | '-' | '*' | '/' | '^'; a: Node; b: Node }
  | { t: 'call'; fn: string; args: Node[] };

/** Functions of lepton_expression.rst with their argument counts. */
export const FUNCTION_ARITY: Record<string, number> = {
  sqrt: 1, exp: 1, log: 1, sin: 1, cos: 1, sec: 1, csc: 1, tan: 1, cot: 1,
  asin: 1, acos: 1, atan: 1, sinh: 1, cosh: 1, tanh: 1, erf: 1, erfc: 1,
  abs: 1, min: 2, max: 2, delta: 1, step: 1,
};
/** Custom function of pair_style lepton (pair_lepton.rst): zbl(zi,zj,r). */
export const ZBL_ARITY = 3;

type Tok = { k: 'num'; v: number } | { k: 'id'; s: string } | { k: 'op'; s: string };

const NUM = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/;
const ID = /^[A-Za-z_][A-Za-z0-9_]*/;

const tokenize = (text: string, what: string): Tok[] => {
  const src = text.replace(/[\s'"]/g, '');
  const out: Tok[] = [];
  let p = 0;
  while (p < src.length) {
    const rest = src.slice(p);
    const c = src[p];
    let m: RegExpExecArray | null;
    if ((m = NUM.exec(rest))) {
      out.push({ k: 'num', v: Number(m[0]) });
      p += m[0].length;
    } else if ((m = ID.exec(rest))) {
      out.push({ k: 'id', s: m[0] });
      p += m[0].length;
    } else if ('+-*/^(),=;'.includes(c)) {
      out.push({ k: 'op', s: c });
      p++;
    } else {
      throw new StyleError(`lepton expression "${what}": unexpected character '${c}'`);
    }
  }
  return out;
};

class Parser {
  private p = 0;
  constructor(private readonly toks: Tok[], private readonly what: string, private readonly zbl: boolean) {}

  private fail(msg: string): never {
    throw new StyleError(`lepton expression "${this.what}": ${msg}`);
  }

  private peekOp(s: string): boolean {
    const t = this.toks[this.p];
    return t !== undefined && t.k === 'op' && t.s === s;
  }

  parseAll(): Node {
    const n = this.sum();
    if (this.p < this.toks.length) this.fail(`unexpected text at '${this.describe(this.toks[this.p])}'`);
    return n;
  }

  private describe(t: Tok): string {
    return t.k === 'num' ? String(t.v) : t.s;
  }

  private sum(): Node {
    let a = this.prod();
    while (this.peekOp('+') || this.peekOp('-')) {
      const op = (this.toks[this.p++] as { s: string }).s as '+' | '-';
      a = { t: 'bin', op, a, b: this.prod() };
    }
    return a;
  }

  private prod(): Node {
    let a = this.unary();
    while (this.peekOp('*') || this.peekOp('/')) {
      const op = (this.toks[this.p++] as { s: string }).s as '*' | '/';
      a = { t: 'bin', op, a, b: this.unary() };
    }
    return a;
  }

  private unary(): Node {
    if (this.peekOp('-')) {
      this.p++;
      return { t: 'neg', a: this.unary() };
    }
    return this.power();
  }

  private power(): Node {
    const base = this.primary();
    if (this.peekOp('^')) {
      this.p++;
      // right-associative; the exponent may carry a unary minus ("2^-1")
      return { t: 'bin', op: '^', a: base, b: this.unary() };
    }
    return base;
  }

  private primary(): Node {
    const t = this.toks[this.p];
    if (t === undefined) this.fail('unexpected end of expression');
    if (t.k === 'num') {
      this.p++;
      return { t: 'num', v: t.v };
    }
    if (t.k === 'id') {
      this.p++;
      if (this.peekOp('(')) return this.call(t.s);
      if (t.s.startsWith('v_')) {
        if (t.s.length === 2) this.fail(`variable reference '${t.s}' has no name`);
        return { t: 'vref', name: t.s.slice(2) };
      }
      return { t: 'name', name: t.s };
    }
    if (t.s === '(') {
      this.p++;
      const inner = this.sum();
      if (!this.peekOp(')')) this.fail('unbalanced parentheses');
      this.p++;
      return inner;
    }
    this.fail(`unexpected '${t.s}'`);
  }

  private call(fn: string): Node {
    this.p++; // (
    const args: Node[] = [this.sum()];
    while (this.peekOp(',')) {
      this.p++;
      args.push(this.sum());
    }
    if (!this.peekOp(')')) this.fail(`unbalanced parentheses in call to ${fn}`);
    this.p++;
    const want = fn === 'zbl' && this.zbl ? ZBL_ARITY : FUNCTION_ARITY[fn];
    if (fn === 'zbl' && !this.zbl) this.fail('zbl(zi,zj,r) is only defined for pair styles lepton, lepton/coul and lepton/sphere');
    if (want === undefined) this.fail(`unknown function ${fn}`);
    if (args.length !== want) this.fail(`wrong number of arguments to function ${fn}: ${args.length} given, ${want} needed`);
    return { t: 'call', fn, args };
  }
}

/** A parsed expression: the main value and the ordered definitions "name = expr". */
export interface Parsed {
  main: Node;
  /** Definition k (1-based segment index k+1 in the string). */
  defs: { name: string; expr: Node }[];
}

const RESERVED = /^v_/;

/**
 * Parses a Lepton expression string. Empty definition segments are skipped;
 * an empty main expression, a malformed definition, a repeated definition
 * name or a definition named like a variable reference is a StyleError.
 */
export const parseLepton = (text: string, opts: { zbl: boolean }): Parsed => {
  const toks = tokenize(text, text);
  const segs: Tok[][] = [[]];
  for (const t of toks) {
    if (t.k === 'op' && t.s === ';') segs.push([]);
    else segs[segs.length - 1].push(t);
  }
  if (segs[0].length === 0) throw new StyleError(`lepton expression "${text}": empty expression`);
  const main = new Parser(segs[0], text, opts.zbl).parseAll();
  const defs: Parsed['defs'] = [];
  const seen = new Set<string>();
  for (const seg of segs.slice(1)) {
    if (seg.length === 0) continue;
    const [head, eq] = seg;
    if (!head || head.k !== 'id' || !eq || eq.k !== 'op' || eq.s !== '=') {
      throw new StyleError(`lepton expression "${text}": a value definition must have the form name=expression`);
    }
    if (RESERVED.test(head.s)) throw new StyleError(`lepton expression "${text}": '${head.s}' cannot be defined (v_ names are variable references)`);
    if (seen.has(head.s)) throw new StyleError(`lepton expression "${text}": value '${head.s}' is defined more than once`);
    seen.add(head.s);
    const expr = new Parser(seg.slice(2), text, opts.zbl);
    if (seg.length === 2) throw new StyleError(`lepton expression "${text}": value '${head.s}' has no expression`);
    defs.push({ name: head.s, expr: expr.parseAll() });
  }
  return { main, defs };
};
