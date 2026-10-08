import { StyleError } from '../force/types';
import type { Node, Parsed } from './parse';

/*
 * Name resolution and exact symbolic differentiation of Lepton expressions.
 *
 * Name rule, measured with native LAMMPS (black box, pair style lepton):
 * a name used in segment s resolves to the first definition at a later
 * segment (so "r^2; r=2" gives 4 and "a^2; a=b; b=r" gives r^4 with r = 1.3);
 * a name with no later definition is a built-in variable of the style (r,
 * qi, theta, ...); a use after its definition does not see it (lepton_expression.rst:
 * "All uses of a value must appear *before* that value's definition.").
 * Here an undefined name is a StyleError instead of the value 0 that native
 * LAMMPS substitutes.
 *
 * Derivatives are exact (chain rule on the resolved tree); the styles need
 * dE/dr, dE/dtheta, dE/dphi and the gradient of V(x, y, z).
 */

/** Resolved tree: builtin variables become 'var', equal-style references stay 'vref'. */
export type RNode =
  | { t: 'num'; v: number }
  | { t: 'var'; name: string }
  | { t: 'vref'; name: string }
  | { t: 'neg'; a: RNode }
  | { t: 'bin'; op: '+' | '-' | '*' | '/' | '^'; a: RNode; b: RNode }
  | { t: 'call'; fn: string; args: RNode[] };

export const num = (v: number): RNode => ({ t: 'num', v });
const isNum = (n: RNode, v?: number): boolean => n.t === 'num' && (v === undefined || n.v === v);
const val = (n: RNode): number => (n.t === 'num' ? n.v : Number.NaN);

export const add = (a: RNode, b: RNode): RNode => {
  if (isNum(a) && isNum(b)) return num(val(a) + val(b));
  if (isNum(a, 0)) return b;
  if (isNum(b, 0)) return a;
  return { t: 'bin', op: '+', a, b };
};
export const sub = (a: RNode, b: RNode): RNode => {
  if (isNum(a) && isNum(b)) return num(val(a) - val(b));
  if (isNum(b, 0)) return a;
  if (isNum(a, 0)) return neg(b);
  return { t: 'bin', op: '-', a, b };
};
export const mul = (a: RNode, b: RNode): RNode => {
  if (isNum(a) && isNum(b)) return num(val(a) * val(b));
  if (isNum(a, 0) || isNum(b, 0)) return num(0);
  if (isNum(a, 1)) return b;
  if (isNum(b, 1)) return a;
  return { t: 'bin', op: '*', a, b };
};
export const div = (a: RNode, b: RNode): RNode => {
  if (isNum(a) && isNum(b)) return num(val(a) / val(b));
  if (isNum(a, 0)) return num(0);
  if (isNum(b, 1)) return a;
  return { t: 'bin', op: '/', a, b };
};
export const pow = (a: RNode, b: RNode): RNode => {
  if (isNum(b, 1)) return a;
  if (isNum(a) && isNum(b)) return num(val(a) ** val(b));
  return { t: 'bin', op: '^', a, b };
};
export const neg = (a: RNode): RNode => (isNum(a) ? num(-val(a)) : { t: 'neg', a });
export const call = (fn: string, ...args: RNode[]): RNode => ({ t: 'call', fn, args });

/** Resolves all names of a parsed expression against the style's built-in variables. */
export const resolveNames = (p: Parsed, builtins: ReadonlySet<string>): RNode => {
  const memo = new Map<number, RNode>();
  // definition k (0-based in p.defs) is in segment k+1; a use in segment s sees definitions k with k+1 > s
  const defAt = (k: number): RNode => {
    const hit = memo.get(k);
    if (hit) return hit;
    const r = walk(p.defs[k].expr, k + 1);
    memo.set(k, r);
    return r;
  };
  const walk = (n: Node, seg: number): RNode => {
    switch (n.t) {
      case 'num': return num(n.v);
      case 'vref': return { t: 'vref', name: n.name };
      case 'neg': return neg(walk(n.a, seg));
      case 'bin': return { t: 'bin', op: n.op, a: walk(n.a, seg), b: walk(n.b, seg) };
      case 'call': return { t: 'call', fn: n.fn, args: n.args.map((x) => walk(x, seg)) };
      case 'name': {
        for (let k = seg; k < p.defs.length; k++) {
          if (p.defs[k].name === n.name) return defAt(k);
        }
        if (builtins.has(n.name)) return { t: 'var', name: n.name };
        throw new StyleError(`lepton expression: name '${n.name}' is neither a built-in variable (${[...builtins].join(', ')}) nor defined`);
      }
    }
  };
  return walk(p.main, 0);
};

/** True if the resolved tree depends on builtin variable `wrt`. */
export const dependsOn = (n: RNode, wrt: string): boolean => {
  switch (n.t) {
    case 'num': case 'vref': return false;
    case 'var': return n.name === wrt;
    case 'neg': return dependsOn(n.a, wrt);
    case 'bin': return dependsOn(n.a, wrt) || dependsOn(n.b, wrt);
    case 'call': return n.args.some((x) => dependsOn(x, wrt));
  }
};

/**
 * d n / d wrt. Functions with a kink use the one-sided rules of the
 * selected branch ("min": a <= b selects a'; "max": a >= b selects a').
 * delta has zero derivative; step(u)' = delta(u) u'.
 * zbl(zi,zj,r): the atomic numbers must not depend on wrt; the derivative
 * is the internal function __zbl_d (see compile.ts).
 */
export const diff = (n: RNode, wrt: string): RNode => {
  if (!dependsOn(n, wrt)) return num(0);
  switch (n.t) {
    case 'num': case 'vref': return num(0);
    case 'var': return num(n.name === wrt ? 1 : 0);
    case 'neg': return neg(diff(n.a, wrt));
    case 'bin': {
      const { a, b } = n;
      const da = diff(a, wrt), db = diff(b, wrt);
      switch (n.op) {
        case '+': return add(da, db);
        case '-': return sub(da, db);
        case '*': return add(mul(da, b), mul(a, db));
        case '/': return dependsOn(b, wrt) ? div(sub(mul(da, b), mul(a, db)), pow(b, num(2))) : div(da, b);
        case '^': {
          if (!dependsOn(b, wrt)) return mul(mul(b, pow(a, sub(b, num(1)))), da);
          return mul(pow(a, b), add(mul(db, call('log', a)), div(mul(b, da), a)));
        }
      }
      break;
    }
    case 'call': return callDiff(n.fn, n.args, wrt);
  }
  throw new StyleError('lepton: internal error in differentiation');
};

const callDiff = (fn: string, args: RNode[], wrt: string): RNode => {
  const [u] = args;
  const du = (x: RNode) => diff(x, wrt);
  switch (fn) {
    case 'sqrt': return div(du(u), mul(num(2), call('sqrt', u)));
    case 'exp': return mul(call('exp', u), du(u));
    case 'log': return div(du(u), u);
    case 'sin': return mul(call('cos', u), du(u));
    case 'cos': return neg(mul(call('sin', u), du(u)));
    case 'tan': return div(du(u), pow(call('cos', u), num(2)));
    case 'sec': return mul(mul(call('sec', u), call('tan', u)), du(u));
    case 'csc': return neg(mul(mul(call('csc', u), call('cot', u)), du(u)));
    case 'cot': return neg(div(du(u), pow(call('sin', u), num(2))));
    case 'asin': return div(du(u), call('sqrt', sub(num(1), pow(u, num(2)))));
    case 'acos': return neg(div(du(u), call('sqrt', sub(num(1), pow(u, num(2))))));
    case 'atan': return div(du(u), add(num(1), pow(u, num(2))));
    case 'sinh': return mul(call('cosh', u), du(u));
    case 'cosh': return mul(call('sinh', u), du(u));
    case 'tanh': return div(du(u), pow(call('cosh', u), num(2)));
    case 'erf': return mul(mul(num(2 / Math.sqrt(Math.PI)), call('exp', neg(pow(u, num(2))))), du(u));
    case 'erfc': return neg(mul(mul(num(2 / Math.sqrt(Math.PI)), call('exp', neg(pow(u, num(2))))), du(u)));
    case 'abs': return mul(call('__sgn', u), du(u));
    case 'min': return call('__sel_le', args[0], args[1], du(args[0]), du(args[1]));
    case 'max': return call('__sel_ge', args[0], args[1], du(args[0]), du(args[1]));
    case 'delta': return num(0);
    case 'step': return mul(call('delta', u), du(u));
    case 'zbl': {
      if (dependsOn(args[0], wrt) || dependsOn(args[1], wrt)) {
        throw new StyleError('lepton: zbl atomic numbers (first two arguments) must not depend on the differentiation variable');
      }
      return mul(call('__zbl_d', args[0], args[1], args[2]), du(args[2]));
    }
  }
  throw new StyleError(`lepton: no derivative rule for ${fn}`);
};
