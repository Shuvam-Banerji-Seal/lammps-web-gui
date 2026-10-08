import { StyleError } from '../force/types';
import { parseLepton } from './parse';
import { resolveNames, diff, dependsOn, type RNode } from './resolve';
import { compileNode, type Fn, type ZblConst } from './compile';

/*
 * Public entry of the Lepton expression engine (docs.lammps.org/lepton_expression.html):
 * compileLepton parses an expression, resolves its definitions and built-in
 * variables, and returns the compiled value and exact first derivatives.
 *
 * The env array holds the built-in variables first (in `builtins` order),
 * then one slot per equal-style variable reference (`vrefs`, v_name in the
 * expression), whose current values the caller writes before each evaluation.
 */

export interface LeptonProgram {
  /** Value of the expression. */
  value: Fn;
  /** dvalue/dwrt[k] for each requested variable (same order as opts.wrt). */
  deriv: Fn[];
  /** Equal-style variable names referenced as v_name; env slot = builtins.length + index. */
  vrefs: string[];
  builtins: string[];
}

export interface LeptonOptions {
  /** Built-in variables of the style, e.g. ['r', 'qi', 'qj'] for pair lepton/coul. */
  builtins: string[];
  /** Variables to differentiate with respect to (must be builtins). */
  wrt: string[];
  /** Allow zbl(zi,zj,r) (pair styles only); constants are read at evaluation. */
  zbl?: ZblConst | null;
}

export const compileLepton = (text: string, opts: LeptonOptions): LeptonProgram => {
  const builtins = [...opts.builtins];
  for (const w of opts.wrt) {
    if (!builtins.includes(w)) throw new StyleError(`lepton: cannot differentiate with respect to '${w}'`);
  }
  const parsed = parseLepton(text, { zbl: !!opts.zbl });
  const tree: RNode = resolveNames(parsed, new Set(builtins));
  const vrefs: string[] = [];
  collectVrefs(tree, vrefs);
  const slotOf = (kind: 'var' | 'vref', name: string): number => {
    if (kind === 'var') return builtins.indexOf(name);
    return builtins.length + vrefs.indexOf(name);
  };
  const zbl = opts.zbl ?? null;
  const value = compileNode(tree, slotOf, zbl);
  const deriv = opts.wrt.map((w) => {
    const d: RNode = dependsOn(tree, w) ? diff(tree, w) : { t: 'num', v: 0 };
    return compileNode(d, slotOf, zbl);
  });
  return { value, deriv, vrefs, builtins };
};

const collectVrefs = (n: RNode, out: string[]): void => {
  switch (n.t) {
    case 'vref': if (!out.includes(n.name)) out.push(n.name); return;
    case 'neg': collectVrefs(n.a, out); return;
    case 'bin': collectVrefs(n.a, out); collectVrefs(n.b, out); return;
    case 'call': for (const a of n.args) collectVrefs(a, out); return;
    default: return;
  }
};

/** Fills the equal-variable slots of an env from their current values. */
export const fillVrefs = (env: Float64Array, prog: LeptonProgram, value: (name: string) => number): void => {
  const base = prog.builtins.length;
  for (let k = 0; k < prog.vrefs.length; k++) env[base + k] = value(prog.vrefs[k]);
};

export { type ZblConst } from './compile';
