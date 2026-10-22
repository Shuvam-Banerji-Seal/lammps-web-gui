import type { System } from './system';
import { StyleError } from './force/types';
import { bounds } from './force/util';

/*
 * References to computes, fixes and variables in command arguments, read as native LAMMPS reads them: the ArgInfo
 * class (src/arg_info.cpp at b332f28) and utils::expand_args (src/utils.cpp at b332f28), written in the engine's own
 * terms.
 */

/** The variable styles native's Variable::equalstyle accepts (without Python). */
export const EQUAL_STYLES: ReadonlySet<string> = new Set(['equal', 'timer', 'internal']);

/** The kind of a reference: c_ (compute), f_ (fix), v_ (variable), d_/i_ (custom per-atom), or none of them. */
export type ArgWhich = 'c' | 'f' | 'v' | 'd' | 'i' | 'none' | 'unknown';

export interface ArgRef {
  which: ArgWhich;
  /** The ID or name (the whole word for 'none'). */
  name: string;
  /** The number of bracketed indices (0, 1 or 2). */
  dim: number;
  /** The first and second index (0 without brackets). */
  index1: number;
  index2: number;
}

/** std::stoi: leading blanks and a sign, then at least one digit; what follows the digits is ignored. */
const stoi = (t: string): number | null => {
  const m = /^\s*([+-]?\d+)/.exec(t);
  return m ? Number(m[1]) : null;
};

/**
 * One argument as native's ArgInfo reads it, with the reference kinds in `allowed` (default: computes, fixes and
 * variables): 'none' for a word that is no reference of an allowed kind, 'unknown' for brackets that do not hold a
 * number.
 */
export const argInfo = (arg: string, allowed = 'cfv'): ArgRef => {
  const r: ArgRef = { which: 'none', name: arg, dim: 0, index1: -1, index2: -1 };
  const two = arg.length > 3 && arg[1] === '2' && arg[2] === '_';
  if (!two && !(arg.length > 2 && arg[1] === '_')) {
    r.index1 = 0;
    return r;
  }
  if (!'cfvdi'.includes(arg[0]) || !allowed.includes(arg[0])) {
    r.index1 = 0;
    return r;
  }
  r.which = arg[0] as ArgWhich;
  const offset = arg[1] === '_' ? 2 : 3;
  const h1 = arg.indexOf('[', offset);
  if (h1 < 0) {
    r.index1 = 0;
    r.name = arg.slice(offset);
    return r;
  }
  r.name = arg.slice(offset, h1);
  r.dim = 1;
  let h2 = arg.indexOf('[', h1 + 1);
  if (h2 >= 0) {
    r.dim = 2;
    if (arg[arg.length - 1] !== ']') r.which = 'unknown';
    else {
      const v = stoi(arg.slice(h2 + 1, arg.length - 1));
      if (v === null) r.which = 'unknown';
      else r.index2 = v;
    }
  } else h2 = arg.length;
  if (arg[h2 - 1] !== ']' || (r.dim === 1 && arg.indexOf(']') !== h2 - 1)) {
    r.which = 'unknown';
  } else {
    const v = stoi(arg.slice(h1 + 1, arg.length - 1));
    if (v === null) r.which = 'unknown';
    else r.index1 = v;
  }
  return r;
};

/** The largest vector-variable range (native MAXSMALLINT). */
const MAX_SMALL_INT = 2147483647;

/**
 * Expands c_ID[*], f_ID[*] and v_name[*] words (with an optional i and j around the *), as native's utils::expand_args
 * does: mode 0 expands a compute or fix global vector and a vector-style variable (which needs an upper bound),
 * mode 1 a global array's columns; either mode expands per-atom and then local array columns. A word that does not
 * expand is kept. Returns the words and, for each, the index of the word it came from.
 */
export const expandArgs = (sys: System, words: string[], mode: 0 | 1): { words: string[]; from: number[] } => {
  const out: string[] = [];
  const from: number[] = [];
  words.forEach((word, iarg) => {
    const m = /^([cfv])_(\w+)\[(\d*\*\d*)\](.*)$/.exec(word);
    let nmax = -1;
    if (m) {
      const [, kind, id, wc] = m;
      if (kind === 'c' || kind === 'f') {
        const obj = kind === 'c' ? sys.computes.find((c) => c.id === id) : sys.fixes.find((f) => f.id === id);
        if (obj) {
          let nvec = obj.sizeVector;
          // an engine compute that sizes its vector when it is first computed reports 0 until then
          if (kind === 'c' && mode === 0 && obj.vectorFlag && nvec === 0) nvec = sys.compute(id).vectorValues().length;
          if (mode === 0 && obj.vectorFlag) nmax = nvec;
          else if (mode === 1 && obj.arrayFlag) nmax = obj.sizeArrayCols;
          else if (obj.peratomFlag && obj.sizePeratomCols) nmax = obj.sizePeratomCols;
          else if (obj.localFlag && obj.sizeLocalCols) nmax = obj.sizeLocalCols;
        }
      } else if (mode === 0 && sys.vars.get(id)?.style === 'vector') {
        const [, hi] = bounds(wc, 1, MAX_SMALL_INT);
        if (hi >= MAX_SMALL_INT) throw new StyleError(`Upper bound required to expand vector style variable ${id}`);
        nmax = hi;
      }
      if (nmax >= 0) {
        const [lo, hi] = bounds(wc, 1, nmax);
        for (let k = lo; k <= hi; k++) {
          out.push(`${kind}_${id}[${k}]${m[4]}`);
          from.push(iarg);
        }
        return;
      }
    }
    out.push(word);
    from.push(iarg);
  });
  return { words: out, from };
};
