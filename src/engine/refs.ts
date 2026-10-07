import type { System } from './system';
import { StyleError } from './force/types';

/*
 * References to computed quantities by name, for the fixes and computes that
 * take them as inputs (fix ave/time, fix ave/atom, fix ave/histo, fix vector,
 * compute reduce, ...).
 *
 * docs.lammps.org/fix_ave_time.html: "value = c_ID, c_ID[N], f_ID, f_ID[N],
 * v_name"; "c_ID = global scalar or vector calculated by a compute with ID";
 * "c_ID[I] = Ith component of global vector or Ith column of global array
 * calculated by a compute with ID, I can include wildcard (see below)";
 * "v_name = value(s) calculated by an equal-style or vector-style variable
 * with name"; "v_name[I] = value calculated by a vector-style variable with
 * name".
 * docs.lammps.org/compute_reduce.html: "x,y,z,vx,vy,vz,fx,fy,fz = atom
 * attribute (position, velocity, force component)"; "c_ID = per-atom or
 * local vector calculated by a compute with ID"; "c_ID[I] = Ith column of
 * per-atom or local array calculated by a compute with ID"; "v_name =
 * per-atom vector calculated by an atom-style variable with name".
 *
 * Values are raw: thermo's normalization of extensive quantities
 * (thermo_modify norm) is not applied here.
 */

export interface Ref {
  kind: 'c' | 'f' | 'v' | 'attr';
  /** Compute / fix ID, variable name, or attribute name for 'attr'. */
  id: string;
  /** 1-based index from [I], or null. */
  index: number | null;
  /** The original word, for messages. */
  text: string;
}

const ATTRS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz'];

/** Parses c_ID, c_ID[I], f_ID, f_ID[I], v_name, v_name[I], or an atom attribute (x ... fz). */
export const parseRef = (w: string, allowAttrs = false): Ref => {
  if (allowAttrs && ATTRS.includes(w)) return { kind: 'attr', id: w, index: null, text: w };
  const m = /^([cfv])_([A-Za-z0-9_\-/]+?)(?:\[(\d+)\])?$/.exec(w);
  if (!m) {
    const what = allowAttrs ? 'x, y, z, vx, vy, vz, fx, fy, fz, c_ID, c_ID[I], f_ID, f_ID[I] or v_name' : 'c_ID, c_ID[I], f_ID, f_ID[I], v_name or v_name[I]';
    throw new StyleError(`invalid input value '${w}': expected ${what}`);
  }
  const index = m[3] === undefined ? null : Number(m[3]);
  if (index !== null && index < 1) throw new StyleError(`invalid index in '${w}': indices start at 1`);
  return { kind: m[1] as Ref['kind'], id: m[2], index, text: w };
};

/** A global scalar: c_ID, c_ID[I] (vector element), f_ID, f_ID[I], v_name (equal-style), v_name[I]. */
export const globalScalar = (sys: System, r: Ref): number => {
  if (r.kind === 'v') {
    if (r.index === null) return sys.equalVariable(r.id);
    const v = sys.vars.evalVector(r.id, sys.formulaEnv);
    if (r.index > v.length) throw new StyleError(`${r.text}: index out of range 1..${v.length}`);
    return v[r.index - 1];
  }
  if (r.kind === 'c') {
    const c = sys.compute(r.id);
    if (r.index === null) return c.scalarValue();
    const v = c.vectorValues();
    if (r.index > v.length) throw new StyleError(`${r.text}: index out of range 1..${v.length}`);
    return v[r.index - 1];
  }
  if (r.kind === 'f') {
    const f = sys.fix(r.id);
    if (r.index === null) {
      if (!f.scalarFlag) throw new StyleError(`${r.text}: fix ${r.id} does not calculate a global scalar`);
      return f.computeScalar();
    }
    if (!f.vectorFlag) throw new StyleError(`${r.text}: fix ${r.id} does not calculate a global vector`);
    if (r.index > f.sizeVector) throw new StyleError(`${r.text}: index out of range 1..${f.sizeVector}`);
    return f.computeVector(r.index - 1);
  }
  throw new StyleError(`${r.text} is a per-atom attribute, not a global value`);
};

/** A global vector: c_ID (vector), c_ID[I] (array column I), f_ID, f_ID[I], v_name (vector-style). */
export const globalVector = (sys: System, r: Ref): Float64Array => {
  if (r.kind === 'v') {
    if (r.index !== null) throw new StyleError(`${r.text}: a vector-style variable element is a scalar, not a vector`);
    return sys.vars.evalVector(r.id, sys.formulaEnv);
  }
  if (r.kind === 'c') {
    const c = sys.compute(r.id);
    if (r.index === null) return c.vectorValues().slice();
    if (!c.arrayFlag) throw new StyleError(`${r.text}: compute ${r.id} does not calculate a global array`);
    const a = c.arrayValues();
    const rows = c.sizeArrayRows, cols = c.sizeArrayCols;
    if (r.index > cols) throw new StyleError(`${r.text}: column out of range 1..${cols}`);
    const out = new Float64Array(rows);
    for (let i = 0; i < rows; i++) out[i] = a[i * cols + r.index - 1];
    return out;
  }
  if (r.kind === 'f') {
    const f = sys.fix(r.id);
    if (r.index === null) {
      if (!f.vectorFlag) throw new StyleError(`${r.text}: fix ${r.id} does not calculate a global vector`);
      const out = new Float64Array(f.sizeVector);
      for (let i = 0; i < f.sizeVector; i++) out[i] = f.computeVector(i);
      return out;
    }
    if (!f.arrayFlag) throw new StyleError(`${r.text}: fix ${r.id} does not calculate a global array`);
    if (r.index > f.sizeArrayCols) throw new StyleError(`${r.text}: column out of range 1..${f.sizeArrayCols}`);
    const out = new Float64Array(f.sizeArrayRows);
    for (let i = 0; i < f.sizeArrayRows; i++) out[i] = f.computeArray(i, r.index - 1);
    return out;
  }
  throw new StyleError(`${r.text} is a per-atom attribute, not a global vector`);
};

/** Per-atom values over the owned atoms (length state.n). */
export const peratomValues = (sys: System, r: Ref): Float64Array => {
  const s = sys.state;
  const out = new Float64Array(s.n);
  if (r.kind === 'attr') {
    const d = 'xyz'.indexOf(r.id[r.id.length - 1]);
    // forces are the last ones computed, as dump custom reports them
    const src = r.id.length === 1 ? s.x : r.id[0] === 'v' ? s.v : s.f;
    for (let i = 0; i < s.n; i++) out[i] = src[3 * i + d];
    return out;
  }
  if (r.kind === 'v') {
    if (r.index !== null) throw new StyleError(`${r.text}: an atom-style variable has no columns`);
    out.set(sys.atomVariable(r.id).subarray(0, s.n));
    return out;
  }
  const obj = r.kind === 'c' ? sys.compute(r.id) : sys.fix(r.id);
  if (!obj.peratomFlag) throw new StyleError(`${r.text}: ${r.kind === 'c' ? 'compute' : 'fix'} ${r.id} does not calculate per-atom values`);
  const cols = obj.sizePeratomCols;
  const vals = r.kind === 'c' ? sys.compute(r.id).peratomValues() : (cols ? sys.fix(r.id).arrayAtom : sys.fix(r.id).vectorAtom);
  if (r.index === null) {
    if (cols !== 0) throw new StyleError(`${r.text}: it calculates a per-atom array; give a column, e.g. ${r.text}[1]`);
    out.set(vals.subarray(0, s.n));
    return out;
  }
  if (cols === 0) throw new StyleError(`${r.text}: it calculates a per-atom vector, which has no columns`);
  if (r.index > cols) throw new StyleError(`${r.text}: column out of range 1..${cols}`);
  for (let i = 0; i < s.n; i++) out[i] = vals[i * cols + r.index - 1];
  return out;
};

/**
 * Expands wildcards — fix_ave_time.html: "the bracketed index I can be
 * specified using a wildcard asterisk with the index to effectively specify
 * multiple values. This takes the form "*" or "*n" or "m*" or "m*n". If N is
 * the size of the vector (for mode = scalar) or the number of columns in the
 * array (for mode = vector), then an asterisk with no numeric values means
 * all indices from 1 to N. A leading asterisk means all indices from 1 to n
 * (inclusive). A trailing asterisk means all indices from n to N
 * (inclusive). A middle asterisk means all indices from m to n (inclusive)."
 * `context` picks N: the global vector length ('scalar' inputs), the global
 * array column count ('vector' inputs) or the per-atom column count
 * ('peratom', as for compute reduce and fix ave/atom).
 */
export const expandWildcards = (sys: System, words: string[], context: 'scalar' | 'vector' | 'peratom'): string[] => {
  const out: string[] = [];
  for (const w of words) {
    const m = /^([cfv])_([A-Za-z0-9_\-/]+)\[(\d*)\*(\d*)\]$/.exec(w);
    if (!m) { out.push(w); continue; }
    const [, kind, id, lo, hi] = m;
    let n: number;
    if (kind === 'v') {
      if (context !== 'scalar') throw new StyleError(`${w}: a variable wildcard needs a vector-style variable in a scalar input list`);
      n = sys.vars.evalVector(id, sys.formulaEnv).length;
    } else {
      const obj = kind === 'c' ? sys.compute(id) : sys.fix(id);
      n = context === 'scalar' ? obj.sizeVector : context === 'vector' ? obj.sizeArrayCols : obj.sizePeratomCols;
      if (context === 'scalar' && kind === 'c' && obj.vectorFlag) n = sys.compute(id).vectorValues().length;
    }
    const a = lo === '' ? 1 : Number(lo), b = hi === '' ? n : Math.min(Number(hi), n);
    if (n < 1) throw new StyleError(`${w}: ${kind === 'c' ? 'compute' : kind === 'f' ? 'fix' : 'variable'} ${id} has no ${context === 'peratom' ? 'per-atom columns' : context === 'vector' ? 'array columns' : 'vector elements'} to expand`);
    for (let k = a; k <= b; k++) out.push(`${kind}_${id}[${k}]`);
  }
  return out;
};
