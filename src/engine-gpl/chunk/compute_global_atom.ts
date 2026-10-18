/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/compute_global_atom.cpp and
 * src/compute_global_atom.h at commit b332f28 (compute global/atom). Copyright (2003) Sandia Corporation (LAMMPS
 * development team). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_global_atom.html: "compute ID group-ID global/atom index value1 value2 ..."; "This compute
 * takes one or more global vectors or arrays and creates per-atom values from them, using per-atom integer indices"; the
 * index is "rounded down from double values" and "indices are decremented from 1 to N -> 0 to N-1", so the atom gets the
 * value at element index of the input. Atoms outside the group get 0. One value gives a per-atom vector, several a
 * per-atom array with one column per value.
 *
 * Refused (named StyleError): wildcard value arguments. Native init() tests the fix index reference the wrong way round
 * ("if (reference.val.f) error->all(... does not exist)"), so a f_ID index fails at the start of a run whenever the fix
 * exists; this port reproduces that refusal in init().
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';

const ERR20 = '\nFor more information see https://docs.lammps.org/err0020';
const REF = /^([cfv])_([A-Za-z0-9_\-/]+)(?:\[(\d+)\])?$/;

interface Src {
  kind: 'c' | 'f' | 'v';
  id: string;
  argindex: number;
}

const parseSrc = (w: string, msg: string): Src => {
  const m = REF.exec(w);
  if (!m) throw new StyleError(msg);
  return { kind: m[1] as Src['kind'], id: m[2], argindex: m[3] === undefined ? 0 : Number(m[3]) };
};

export class ComputeGlobalAtom extends Compute {
  readonly style = 'global/atom';
  private readonly reference: Src;
  private readonly values: Src[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError('Illegal compute global/atom command: missing argument(s)');
    this.reference = parseSrc(args[0], `Illegal compute global/atom index property: ${args[0]}`);
    this.values = args.slice(1).map((w) => {
      if (w.includes('*')) throw new StyleError(`compute global/atom: wildcard value ${w} is not supported by the engine yet`);
      return parseSrc(w, `Illegal compute global/atom global property: ${w}`);
    });
    const r = this.reference;
    if (r.kind === 'c') {
      const c = sys.computes.find((x) => x.id === r.id);
      if (!c) throw new StyleError(`Compute ID ${r.id} for compute global/atom index does not exist`);
      if (!c.peratomFlag) throw new StyleError(`Compute global/atom compute ${r.id} does not calculate a per-atom vector or array`);
      if (r.argindex === 0 && c.sizePeratomCols !== 0) throw new StyleError(`Compute global/atom compute ${r.id} does not calculate a per-atom vector`);
      if (r.argindex && c.sizePeratomCols === 0) throw new StyleError('Compute global/atom compute does not calculate a per-atom array');
      if (r.argindex && r.argindex > c.sizePeratomCols) throw new StyleError(`Compute global/atom compute ${r.id} array is accessed out-of-range${ERR20}`);
    } else if (r.kind === 'f') {
      const f = sys.fixes.find((x) => x.id === r.id);
      if (!f) throw new StyleError(`Fix ID ${r.id} for compute global/atom does not exist`);
      if (!f.peratomFlag) throw new StyleError(`Compute global/atom fix ${r.id} does not calculate a per-atom vector or array`);
      if (r.argindex === 0 && f.sizePeratomCols !== 0) throw new StyleError(`Compute global/atom fix ${r.id} does not calculate a per-atom vector`);
      if (r.argindex && f.sizePeratomCols === 0) throw new StyleError(`Compute global/atom fix ${r.id} does not calculate a per-atom array`);
      if (r.argindex && r.argindex > f.sizePeratomCols) throw new StyleError(`Compute global/atom fix ${r.id} array is accessed out-of-range${ERR20}`);
    } else {
      const v = sys.vars.get(r.id);
      if (!v) throw new StyleError(`Variable name ${r.id} for compute global/atom index does not exist`);
      if (v.style !== 'atom') throw new StyleError(`Compute global/atom index variable ${r.id} is not atom-style variable`);
    }
    for (const val of this.values) {
      if (val.kind === 'c') {
        const c = sys.computes.find((x) => x.id === val.id);
        if (!c) throw new StyleError(`Compute ID ${val.id} for compute global/atom does not exist`);
        if (val.argindex === 0) {
          if (!c.vectorFlag) throw new StyleError(`Compute ID ${val.id} for global/atom compute does not calculate a global vector`);
        } else {
          if (!c.arrayFlag) throw new StyleError(`Compute ID ${val.id} for global/atom compute does not calculate a global array`);
          if (val.argindex > c.sizeArrayCols) throw new StyleError(`Compute global/atom compute ${val.id} array is accessed out-of-range${ERR20}`);
        }
      } else if (val.kind === 'f') {
        const f = sys.fixes.find((x) => x.id === val.id);
        if (!f) throw new StyleError(`Fix ID ${val.id} for compute global/atom does not exist`);
        if (val.argindex === 0) {
          if (!f.vectorFlag) throw new StyleError(`Fix ID ${val.id} for compute global/atom compute does not calculate a global vector`);
        } else {
          if (!f.arrayFlag) throw new StyleError(`Fix ID ${val.id} for compute global/atom compute does not calculate a global array`);
          if (val.argindex > f.sizeArrayCols) throw new StyleError(`Compute global/atom fix ${val.id} array is accessed out-of-range${ERR20}`);
        }
      } else {
        const v = sys.vars.get(val.id);
        if (!v) throw new StyleError(`Variable name ${val.id} for compute global/atom does not exist`);
        if (v.style !== 'vector') throw new StyleError(`Compute global/atom variable ${val.id} is not vector-style variable`);
      }
    }
    this.peratomFlag = true;
    this.sizePeratomCols = this.values.length === 1 ? 0 : this.values.length;
  }

  init(): void {
    const r = this.reference;
    if (r.kind === 'c') {
      if (!this.sys.computes.some((x) => x.id === r.id)) throw new StyleError(`Compute ID ${r.id} for compute global/atom index does not exist`);
    } else if (r.kind === 'f') {
      // native ComputeGlobalAtom::init tests the fix pointer the wrong way round (src/compute_global_atom.cpp)
      if (this.sys.fixes.some((x) => x.id === r.id)) throw new StyleError(`Fix ID ${r.id} for compute global/atom index does not exist`);
    } else if (!this.sys.vars.get(r.id)) {
      throw new StyleError(`Variable name ${r.id} for compute global/atom index does not exist`);
    }
  }

  /** The per-atom integer index (1-based, truncated) of each atom from the index reference. */
  private indexValues(n: number): Float64Array {
    const sys = this.sys;
    const r = this.reference;
    const src = new Float64Array(n);
    if (r.kind === 'c') {
      const c = sys.compute(r.id);
      const pv = c.peratomValues();
      if (r.argindex === 0) for (let i = 0; i < n; i++) src[i] = pv[i];
      else {
        const cols = c.sizePeratomCols;
        for (let i = 0; i < n; i++) src[i] = pv[i * cols + r.argindex - 1];
      }
    } else if (r.kind === 'f') {
      throw new StyleError(`Fix ID ${r.id} for compute global/atom index does not exist`);
    } else {
      const av = sys.atomVariable(r.id);
      for (let i = 0; i < n; i++) src[i] = av[i];
    }
    for (let i = 0; i < n; i++) src[i] = Math.trunc(src[i]) - 1;
    return src;
  }

  /** The global source of one value: its elements and their number (vmax). */
  private sourceOf(val: Src): { source: Float64Array; vmax: number } {
    const sys = this.sys;
    if (val.kind === 'c') {
      const c = sys.compute(val.id);
      if (val.argindex === 0) return { source: c.vectorValues(), vmax: c.sizeVector };
      const arr = c.arrayValues();
      const cols = c.sizeArrayCols;
      const col = val.argindex - 1;
      const rows = c.sizeArrayRows;
      const source = new Float64Array(rows);
      for (let j = 0; j < rows; j++) source[j] = arr[j * cols + col];
      return { source, vmax: rows };
    }
    if (val.kind === 'f') {
      const f = sys.fix(val.id);
      if (val.argindex === 0) {
        const source = new Float64Array(f.sizeVector);
        for (let j = 0; j < f.sizeVector; j++) source[j] = f.computeVector(j);
        return { source, vmax: f.sizeVector };
      }
      const col = val.argindex - 1;
      const source = new Float64Array(f.sizeArrayRows);
      for (let j = 0; j < f.sizeArrayRows; j++) source[j] = f.computeArray(j, col);
      return { source, vmax: f.sizeArrayRows };
    }
    const vec = sys.vars.evalVector(val.id, sys.formulaEnv);
    return { source: vec, vmax: vec.length };
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const n = s.n;
    const idx = this.indexValues(n);
    const nv = this.values.length;
    const scalarOut = nv === 1;
    const out = scalarOut ? new Float64Array(n) : new Float64Array(n * nv);
    for (let k = 0; k < nv; k++) {
      const { source, vmax } = this.sourceOf(this.values[k]);
      for (let i = 0; i < n; i++) {
        let v = 0;
        if (s.mask[i] & this.groupBit) {
          const j = idx[i];
          if (j >= 0 && j < vmax) v = source[j];
        }
        if (scalarOut) out[i] = v;
        else out[i * nv + k] = v;
      }
    }
    if (scalarOut) this.vectorAtom = out;
    else this.arrayAtom = out;
  }
}
