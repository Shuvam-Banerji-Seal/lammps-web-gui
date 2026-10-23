/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/compute_slice.cpp and
 * src/compute_slice.h at commit b332f28 (compute slice). Copyright (2003) Sandia Corporation (LAMMPS development team).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_slice.html: "compute ID group-ID slice Nstart Nstop Nskip input1 ... inputM"; "This compute
 * takes a global vector or global array and extracts a subset of its values, a slice of the vector or array"; "The
 * indices Nstart to Nstop are used (1-based), with an increment of Nskip."
 *
 * Scope of this port: a single input (a global vector result) is supported. With several inputs (a global array) the
 * native compute_array (src/compute_slice.cpp, "extract_one(0, &array[m][0], values.size())") always reads the first
 * input and writes column m through a flat pointer past the end of the array, so it is refused here rather than
 * reproduced. A slice with Nstop - Nstart not a multiple of Nskip is refused too: native sizes the vector with the
 * integer quotient but fills ceil(...) values.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { inumeric } from '../../engine/force/util';
import type { System } from '../../engine/system';

const ERR20 = '\nFor more information see https://docs.lammps.org/err0020';
const REF = /^([cfv])_([A-Za-z0-9_\-/]+)(?:\[(\d+)\])?$/;


/** One input: a global vector or array of a compute (c_), a fix (f_), or a vector-style variable (v_). */
interface SliceValue {
  kind: 'c' | 'f' | 'v';
  id: string;
  argindex: number;
  word: string;
}

export class ComputeSlice extends Compute {
  readonly style = 'slice';
  private readonly nstart: number;
  private readonly nstop: number;
  private readonly nskip: number;
  private readonly value: SliceValue;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError('Illegal compute slice command: missing argument(s)');
    // As native LAMMPS does (src/compute_slice.cpp ComputeSlice::ComputeSlice at b332f28): utils::inumeric reads nstart, nstop, nskip
    this.nstart = inumeric(args[0]);
    this.nstop = inumeric(args[1]);
    this.nskip = inumeric(args[2]);
    if (this.nstart < 1) throw new StyleError(`Invalid compute slice nstart value ${this.nstart} < 1`);
    if (this.nstop < this.nstart) throw new StyleError(`Invalid compute slice nstop value ${this.nstop} < ${this.nstart}`);
    if (this.nskip < 1) throw new StyleError(`Invalid compute slice nskip value < 1: ${this.nskip}`);
    // As native LAMMPS does (src/compute_slice.cpp ComputeSlice::ComputeSlice at b332f28): every input word is parsed first, and the
    // first word that is not a valid input is "Illegal compute slice argument", before any input is looked up.
    for (let k = 3; k < args.length; k++) {
      const mk = REF.exec(args[k]);
      if (!mk) throw new StyleError(`Illegal compute slice argument: ${args[k]}`);
    }
    if (args.length > 4) {
      throw new StyleError('compute slice with more than one input (a global array) is not supported by the engine yet: native compute_array reads input 1 for every column (src/compute_slice.cpp)');
    }
    const w = args[3];
    const m = REF.exec(w)!;
    const kind = m[1] as 'c' | 'f' | 'v';
    const argindex = m[3] === undefined ? 0 : Number(m[3]);
    if (argindex < 0) throw new StyleError(`Illegal compute slice argument: ${w}`);
    this.value = { kind, id: m[2], argindex, word: w };
    const nstop = this.nstop;
    const nskip = this.nskip;
    if (kind === 'c') {
      const c = sys.computes.find((x) => x.id === m[2]);
      if (!c) throw new StyleError(`Compute ID ${m[2]} for compute slice does not exist`);
      if (c.vectorFlag) {
        if (argindex) throw new StyleError(`Compute slice compute ${m[2]} does not calculate a global array`);
        if (nstop > c.sizeVector) throw new StyleError(`Compute slice compute ${m[2]} vector is accessed out-of-range${ERR20}`);
      } else if (c.arrayFlag) {
        if (argindex === 0) throw new StyleError(`Compute slice compute ${m[2]} does not calculate a global vector`);
        if (argindex > c.sizeArrayCols) throw new StyleError(`Compute slice compute ${m[2]} array is accessed out-of-range${ERR20}`);
        if (nstop > c.sizeArrayRows) throw new StyleError(`Compute slice compute ${m[2]} array is accessed out-of-range${ERR20}`);
      } else {
        throw new StyleError(`Compute slice compute ${m[2]} does not calculate global vector or array`);
      }
    } else if (kind === 'f') {
      const f = sys.fixes.find((x) => x.id === m[2]);
      if (!f) throw new StyleError(`Fix ID ${m[2]} for compute slice does not exist`);
      if (f.vectorFlag) {
        if (argindex) throw new StyleError(`Compute slice fix ${m[2]} does not calculate a global array`);
        if (nstop > f.sizeVector) throw new StyleError(`Compute slice fix ${m[2]} vector is accessed out-of-range${ERR20}`);
      } else if (f.arrayFlag) {
        if (argindex === 0) throw new StyleError(`Compute slice fix ${m[2]} does not calculate a global vector`);
        if (argindex > f.sizeArrayCols) throw new StyleError(`Compute slice fix ${m[2]} array is accessed out-of-range${ERR20}`);
        if (nstop > f.sizeArrayRows) throw new StyleError(`Compute slice fix ${m[2]} array is accessed out-of-range${ERR20}`);
      } else {
        throw new StyleError(`Compute slice fix ${m[2]} does not calculate global vector or array`);
      }
    } else {
      const v = sys.vars.get(m[2]);
      if (!v) throw new StyleError(`Variable name ${m[2]} for compute slice does not exist`);
      if (argindex === 0 && v.style !== 'vector') throw new StyleError(`Compute slice variable ${m[2]} is not vector-style variable`);
      if (argindex) throw new StyleError(`Compute slice vector variable ${m[2]} cannot be indexed`);
    }
    if ((nstop - this.nstart) % nskip !== 0) {
      throw new StyleError(`compute slice: Nstop - Nstart (${nstop - this.nstart}) must be a multiple of Nskip (${nskip}); native sizes the vector by the integer quotient and fills one more value`);
    }
    // a vector: extensive or intensive as the input (native constructor, single value)
    const size = Math.floor((nstop - this.nstart) / nskip);
    this.vectorFlag = true;
    this.sizeVector = size;
    this.vector = new Float64Array(size);
    // (the engine marks a vector whose elements differ, native's extvector -1, by its extlist; a column of an array
    // takes the array's flag, native "extvector = val.val.c->extarray")
    const src = kind === 'c' ? sys.computes.find((x) => x.id === m[2]) : kind === 'f' ? sys.fixes.find((x) => x.id === m[2]) : undefined;
    if (!src) {
      this.extvector = 0;
    } else if (argindex === 0) {
      this.extvector = src.extvector;
      if (src.extlist) {
        const list: number[] = [];
        for (let i = this.nstart; i < nstop; i += nskip) list.push(src.extlist[i - 1] ?? 0);
        this.extlist = list;
      }
    } else {
      this.extvector = src.extarray;
    }
  }

  protected computeVector(): void {
    const { kind, id, argindex } = this.value;
    const sys = this.sys;
    const out = this.vector;
    let j = 0;
    if (kind === 'c') {
      const c = sys.compute(id);
      if (argindex === 0) {
        const cv = c.vectorValues();
        for (let i = this.nstart; i < this.nstop; i += this.nskip) out[j++] = cv[i - 1];
      } else {
        const ca = c.arrayValues();
        const cols = c.sizeArrayCols;
        for (let i = this.nstart; i < this.nstop; i += this.nskip) out[j++] = ca[(i - 1) * cols + argindex - 1];
      }
    } else if (kind === 'f') {
      const f = sys.fix(id);
      const step = sys.state.step;
      if (f.globalFreq > 0 && step % f.globalFreq) {
        throw new StyleError(`Fix ${id} used in compute slice not computed at compatible time${ERR20}`);
      }
      for (let i = this.nstart; i < this.nstop; i += this.nskip) {
        out[j++] = argindex === 0 ? f.computeVector(i - 1) : f.computeArray(i - 1, argindex - 1);
      }
    } else {
      const vec = sys.vars.evalVector(id, sys.formulaEnv);
      if (vec.length < this.nstop) throw new StyleError(`Compute slice variable ${id} is not long enough`);
      for (let i = this.nstart; i < this.nstop; i += this.nskip) out[j++] = vec[i - 1];
    }
  }
}
