import { Compute } from './compute';
import { ComputeOrientorderAtom } from './orientorder';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { peratomValues, type Ref } from '../refs';
import { bounds, inumeric, numeric } from '../force/util';
import { ALL_GROUP_BIT } from '../atoms';
import { CellList } from '../cellList';

/*
 * Wave-2 computes: reduce, reduce/region, displace/atom, coord/atom.
 * Each class cites and quotes the docs.lammps.org page it implements.
 */

const REDUCE_MODES = ['sum', 'min', 'minabs', 'max', 'maxabs', 'ave', 'sumsq', 'avesq', 'sumabs', 'aveabs'] as const;
type ReduceMode = (typeof REDUCE_MODES)[number];
/** compute_reduce.html: values are "extensive" only for these modes. */
const EXTENSIVE_MODES: ReadonlySet<string> = new Set(['sum', 'sumsq', 'sumabs']);

/*
 * compute ID group-ID style arg mode input1 input2 ... keyword args ...
 *   — docs.lammps.org/compute_reduce.html, with
 *     *reduce* arg = none
 *     *reduce/region* arg = region-ID
 *       region-ID = ID of region to use for choosing atoms
 *     mode = *sum* or *min* or *minabs* or *max* or *maxabs* or *ave* or *sumsq* or *avesq* or *sumabs* or *aveabs*
 *     input = *x* or *y* or *z* or *vx* or *vy* or *vz* or *fx* or *fy* or *fz* or c_ID or c_ID[N] or f_ID or f_ID[N] or v_name
 *     keyword = *replace* or *inputs*
 *     *replace* args = vec1 vec2
 *       vec1 = reduced value from this input vector will be replaced
 *       vec2 = replace it with vec1[N] where N is index of max/min value from vec2
 *     *inputs* arg = peratom or local
 *       peratom = all inputs are per-atom quantities (default)
 *       local = all input are local quantities
 *
 * "Define a calculation that "reduces" one or more vector inputs into
 * scalar values, one per listed input."
 *
 * "The reduction operation is specified by the *mode* setting.  The *sum*
 * option adds the values in the vector into a global total.  The *min*
 * or *max* options find the minimum or maximum value across all vector
 * values.  The *minabs* or *maxabs* options find the minimum or maximum
 * value across all absolute vector values.  The *ave* setting adds the
 * vector values into a global total, then divides by the number of
 * values in the vector.  The *sumsq* option sums the square of the
 * values in the vector into a global total.  The *avesq* setting does
 * the same as *sumsq*, then divides the sum of squares by the number of
 * values. ... The *sumabs* option sums the absolute values in the
 * vector into a global total.  The *aveabs* setting does the same as
 * *sumabs*, then divides the sum of absolute values by the number of
 * values."
 *
 * "For per-atom inputs,
 * the group specified with this command means only atoms within the
 * group contribute to the result.  Likewise for per-atom inputs, if the
 * compute reduce/region command is used, the atoms must also currently
 * be within the region."
 *
 * "If the *replace* keyword is used, two indices *vec1* and *vec2* are
 * specified, where each index ranges from 1 to the number of input
 * values.  The replace keyword can only be used if the *mode* is *min*
 * or *max*\ .  It works as follows.  A min/max is computed as usual on
 * the *vec2* input vector.  The index :math:`N` of that value within
 * *vec2* is also stored.  Then, instead of performing a min/max on the
 * *vec1* input vector, the stored index is used to select the :math:`N`\
 * th element of the *vec1* vector."
 *
 * "If a single input is specified this compute produces a global scalar
 * value.  If multiple inputs are specified, this compute produces a
 * global vector of values, the length of which is equal to the number of
 * inputs specified."
 *
 * "All the scalar or vector values calculated by this compute are
 * "intensive", except when the *sum*, *sumabs*, or *sumsq* modes are used on
 * per-atom or local vectors, in which case the calculated values are
 * "extensive"."
 *
 * Default: "The default value for the *inputs* keyword is peratom."
 *
 * Wildcards (same asterisk form as fix ave/time, documented on this page
 * for the bracketed index I of c_ID / f_ID inputs): expandWildcards in
 * refs.ts implements "This takes the form "*" or "*n" or "m*" or "m*n"."
 * Local array columns are expanded against the local column count here.
 *
 * inputs local (the *inputs* keyword "allows selection of whether all the
 * inputs are per-atom or local quantities"; "For the compute reduce command,
 * the inputs can be either per-atom or local quantities and must all be of
 * the same kind"; "If a compute *only* produces local
 * data, like for example the :doc:`compute bond/local command
 * <compute_bond_local>`, the setting "inputs local" is *required*."):
 * the engine reduces the local vectors and local array columns of its local
 * computes (bond/local, angle/local, dihedral/local, improper/local,
 * property/local, pair/local and voronoi/atom neighbors yes) through the same
 * modes. The engine has no fix with local output, so an f_ID local input is a
 * StyleError naming it.
 *
 * Measured with native LAMMPS (black box): the group of compute reduce does
 * not filter local rows — reduce sum c_bond[1] is identical for group all and
 * for a subgroup, so the only group that restricts local rows is the one of
 * the input compute; the count used by ave is the number of local rows.
 * Measured with native LAMMPS (black box): an empty local list gives 0 for
 * sum, sumsq, sumabs, ave, avesq and aveabs, 1e20 for min and minabs, -1e20
 * for max and 0 for maxabs. Measured with native LAMMPS (black box):
 * compute reduce/region rejects local inputs ("Compute reduce/region cannot
 * use local data as input") and mixing a per-atom and a local input in one
 * compute is an error.
 */
export class ComputeReduce extends Compute {
  readonly style: string = 'reduce';
  protected readonly mode: ReduceMode;
  protected readonly inputs: Ref[];
  private readonly replaces: Array<readonly [number, number]> = [];
  /** The input words as written (after wildcard expansion), for the thermo column names. */
  private inputWords: string[] = [];
  private readonly regionId: string | null;
  /** True when the *inputs local* keyword selected local quantities (default peratom). */
  private localInputs = false;

  constructor(sys: System, id: string, group: string, args: string[], regionId: string | null = null) {
    super(sys, id, group, args);
    this.regionId = regionId;
    // As native LAMMPS does (src/compute_reduce.cpp ComputeReduce::ComputeReduce at b332f28): the mode, the inputs up to
    // the first word that is no input, then the keywords, then each input against its compute, fix or variable
    const style = regionId === null ? 'reduce' : 'reduce/region';
    if (args.length < 2) throw new StyleError(`Illegal compute ${style} command: missing argument(s)`);
    const mode = args[0];
    if (!(REDUCE_MODES as readonly string[]).includes(mode)) throw new StyleError(`Unknown compute ${style} mode: ${mode}`);
    this.mode = mode as ReduceMode;
    const words = expandArgs(sys, args.slice(1));
    let k = 0;
    for (; k < words.length; k++) {
      const a = argInfo(words[k]);
      if (a.type === 'unknown' || a.dim > 1) throw new StyleError(`Illegal compute ${style} argument: ${words[k]}`);
      if (a.type === 'none' && !ATOM_ATTRS.includes(words[k])) break;
    }
    const inputWords = words.slice(0, k);
    const replace = new Array<number>(inputWords.length).fill(-1);
    for (; k < words.length; k++) {
      const w = words[k];
      if (w === 'replace') {
        if (k + 3 > words.length) throw new StyleError(`Illegal compute ${style} replace command: missing argument(s)`);
        if (this.mode !== 'min' && this.mode !== 'max') throw new StyleError(`Compute ${style} replace requires min or max mode`);
        const col1 = inumeric(words[k + 1]) - 1;
        const col2 = inumeric(words[k + 2]) - 1;
        if (col1 < 0 || col1 >= inputWords.length) throw new StyleError(`Invalid compute ${style} replace first column index ${col1}`);
        if (col2 < 0 || col2 >= inputWords.length) throw new StyleError(`Invalid compute ${style} replace second column index ${col2}`);
        // native's two messages below have no argument for their "{}" and print it as it is
        if (col1 === col2) throw new StyleError('Compute {} replace columns must be different');
        if (replace[col1] >= 0 || replace[col2] >= 0) throw new StyleError('Compute {} replace column already used for another replacement');
        replace[col1] = col2;
        this.replaces.push([col1 + 1, col2 + 1]);
        k += 2;
      } else if (w === 'inputs') {
        if (k + 2 > words.length) throw new StyleError(`Illegal compute ${style} inputs command: missing argument(s)`);
        if (words[k + 1] === 'peratom') this.localInputs = false;
        else if (words[k + 1] === 'local') this.localInputs = true;
        else throw new StyleError(`Unknown compute ${style} inputs argument: ${words[k + 1]}`);
        k += 1;
      } else throw new StyleError(`Unknown compute ${style} keyword: ${w}`);
    }
    this.inputWords = inputWords;
    this.inputs = inputWords.map((w) => this.parseInput(w, style));
    // src/compute_reduce_region.cpp: "Compute reduce/region cannot use local data as input"
    if (regionId !== null && this.localInputs) throw new StyleError('Compute reduce/region cannot use local data as input');
    const n = this.inputs.length;
    if (n === 1) {
      this.scalarFlag = true;
      this.extscalar = EXTENSIVE_MODES.has(this.mode) ? 1 : 0;
    } else {
      this.vectorFlag = true;
      this.sizeVector = n;
      this.vector = new Float64Array(n);
      this.extvector = EXTENSIVE_MODES.has(this.mode) ? 1 : 0;
    }
  }

  /** One input: must be a producer of the selected kind ("must all be of the same kind (per-atom or local)"). */

  /**
   * thermo_modify colname auto names (compute_reduce.rst, versionadded 10Dec2025): the compute ID, a colon, the
   * mode, then the input in parentheses; with the replace keyword, the replaced input, a left arrow, the mode and
   * the second input. Measured with native LAMMPS (black box): compute r all reduce min x y gives c_r:min(x) and
   * c_r:min(y) for c_r[1] and c_r[2].
   */
  autoColumnNames(): { scalar?: string; vector?: string[] } {
    const names = this.inputWords.map((w) => `${this.mode}(${w})`);
    for (const [v1, v2] of this.replaces) {
      const first = this.inputWords[v1 - 1], second = this.inputWords[v2 - 1];
      if (first !== undefined && second !== undefined) names[v1 - 1] = `${first}<-${this.mode}(${second})`;
    }
    if (names.length === 1) return { scalar: names[0] };
    return { vector: names };
  }

  /** One input, checked against its compute, fix or variable for the per-atom or local inputs (native's messages). */
  private parseInput(w: string, style: string): Ref {
    if (ATOM_ATTRS.includes(w)) {
      // native's message has no argument for its "{}"
      if (this.localInputs) throw new StyleError('Compute {} inputs must be all local');
      return { kind: 'attr', id: w, index: null, text: w };
    }
    const a = argInfo(w);
    const r: Ref = { kind: a.type as 'c' | 'f' | 'v', id: a.name, index: a.index > 0 ? a.index : null, text: w };
    if (r.kind === 'v') {
      if (this.localInputs) throw new StyleError('Compute {} inputs must be all local');
      const v = this.sys.vars.get(r.id);
      if (!v) throw new StyleError(`Variable name ${r.id} for compute ${style} does not exist`);
      if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`Compute ${style} variable ${r.id} is not atom-style variable`);
      return r;
    }
    const kind = r.kind === 'c' ? 'compute' : 'fix';
    const obj = r.kind === 'c' ? this.sys.computes.find((c) => c.id === r.id) : this.sys.fixes.find((f) => f.id === r.id);
    if (!obj) throw new StyleError(`${r.kind === 'c' ? 'Compute ID' : 'Fix ID'} ${r.id} for compute ${style} does not exist`);
    const range = `Compute ${style} ${kind} ${r.id} array is accessed out-of-range\nFor more information see https://docs.lammps.org/err0020`;
    if (this.localInputs) {
      const local = (obj as { localFlag?: boolean }).localFlag ?? false;
      const cols = (obj as { sizeLocalCols?: number }).sizeLocalCols ?? 0;
      if (!local) throw new StyleError(`Compute ${style} ${kind} ${r.id} does not calculate local values`);
      if (r.index === null && cols !== 0) throw new StyleError(`Compute ${style} ${kind} ${r.id} does not calculate a local vector`);
      if (r.index !== null && cols === 0) throw new StyleError(`Compute ${style} ${kind} ${r.id} does not calculate a local array`);
      if (r.index !== null && r.index > cols) throw new StyleError(range);
      return r;
    }
    if (!obj.peratomFlag) throw new StyleError(`Compute ${style} ${kind} ${r.id} does not calculate per-atom values`);
    if (r.index === null && obj.sizePeratomCols !== 0) throw new StyleError(`Compute ${style} ${kind} ${r.id} does not calculate a per-atom vector`);
    if (r.index !== null && obj.sizePeratomCols === 0) throw new StyleError(`Compute ${style} ${kind} ${r.id} does not calculate a per-atom array`);
    if (r.index !== null && r.index > obj.sizePeratomCols) throw new StyleError(range);
    return r;
  }

  /** As native LAMMPS does (ComputeReduce::init at b332f28): the computes, fixes and variables of the inputs and the region again. */
  init(): void {
    const style = this.regionId === null ? 'reduce' : 'reduce/region';
    for (const r of this.inputs) {
      if (r.kind === 'c' && !this.sys.computes.some((c) => c.id === r.id)) throw new StyleError(`Compute ID ${r.id} for compute ${style} does not exist`);
      if (r.kind === 'f' && !this.sys.fixes.some((f) => f.id === r.id)) throw new StyleError(`Fix ID ${r.id} for compute ${style} does not exist`);
      if (r.kind === 'v' && !this.sys.vars.has(r.id)) throw new StyleError(`Variable name ${r.id} for compute ${style} does not exist`);
    }
    if (this.regionId !== null && !this.sys.regions.has(this.regionId)) throw new StyleError(`Region ${this.regionId} for compute reduce/region does not exist`);
  }

  /** Atoms that contribute: the compute group, plus (reduce/region) "the atoms must also currently be within the region". */
  protected selected(i: number): boolean {
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return false;
    if (this.regionId !== null) {
      const r = this.sys.region(this.regionId);
      if (!r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) return false;
    }
    return true;
  }

  /** One mode reduction over the selected atoms of one per-atom input vector. */
  private reduceOne(vals: Float64Array): number {
    let sum = 0, sumsq = 0, sumabs = 0, count = 0;
    let mn = Infinity, mx = -Infinity, mnabs = Infinity, mxabs = -Infinity;
    for (let i = 0; i < this.sys.state.n; i++) {
      if (!this.selected(i)) continue;
      const v = vals[i];
      count++;
      sum += v;
      sumsq += v * v;
      const a = Math.abs(v);
      sumabs += a;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      if (a < mnabs) mnabs = a;
      if (a > mxabs) mxabs = a;
    }
    switch (this.mode) {
      case 'sum': return sum;
      // per-atom input with no atom selected: min and max keep their 1e20 start values (measured with native
      // LAMMPS, black box: the pour thermo row at step 0 has no atoms and reports c_tmin 1e+20 and c_tmax -1e+20)
      case 'min': return count ? mn : 1e20;
      case 'minabs': return count ? mnabs : 0;
      case 'max': return count ? mx : -1e20;
      case 'maxabs': return count ? mxabs : 0;
      case 'ave': return count ? sum / count : 0;
      case 'sumsq': return sumsq;
      case 'avesq': return count ? sumsq / count : 0;
      case 'sumabs': return sumabs;
      case 'aveabs': return count ? sumabs / count : 0;
    }
  }

  /**
   * One mode reduction over the local rows of one local input (a local vector, or one column of a
   * local array). "Each listed input is operated on independently"; the rows come straight from the
   * local compute, whose own group already selected them (the reduce group does not filter them).
   * Empty-list sentinels are the measured native ones (see the module comment).
   */
  private reduceLocalOne(r: Ref): number {
    const src = this.localSource(r);
    let sum = 0, sumsq = 0, sumabs = 0;
    let mn = Infinity, mx = -Infinity, mnabs = Infinity, mxabs = 0;
    for (let row = 0; row < src.rows; row++) {
      const v = src.data[row * src.ncol + src.col];
      sum += v;
      sumsq += v * v;
      const a = Math.abs(v);
      sumabs += a;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      if (a < mnabs) mnabs = a;
      if (a > mxabs) mxabs = a;
    }
    switch (this.mode) {
      case 'sum': return sum;
      case 'min': return src.rows ? mn : 1e20;
      case 'minabs': return src.rows ? mnabs : 1e20;
      case 'max': return src.rows ? mx : -1e20;
      case 'maxabs': return src.rows ? mxabs : 0;
      case 'ave': return src.rows ? sum / src.rows : 0;
      case 'sumsq': return sumsq;
      case 'avesq': return src.rows ? sumsq / src.rows : 0;
      case 'sumabs': return sumabs;
      case 'aveabs': return src.rows ? sumabs / src.rows : 0;
    }
  }

  /** The local data of one validated local input: its rows, column count and selected column. */
  private localSource(r: Ref): { data: Float64Array; ncol: number; col: number; rows: number } {
    const c = this.sys.compute(r.id);
    const data = c.localValues(); // evaluated once per state epoch; sets localRows
    return { data, ncol: Math.max(1, c.sizeLocalCols), col: r.index === null ? 0 : r.index - 1, rows: c.localRows };
  }

  protected computeScalar(): number {
    return this.localInputs ? this.reduceLocalOne(this.inputs[0]) : this.reduceOne(peratomValues(this.sys, this.inputs[0]));
  }

  protected computeVector(): void {
    if (this.localInputs) {
      for (let k = 0; k < this.inputs.length; k++) this.vector[k] = this.reduceLocalOne(this.inputs[k]);
      this.applyReplacesLocal();
      return;
    }
    for (let k = 0; k < this.inputs.length; k++) this.vector[k] = this.reduceOne(peratomValues(this.sys, this.inputs[k]));
    this.applyReplaces();
  }

  /** replace on local inputs: the vec2 min/max row selects the reported value of vec1 (compute_reduce.html). */
  private applyReplacesLocal(): void {
    if (!this.replaces.length) return;
    for (const [v1, v2] of this.replaces) {
      const a = this.localSource(this.inputs[v1 - 1]);
      const b = this.localSource(this.inputs[v2 - 1]);
      let best = this.mode === 'min' ? Infinity : -Infinity;
      let idx = -1;
      for (let row = 0; row < b.rows; row++) {
        const v = b.data[row * b.ncol + b.col];
        if (this.mode === 'min' ? v < best : v > best) { best = v; idx = row; }
      }
      if (idx >= 0 && idx < a.rows) this.vector[v1 - 1] = a.data[idx * a.ncol + a.col];
    }
  }

  /** replace: the vec2 min/max index selects the reported element of vec1 (compute_reduce.html). */
  private applyReplaces(): void {
    if (!this.replaces.length) return;
    const s = this.sys.state;
    for (const [v1, v2] of this.replaces) {
      const va = peratomValues(this.sys, this.inputs[v1 - 1]);
      const vb = peratomValues(this.sys, this.inputs[v2 - 1]);
      let best = this.mode === 'min' ? Infinity : -Infinity;
      let idx = -1;
      for (let i = 0; i < s.n; i++) {
        if (!this.selected(i)) continue;
        if (this.mode === 'min' ? vb[i] < best : vb[i] > best) { best = vb[i]; idx = i; }
      }
      if (idx >= 0) this.vector[v1 - 1] = va[idx];
    }
  }
}

/*
 * compute ID group-ID reduce/region region-ID mode input1 ... — the
 * reduce/region variant on docs.lammps.org/compute_reduce.html:
 * "The compute reduce/region command can only be
 * used with per-atom inputs." and "the atoms must also currently
 * be within the region."
 */
export class ComputeReduceRegion extends ComputeReduce {
  readonly style: string = 'reduce/region';

  constructor(sys: System, id: string, group: string, args: string[]) {
    // src/compute_reduce.cpp: three words at least, and the region must exist
    if (args.length < 3) throw new StyleError('Illegal compute reduce/region command: missing argument(s)');
    if (!sys.regions.has(args[0])) throw new StyleError(`Region ${args[0]} for compute reduce/region does not exist`);
    super(sys, id, group, args.slice(1), args[0]);
  }
}

/*
 * compute ID group-ID displace/atom — docs.lammps.org/compute_displace_atom.html:
 *   compute ID group-ID displace/atom
 * "zero or more keyword/arg pairs may be appended" and
 *     keyword = *refresh*
 *     *refresh* arg = name of per-atom variable
 * "Define a computation that calculates the current displacement of each
 * atom in the group from its original (reference) coordinates, including
 * all effects due to atoms passing through periodic boundaries."
 *
 * "A vector of four quantities per atom is calculated by this compute.
 * The first three elements of the vector are the :math:`(dx,dy,dz)`
 * displacements.  The fourth component is the total displacement
 * (i.e., :math:`\sqrt{dx^2 + dy^2 + dz^2}`)."
 *
 * "The displacement of an atom is from its original position at the time
 * the compute command was issued.  The value of the displacement will be
 * 0.0 for atoms not in the specified compute group."
 *
 * "Initial coordinates are stored in "unwrapped" form, by using the
 * image flags associated with each atom."
 *
 * "This compute calculates a per-atom array with four columns, which can
 * be accessed by indices 1--4 by any command that uses per-atom values
 * from a compute as input."  "The per-atom array values will be in
 * distance units."  Default: none.
 *
 * The refresh option: "The refresh argument for this compute is the ID of
 * an atom-style variable which calculates a Boolean value (0 or 1) based on
 * the same criterion used by dump_modify thresh.  This compute
 * evaluates the atom-style variable.  For each atom that returns 1 (true),
 * the original (reference) coordinates of the atom (stored by
 * this compute) are updated."  (The engine's dump has no dump_modify
 * refresh yet; the compute side of the option is implemented and
 * callable.)
 */
export class ComputeDisplaceAtom extends Compute {
  readonly style = 'displace/atom';
  peratomFlag = true;
  sizePeratomCols = 4;
  private readonly refreshVar: string | null;
  /** Unwrapped reference positions (3n) captured when the compute command was issued. */
  private ref: Float64Array;
  private readonly u = [0, 0, 0];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    let refreshVar: string | null = null;
    for (let k = 0; k < args.length; k += 2) {
      // As native LAMMPS does (src/compute_displace_atom.cpp at b332f28)
      if (args[k] !== 'refresh' || k + 1 >= args.length) throw new StyleError('Illegal compute displace/atom command');
      refreshVar = args[k + 1];
    }
    this.refreshVar = refreshVar;
    // reference = "original position at the time the compute command was issued", unwrapped
    const s = sys.state;
    const g = sys.geom;
    this.ref = new Float64Array(3 * s.n);
    for (let i = 0; i < s.n; i++) {
      g.unwrap(s.x, s.image, i, this.u);
      this.ref[3 * i] = this.u[0]; this.ref[3 * i + 1] = this.u[1]; this.ref[3 * i + 2] = this.u[2];
    }
  }

  init(): void {
    if (!this.refreshVar) return;
    const v = this.sys.vars.get(this.refreshVar);
    if (!v) throw new StyleError(`compute ${this.id} (displace/atom): variable ${this.refreshVar} does not exist`);
    if (v.style !== 'atom') {
      throw new StyleError(`compute ${this.id} (displace/atom): variable ${this.refreshVar} must be atom-style (is ${v.style}-style)`);
    }
  }

  /** dump_modify refresh calls this at the end of every dump. */
  refresh(): void {
    if (!this.refreshVar) return;
    this.init();
    const s = this.sys.state;
    const g = this.sys.geom;
    const flag = this.sys.atomVariable(this.refreshVar);
    this.ensureRef();
    for (let i = 0; i < s.n; i++) {
      if (flag[i] === 0) continue;
      g.unwrap(s.x, s.image, i, this.u);
      this.ref[3 * i] = this.u[0]; this.ref[3 * i + 1] = this.u[1]; this.ref[3 * i + 2] = this.u[2];
    }
    this.invalidate();
  }

  /** Grows the reference storage for atoms added after the compute was defined (their displacement starts at 0). */
  private ensureRef(): void {
    const s = this.sys.state;
    const n = s.n;
    if (this.ref.length === 3 * n) return;
    const next = new Float64Array(3 * n);
    const keep = Math.min(this.ref.length / 3, n);
    next.set(this.ref.subarray(0, 3 * keep));
    for (let i = keep; i < n; i++) {
      this.sys.geom.unwrap(s.x, s.image, i, this.u);
      next[3 * i] = this.u[0]; next[3 * i + 1] = this.u[1]; next[3 * i + 2] = this.u[2];
    }
    this.ref = next;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    this.ensureRef();
    const out = this.arrayAtom = new Float64Array(4 * s.n);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue; // "0.0 for atoms not in the specified compute group"
      g.unwrap(s.x, s.image, i, this.u);
      const dx = this.u[0] - this.ref[3 * i];
      const dy = this.u[1] - this.ref[3 * i + 1];
      const dz = this.u[2] - this.ref[3 * i + 2];
      out[4 * i] = dx;
      out[4 * i + 1] = dy;
      out[4 * i + 2] = dz;
      out[4 * i + 3] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
  }
}

/*
 * compute ID group-ID coord/atom style args ... —
 * docs.lammps.org/compute_coord_atom.html, with
 *     *cutoff* args = cutoff [*group* group2-ID] typeN
 *       cutoff = distance within which to count coordination neighbors (distance units)
 *       *group* group2-ID = select group-ID to restrict which atoms to consider for coordination number (optional)
 *       typeN = atom type for Nth coordination count (see asterisk form below)
 *     *orientorder* args = orientorderID threshold
 *       orientorderID = ID of an orientorder/atom compute
 *       threshold = minimum value of the product of two "connected" atoms
 *
 * "The *cutoff* cstyle calculates one or more traditional coordination
 * numbers for each atom.  A coordination number is defined as the number
 * of neighbor atoms with specified atom type(s), and optionally within
 * the specified group, that are within the specified cutoff distance from
 * the central atom. The compute group selects only the central atoms; all
 * neighboring atoms, unless selected by type, type range, or group option,
 * are included in the coordination number tally."
 *
 * "One coordination number is
 * computed for each of the *typeN* keywords listed.  If no *typeN*
 * keywords are listed, a single coordination number is calculated, which
 * includes atoms of all types (same as the "\*" format, see below)."  The
 * typeN wildcards: "This takes the form "\*" or "\*n" or "m\*" or "m\*n". If N
 * is the
 * number of atom types, then an asterisk with no numeric values means all
 * types from 1 to N.  A leading asterisk means all types from 1 to n
 * (inclusive).  A trailing asterisk means all types from m to N
 * (inclusive).  A middle asterisk means all types from m to n (inclusive)."
 *
 * "For all *cstyle* settings, all coordination values will be 0.0 for
 * atoms not in the specified compute group."
 *
 * Output info: "For *cstyle* cutoff, this compute can calculate a per-atom
 * vector or
 * array.  If single *type1* keyword is specified (or if none are
 * specified), this compute calculates a per-atom vector.  If multiple
 * *typeN* keywords are specified, this compute calculates a per-atom
 * array, with :math:`N` columns."  Default: "group = all".
 *
 * The orientorder cstyle — the docs: "The *orientorder* cstyle calculates
 * the number of "connected" neighbor atoms J around each central atom I.
 * For this *cstyle*, connected is defined by the orientational order
 * parameter calculated by the compute orientorder/atom command. This
 * *cstyle* thus allows one to apply the ten Wolde's criterion to identify
 * crystal-like atoms in a system". As native LAMMPS does
 * (src/compute_coord_atom.cpp at b332f28): a neighbor J of any type or group
 * within the orientorder compute's cutoff is connected when the dot product
 * of the two atoms' normalized q_l component vectors (the components
 * columns) exceeds the threshold; native reads those columns from the
 * column after the q_l values, so with wl or wl/hat it reads those columns
 * instead, and so does the engine.  Neighbors are counted
 * with minimum-image distances (geom.minimumImage), valid for any cutoff
 * smaller than half the box.  Like LAMMPS's neighbor-list-based tally, the
 * count is over pairs closer than the cutoff; with special_bonds settings
 * that remove 1-2/1-3/1-4 pairs those pairs drop out of LAMMPS's list (the
 * note on the doc page) — for molecular systems the engine counts all pairs
 * within the cutoff instead.
 */
export class ComputeCoordAtom extends Compute {
  readonly style = 'coord/atom';
  peratomFlag = true;
  private readonly cutoff: number;
  private readonly group2Bit: number = ALL_GROUP_BIT; // "Default setting is group 'all.'"
  private readonly ranges: readonly (readonly [number, number])[];
  /** cstyle orientorder: the orientorder/atom compute and the threshold. */
  private readonly orient: { id: string; threshold: number } | null = null;
  private counts = new Float64Array(0);
  private readonly d = [0, 0, 0];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/compute_coord_atom.cpp ComputeCoordAtom::ComputeCoordAtom at b332f28): two words at least
    if (args.length < 2) throw new StyleError('Illegal compute coord/atom command');
    const cstyle = args[0];
    if (cstyle === 'orientorder') {
      // native's checks and texts (compute_coord_atom.cpp constructor)
      if (args.length !== 3) throw new StyleError('Illegal compute coord/atom command');
      const c = sys.computes.find((x) => x.id === args[1]);
      if (!c) throw new StyleError(`Could not find compute coord/atom compute ID ${args[1]}`);
      if (!(c instanceof ComputeOrientorderAtom)) throw new StyleError(`Compute coord/atom compute ID ${args[1]} is not orientorder/atom`);
      const threshold = Number(args[2]);
      if (!Number.isFinite(threshold)) throw new StyleError(`Expected floating point parameter instead of '${args[2]}' in input script or data file`);
      if (threshold <= -1.0 || threshold >= 1.0) throw new StyleError('Compute coord/atom threshold not between -1 and 1');
      this.orient = { id: args[1], threshold };
      this.cutoff = 0;
      this.ranges = [[1, sys.state.ntypes]];
      return;
    }
    if (cstyle !== 'cutoff') throw new StyleError('Invalid cstyle in compute coord/atom');
    // utils::numeric reads the cutoff and takes any value: only its square (cutsq) is used, so a negative cutoff acts as
    // its magnitude and 0 counts nothing. "group group2-ID" is read only right after the cutoff and with a word after
    // it; each later word is a type range read by utils::bounds over 1..ntypes, low > high refused
    this.cutoff = Math.abs(numeric(args[1]));
    const ntypes = sys.state.ntypes;
    const ranges: Array<readonly [number, number]> = [];
    let k = 2;
    if (args.length > 3 && args[2] === 'group') {
      if (sys.groups.find(args[3]) < 0) throw new StyleError(`Group ID ${args[3]} requested by compute coord/atom does not exist`);
      this.group2Bit = sys.groups.bit(args[3]);
      k = 4;
    }
    for (; k < args.length; k++) {
      const r = bounds(args[k], 1, ntypes);
      if (r[0] > r[1]) throw new StyleError('Illegal compute coord/atom command');
      ranges.push(r);
    }
    this.ranges = ranges.length ? ranges : ([[1, ntypes]] as const); // no typeN: "includes atoms of all types"
    if (this.ranges.length > 1) this.sizePeratomCols = this.ranges.length; // else per-atom vector
  }

  init(): void {
    // As native LAMMPS does (src/compute_coord_atom.cpp ComputeCoordAtom::init at b332f28): the orientorder/atom
    // compute's cutoff (cstyle orientorder), then a pair style and a cutoff no longer than its largest cutoff
    let cut = this.cutoff;
    if (this.orient) {
      const c = this.sys.computes.find((x) => x.id === this.orient!.id);
      if (!(c instanceof ComputeOrientorderAtom)) throw new StyleError(`Could not find compute coord/atom compute ID ${this.orient.id}`);
      if (c.qlcomp === null) throw new StyleError('Compute coord/atom requires components option in compute orientorder/atom');
      cut = c.cutoffKeyword ?? this.sys.ff.cutforce;
    }
    if (!this.sys.ff.pair) throw new StyleError('Compute coord/atom requires a pair style be defined');
    if (Math.sqrt(cut * cut) > this.sys.ff.cutforce) throw new StyleError('Compute coord/atom cutoff is longer than pairwise cutoff');
  }

  /** cstyle orientorder: the connected neighbors of each group atom (any type or group), see the header. */
  private computeOrient(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const n = s.n;
    const c = this.sys.computes.find((x) => x.id === this.orient!.id) as ComputeOrientorderAtom;
    const l = c.qlcomp!;
    const normv = c.peratomValues();
    const ncol = c.sizePeratomCols;
    const off = c.nqlist;
    const nc = 2 * (2 * l + 1);
    const cut = c.resolveCutoff();
    const cut2 = cut * cut;
    const threshold = this.orient!.threshold;
    const out = (this.vectorAtom = new Float64Array(n));
    const three = s.dimension === 3;
    const xc = three ? s.x : Float64Array.from({ length: 3 * n }, (_, q) => (q % 3 === 2 ? 0 : s.x[q]));
    const cells = new CellList(xc, n, g.triclinic ? 0 : cut, { length: [g.lx, g.ly, g.lz], periodic: g.periodic });
    const d = this.d;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const xi = s.x[3 * i], yi = s.x[3 * i + 1], zi = s.x[3 * i + 2];
      const cand = cells.candidates(xi, yi, three ? zi : 0);
      let count = 0;
      for (let q = 0; q < cand.length; q++) {
        const j = cand[q];
        if (j === i) continue;
        d[0] = xi - s.x[3 * j];
        d[1] = yi - s.x[3 * j + 1];
        d[2] = three ? zi - s.x[3 * j + 2] : 0;
        g.minimumImage(d);
        if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] >= cut2) continue;
        let dot = 0.0;
        for (let m = 0; m < nc; m++) dot += normv[i * ncol + off + m] * normv[j * ncol + off + m];
        if (dot > threshold) count++;
      }
      out[i] = count;
    }
  }

  protected computePeratom(): void {
    if (this.orient) { this.computeOrient(); return; }
    const s = this.sys.state;
    const g = this.sys.geom;
    const n = s.n;
    const cols = this.ranges.length;
    const out = cols > 1 ? this.arrayAtom = new Float64Array(cols * n) : this.vectorAtom = new Float64Array(n);
    if (this.counts.length !== cols) this.counts = new Float64Array(cols);
    const counts = this.counts;
    const cut2 = this.cutoff * this.cutoff;
    const three = s.dimension === 3;
    // the minimum-image distance is searched in the cells around i (an orthogonal box; a triclinic box has one cell).
    // In 2d the z component is ignored by the distance, so the cells see z = 0.
    const xc = three ? s.x : Float64Array.from({ length: 3 * n }, (_, q) => (q % 3 === 2 ? 0 : s.x[q]));
    const cells = new CellList(xc, n, g.triclinic ? 0 : this.cutoff, { length: [g.lx, g.ly, g.lz], periodic: g.periodic });
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue; // "0.0 for atoms not in the specified compute group"
      counts.fill(0);
      const xi = s.x[3 * i], yi = s.x[3 * i + 1], zi = s.x[3 * i + 2];
      // candidates: the atoms of the cells around i, ascending (the same order as the all-atom loop)
      const cand = cells.candidates(xi, yi, three ? zi : 0);
      for (let c = 0; c < cand.length; c++) {
        const j = cand[c];
        if (j === i) continue;
        if (!(s.mask[j] & this.group2Bit)) continue;
        const d = this.d;
        d[0] = xi - s.x[3 * j];
        d[1] = yi - s.x[3 * j + 1];
        d[2] = three ? zi - s.x[3 * j + 2] : 0;
        g.minimumImage(d);
        if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] >= cut2) continue;
        const t = s.type[j];
        for (let k = 0; k < cols; k++) {
          if (t >= this.ranges[k][0] && t <= this.ranges[k][1]) counts[k]++;
        }
      }
      for (let k = 0; k < cols; k++) {
        if (cols > 1) out[cols * i + k] = counts[k];
        else out[i] = counts[k];
      }
    }
  }
}

/** The atom attributes compute reduce takes by name (compute_reduce.cpp: x, y, z, vx, vy, vz, fx, fy, fz). */
const ATOM_ATTRS = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz'];

/**
 * A c_, f_ or v_ word as native's ArgInfo reads it (src/arg_info.cpp at b332f28): the name up to a bracket, the number of
 * brackets and the first index (std::stoi of the bracket's text); 'none' for any other word (it ends the inputs) and
 * 'unknown' for brackets that do not read.
 */
const argInfo = (w: string): { type: 'c' | 'f' | 'v' | 'none' | 'unknown'; name: string; dim: number; index: number } => {
  if (!(w.length > 2 && w[1] === '_' && (w[0] === 'c' || w[0] === 'f' || w[0] === 'v'))) return { type: 'none', name: w, dim: 0, index: 0 };
  const type = w[0] as 'c' | 'f' | 'v';
  const open1 = w.indexOf('[', 2);
  if (open1 < 0) return { type, name: w.slice(2), dim: 0, index: 0 };
  const name = w.slice(2, open1);
  const open2 = w.indexOf('[', open1 + 1);
  const stoi = (t: string): number | null => { const m = /^\s*[+-]?\d+/.exec(t); return m ? Number(m[0]) : null; };
  if (open2 >= 0) return { type: 'unknown', name, dim: 2, index: 0 };
  if (w[w.length - 1] !== ']' || w.indexOf(']') !== w.length - 1) return { type: 'unknown', name, dim: 1, index: 0 };
  const index = stoi(w.slice(open1 + 1, w.length - 1));
  if (index === null) return { type: 'unknown', name, dim: 1, index: 0 };
  return { type, name, dim: 1, index };
};

/*
 * Wildcard expansion of c_ID[*] and f_ID[*] inputs. As native LAMMPS does (utils::expand_args with mode 1, src/utils.cpp at
 * b332f28): the columns of a global array, else of a per-atom array, else of a local array; a word whose compute or fix
 * has none of them, or does not exist, is kept as written; the range is read by utils::bounds.
 */
const expandArgs = (sys: System, words: string[]): string[] => {
  const out: string[] = [];
  for (const w of words) {
    const m = /^([cf])_(\w+)\[(\d*\*\d*)\](.*)$/.exec(w);
    if (!m) { out.push(w); continue; }
    const [, kind, id, wc, tail] = m;
    const o = (kind === 'c' ? sys.computes.find((c) => c.id === id) : sys.fixes.find((f) => f.id === id)) as
      { arrayFlag: boolean; sizeArrayCols: number; peratomFlag: boolean; sizePeratomCols: number; localFlag?: boolean; sizeLocalCols?: number } | undefined;
    let nmax = 0;
    if (o) {
      if (o.arrayFlag) nmax = o.sizeArrayCols;
      else if (o.peratomFlag && o.sizePeratomCols) nmax = o.sizePeratomCols;
      else if (o.localFlag && o.sizeLocalCols) nmax = o.sizeLocalCols;
    }
    if (!nmax) { out.push(w); continue; }
    const [lo, hi] = bounds(wc, 1, nmax);
    for (let k = lo; k <= hi; k++) out.push(`${kind}_${id}[${k}]${tail}`);
  }
  return out;
};
