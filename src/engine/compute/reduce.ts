import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { expandWildcards, parseRef, peratomValues, type Ref } from '../refs';
import { ALL_GROUP_BIT } from '../atoms';

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
 *
 * Engine limitation: this engine has no computes/fixes producing local
 * quantities, so "inputs local" is a StyleError, never a silent no-op.
 */
export class ComputeReduce extends Compute {
  readonly style: string = 'reduce';
  protected readonly mode: ReduceMode;
  protected readonly inputs: Ref[];
  private readonly replaces: Array<readonly [number, number]> = [];
  private readonly regionId: string | null;

  constructor(sys: System, id: string, group: string, args: string[], regionId: string | null = null) {
    super(sys, id, group, args);
    this.regionId = regionId;
    if (regionId !== null) sys.region(regionId); // "region-ID = ID of region to use for choosing atoms"
    if (args.length < 1) {
      throw new StyleError(`usage: compute ${id} group-ID ${regionId === null ? 'reduce' : `reduce/region ${regionId}`} mode input1 input2 ... keyword args ...`);
    }
    const mode = args[0];
    if (!(REDUCE_MODES as readonly string[]).includes(mode)) {
      throw new StyleError(`unknown reduce mode '${mode}' (use ${REDUCE_MODES.join(', ')})`);
    }
    this.mode = mode as ReduceMode;
    const words: string[] = [];
    for (let k = 1; k < args.length; k++) {
      const w = args[k];
      if (w === 'replace') {
        const v1 = args[k + 1], v2 = args[k + 2];
        if (v1 === undefined || v2 === undefined) throw new StyleError(`compute ${id} (reduce): the replace keyword needs two input indices (vec1 vec2)`);
        if (this.mode !== 'min' && this.mode !== 'max') {
          throw new StyleError(`compute ${id} (reduce): the replace keyword can only be used if the mode is min or max`);
        }
        this.replaces.push([parseIndex(v1, id), parseIndex(v2, id)]);
        k += 2;
      } else if (w === 'inputs') {
        const v = args[k + 1];
        if (v === undefined) throw new StyleError(`compute ${id} (reduce): the inputs keyword needs peratom or local`);
        if (v === 'local') {
          throw new StyleError(`compute ${id} (reduce): inputs local is not supported by the browser engine (it has no local-quantity computes)`);
        }
        if (v !== 'peratom') throw new StyleError(`compute ${id} (reduce): the inputs keyword value must be peratom or local, got '${v}'`);
        k += 1;
      } else {
        words.push(w);
      }
    }
    if (!words.length) throw new StyleError(`compute ${id} (reduce): at least one input value is required`);
    this.inputs = expandWildcards(sys, words, 'peratom').map((w) => this.parseInput(w));
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
    for (const [v1, v2] of this.replaces) {
      if (v1 > n || v2 > n) throw new StyleError(`compute ${id} (reduce): replace index out of range 1..${n} (the number of input values)`);
    }
  }

  /** One input: must be a per-atom producer ("must all be of the same kind (per-atom or local)"). */
  private parseInput(w: string): Ref {
    const r = parseRef(w, true);
    if (r.kind === 'attr' || r.kind === 'v') return r;
    const obj = r.kind === 'c' ? this.sys.compute(r.id) : this.sys.fix(r.id);
    const kind = r.kind === 'c' ? 'compute' : 'fix';
    if (!obj.peratomFlag) {
      throw new StyleError(`compute ${this.id} (reduce): ${w}: ${kind} ${r.id} does not calculate per-atom values`);
    }
    if (r.index === null) {
      if (obj.sizePeratomCols !== 0) throw new StyleError(`compute ${this.id} (reduce): ${w}: it calculates a per-atom array; give a column, e.g. ${w}[1]`);
    } else if (obj.sizePeratomCols === 0) {
      throw new StyleError(`compute ${this.id} (reduce): ${w}: it calculates a per-atom vector, which has no columns`);
    } else if (r.index > obj.sizePeratomCols) {
      throw new StyleError(`compute ${this.id} (reduce): ${w}: column out of range 1..${obj.sizePeratomCols}`);
    }
    return r;
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
      case 'min': return count ? mn : 0;
      case 'minabs': return count ? mnabs : 0;
      case 'max': return count ? mx : 0;
      case 'maxabs': return count ? mxabs : 0;
      case 'ave': return count ? sum / count : 0;
      case 'sumsq': return sumsq;
      case 'avesq': return count ? sumsq / count : 0;
      case 'sumabs': return sumabs;
      case 'aveabs': return count ? sumabs / count : 0;
    }
  }

  protected computeScalar(): number {
    return this.reduceOne(peratomValues(this.sys, this.inputs[0]));
  }

  protected computeVector(): void {
    for (let k = 0; k < this.inputs.length; k++) this.vector[k] = this.reduceOne(peratomValues(this.sys, this.inputs[k]));
    this.applyReplaces();
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
    if (args.length < 1) throw new StyleError(`usage: compute ${id} group-ID reduce/region region-ID mode input1 input2 ...`);
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
      if (args[k] !== 'refresh') throw new StyleError(`unknown compute displace/atom keyword '${args[k]}'`);
      if (k + 1 >= args.length) throw new StyleError(`compute ${id} (displace/atom): the refresh keyword needs the ID of an atom-style variable`);
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

/** Parses a typeN keyword of compute coord/atom: "*" or "*n" or "m*" or "m*n", or one type. */
const parseTypeRange = (id: string, w: string, ntypes: number): readonly [number, number] => {
  const m = /^(\d*)\*(\d*)$/.exec(w);
  if (m) {
    const lo = m[1] === '' ? 1 : Number(m[1]);
    const hi = m[2] === '' ? ntypes : Number(m[2]);
    if (lo < 1 || hi < 1 || lo > ntypes || hi > ntypes || lo > hi) {
      throw new StyleError(`compute ${id} (coord/atom): type range '${w}' is outside 1..${ntypes}`);
    }
    return [lo, hi];
  }
  const t = Number(w);
  if (!Number.isInteger(t) || t < 1 || t > ntypes) {
    throw new StyleError(`compute ${id} (coord/atom): type '${w}' is outside 1..${ntypes}`);
  }
  return [t, t];
};

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
 * The orientorder cstyle needs compute orientorder/atom, which this engine
 * does not have: StyleError, never a silent no-op.  Neighbors are counted
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
  private counts = new Float64Array(0);
  private readonly d = [0, 0, 0];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError(`usage: compute ${id} group-ID coord/atom cutoff cutoff [group group2-ID] typeN ...`);
    const cstyle = args[0];
    if (cstyle === 'orientorder') {
      throw new StyleError(`compute ${id} (coord/atom): cstyle 'orientorder' is not supported by the browser engine (it needs compute orientorder/atom)`);
    }
    if (cstyle !== 'cutoff') throw new StyleError(`unknown compute coord/atom cstyle '${cstyle}' (use cutoff or orientorder)`);
    if (args.length < 2) throw new StyleError(`compute ${id} (coord/atom): the cutoff cstyle needs a cutoff distance`);
    const cutoff = Number(args[1]);
    if (!Number.isFinite(cutoff) || cutoff <= 0) throw new StyleError(`compute ${id} (coord/atom): cutoff '${args[1]}' must be a positive number`);
    this.cutoff = cutoff;
    const ntypes = sys.state.ntypes;
    const ranges: Array<readonly [number, number]> = [];
    for (let k = 2; k < args.length; k++) {
      if (args[k] === 'group') {
        if (ranges.length) throw new StyleError(`compute ${id} (coord/atom): the group keyword must come before the typeN values`);
        if (k + 1 >= args.length) throw new StyleError(`compute ${id} (coord/atom): the group keyword needs a group2-ID`);
        this.group2Bit = sys.groups.bit(args[k + 1]);
        k++;
        continue;
      }
      ranges.push(parseTypeRange(id, args[k], ntypes));
    }
    this.ranges = ranges.length ? ranges : ([[1, ntypes]] as const); // no typeN: "includes atoms of all types"
    if (this.ranges.length > 1) this.sizePeratomCols = this.ranges.length; // else per-atom vector
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const g = this.sys.geom;
    const n = s.n;
    const cols = this.ranges.length;
    const out = cols > 1 ? this.arrayAtom = new Float64Array(cols * n) : this.vectorAtom = new Float64Array(n);
    if (this.counts.length !== cols) this.counts = new Float64Array(cols);
    const counts = this.counts;
    const cut2 = this.cutoff * this.cutoff;
    const three = s.dimension === 3;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue; // "0.0 for atoms not in the specified compute group"
      counts.fill(0);
      const xi = s.x[3 * i], yi = s.x[3 * i + 1], zi = s.x[3 * i + 2];
      for (let j = 0; j < n; j++) {
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

const parseIndex = (w: string, id: string): number => {
  const v = Number(w);
  if (!Number.isInteger(v) || v < 1) throw new StyleError(`compute ${id} (reduce): replace index '${w}' must be a positive integer (input values are numbered from 1)`);
  return v;
};
