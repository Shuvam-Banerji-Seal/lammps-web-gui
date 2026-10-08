import { Compute } from './compute';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import type { System } from '../system';
import { massOf } from '../atoms';
import { parseRef, peratomValues, type Ref } from '../refs';

/*
 * compute chunk/atom, com/chunk and msd/chunk — implemented from the cited
 * docs.lammps.org pages only (never from LAMMPS source). Quotations below are
 * copied character for character from plans/lammps-docs/compute_chunk_atom.rst,
 * compute_com_chunk.rst and compute_msd_chunk.rst. Behaviour the pages leave
 * open was measured with native LAMMPS (black box; probes under /tmp/haiku-chunk).
 *
 * compute chunk/atom — docs.lammps.org/compute_chunk_atom.html:
 *   "Define a computation that calculates an integer chunk ID from 1 to
 *   Nchunk for each atom in the group."
 *   "Chunk IDs range from 1 to *Nchunk* inclusive; some chunks may have no
 *   atoms assigned to them. Atoms that do not belong to any chunk are assigned
 *   a value of 0."
 * Styles: bin/1d, bin/2d, bin/3d, type, molecule, c_ID, c_ID[I], f_ID,
 * f_ID[I], v_name. bin/sphere and bin/cylinder throw a StyleError.
 * Keywords: region, nchunk, limit, ids (once|every), compress, discard,
 * bound, pbc (only "no": "The *pbc* keyword only applies to the *bin/sphere*
 * and *bin/cylinder* styles."), units.
 *
 * Measured with native LAMMPS (black box): with box 0..10 and bin/1d delta 3,
 * origin lower gives 4 bins [0,3) .. [9,12); origin center (5) gives bins
 * [-1,2) .. [8,11); origin upper (10) gives bins [-2,1) .. [7,10). Bins
 * extend in both directions from the origin and are kept when they intersect
 * the bounds. An atom exactly on an edge belongs to the bin above it (x = 3.0
 * with delta 3 from 0 is bin 2). bin/2d and bin/3d number the last dimension
 * fastest ("numbering varies fastest in the last dimension"). Density values
 * use the full bin volume (delta^dims), also for the last bin.
 */

type Units = 'box' | 'lattice' | 'reduced';
type Discard = 'yes' | 'no' | 'mixed';
type Origin = 'lower' | 'center' | 'upper' | number;

interface BinSpec {
  dim: 0 | 1 | 2;
  origin: Origin;
  delta: number;
  lo: 'lower' | number;
  hi: 'upper' | number;
}

const DIMS: Record<string, 0 | 1 | 2> = { x: 0, y: 1, z: 2 };

/** Scaled-space geometry of one bin dimension, from the current box. */
interface DimGeom {
  sLo: number;
  sHi: number;
  origin: number;
  delta: number;
  kmin: number;
  kmax: number;
  periodic: boolean;
  loIsLower: boolean;
  hiIsUpper: boolean;
  boxLo: number;
  boxL: number;
}

/** limit / compress, applied in keyword order after the raw chunk IDs are known. */
type PostOp = { op: 'limit'; mode: 'max' | 'exact'; nc: number } | { op: 'compress' };

const parseBinTriple = (w: string[], k: number, what: string): { spec: BinSpec; next: number } => {
  const dim = DIMS[w[k]];
  if (dim === undefined) throw new StyleError(`${what}: dim must be x, y or z, got '${w[k]}'`);
  const o = w[k + 1];
  if (o === undefined) throw new StyleError(`${what}: missing origin`);
  const origin: Origin = o === 'lower' || o === 'center' || o === 'upper' ? o : parseNum(o, `${what}: origin`);
  const delta = parseNum(w[k + 2], `${what}: delta`);
  if (!(delta > 0)) throw new StyleError(`${what}: delta must be > 0, got ${delta}`);
  return { spec: { dim, origin, delta, lo: 'lower', hi: 'upper' }, next: k + 3 };
};

/** Chunk-ID source: compute chunk/atom. */
export class ComputeChunkAtom extends Compute {
  readonly style = 'chunk/atom';
  readonly kind: 'bin' | 'type' | 'molecule' | 'value';
  private readonly bins: BinSpec[];
  private readonly valueRef: Ref | null;
  private readonly regionId: string | null;
  private readonly postOps: PostOp[];
  private readonly idsOnce: boolean;
  private readonly compress: boolean;
  private readonly discard: Discard;
  private readonly units: Units;
  /** nchunk held constant (nchunk once by default for type and molecule, or fix ave/chunk). */
  private held: boolean;
  private heldN: number | null = null;
  private heldKmin: number[] | null = null;
  private heldKmax: number[] | null = null;
  private haveIds = false;
  private lastEpoch = -1;

  /** Results of the last ensure(). */
  nchunk = 0;
  /** Chunk ID per atom (length = number of atoms). */
  ids = new Int32Array(0);
  /** compress yes: the pre-compression value of chunk c is origIds[c-1]. */
  origIds: Int32Array | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `compute ${id} (chunk/atom)`;
    if (!args.length) throw new StyleError(`usage: compute ${id} group-ID chunk/atom style [args] [keyword value ...]`);
    const style = args[0];
    let k = 1;
    let discard: Discard = 'yes';
    let units: Units = 'lattice';
    let idsOnce = false;
    let compress = false;
    let nchunkMode: 'once' | 'every' | null = null;
    const ops: PostOp[] = [];
    const bins: BinSpec[] = [];
    const bounds = new Map<number, { lo: 'lower' | number; hi: 'upper' | number }>();
    let regionId: string | null = null;
    let valueRef: Ref | null = null;
    let kind: 'bin' | 'type' | 'molecule' | 'value';

    if (style === 'bin/1d' || style === 'bin/2d' || style === 'bin/3d') {
      kind = 'bin';
      const nd = style === 'bin/1d' ? 1 : style === 'bin/2d' ? 2 : 3;
      for (let d = 0; d < nd; d++) {
        const { spec, next } = parseBinTriple(args, k, what);
        bins.push(spec);
        k = next;
      }
      discard = 'mixed';
    } else if (style === 'bin/sphere' || style === 'bin/cylinder') {
      throw new StyleError(`compute chunk/atom style ${style} is not supported by the engine yet`);
    } else if (style === 'type') {
      kind = 'type';
    } else if (style === 'molecule') {
      kind = 'molecule';
    } else if (/^[cfv]_/.test(style)) {
      kind = 'value';
      valueRef = parseRef(style);
    } else {
      throw new StyleError(`compute ${id}: unknown chunk/atom style '${style}'`);
    }

    while (k < args.length) {
      const key = args[k];
      const need = (n: number): string[] => {
        if (k + n >= args.length) throw new StyleError(`${what}: keyword ${key} needs ${n} value(s)`);
        return args.slice(k + 1, k + 1 + n);
      };
      switch (key) {
        case 'region':
          regionId = need(1)[0];
          k += 2;
          break;
        case 'nchunk': {
          const v = need(1)[0];
          if (v !== 'once' && v !== 'every') throw new StyleError(`${what}: nchunk must be once or every, got '${v}'`);
          nchunkMode = v;
          k += 2;
          break;
        }
        case 'limit': {
          if (args[k + 1] === '0') { k += 2; break; }
          const [v, mode] = need(2);
          const nc = parseNum(v, `${what}: limit Nc`);
          if (!Number.isInteger(nc) || nc < 1) throw new StyleError(`${what}: limit Nc must be a positive integer, got '${v}'`);
          if (mode !== 'max' && mode !== 'exact') throw new StyleError(`${what}: limit needs 'max' or 'exact' after Nc, got '${mode}'`);
          ops.push({ op: 'limit', mode, nc });
          k += 3;
          break;
        }
        case 'ids': {
          const v = need(1)[0];
          if (v === 'nfreq') throw new StyleError(`${what}: ids nfreq is not supported by the engine yet`);
          if (v !== 'once' && v !== 'every') throw new StyleError(`${what}: ids must be once, nfreq or every, got '${v}'`);
          idsOnce = v === 'once';
          k += 2;
          break;
        }
        case 'compress': {
          const v = need(1)[0];
          if (v !== 'yes' && v !== 'no') throw new StyleError(`${what}: compress must be yes or no`);
          compress = v === 'yes';
          if (compress) ops.push({ op: 'compress' });
          k += 2;
          break;
        }
        case 'discard': {
          const v = need(1)[0];
          if (v !== 'yes' && v !== 'no' && v !== 'mixed') throw new StyleError(`${what}: discard must be yes, no or mixed, got '${v}'`);
          if (kind! !== 'bin' && v === 'mixed') throw new StyleError(`${what}: discard mixed is only for the binning styles`);
          discard = v;
          k += 2;
          break;
        }
        case 'bound': {
          const [dw, lo, hi] = need(3);
          const d = DIMS[dw];
          if (d === undefined) throw new StyleError(`${what}: bound dimension must be x, y or z, got '${dw}'`);
          bounds.set(d, {
            lo: lo === 'lower' ? 'lower' : parseNum(lo, `${what}: bound lo`),
            hi: hi === 'upper' ? 'upper' : parseNum(hi, `${what}: bound hi`),
          });
          k += 4;
          break;
        }
        case 'pbc': {
          const v = need(1)[0];
          if (v === 'yes') throw new StyleError(`${what}: pbc yes applies only to bin/sphere and bin/cylinder, which are not supported`);
          if (v !== 'no') throw new StyleError(`${what}: pbc must be yes or no, got '${v}'`);
          k += 2;
          break;
        }
        case 'units': {
          const v = need(1)[0];
          if (v !== 'box' && v !== 'lattice' && v !== 'reduced') throw new StyleError(`${what}: units must be box, lattice or reduced, got '${v}'`);
          units = v;
          k += 2;
          break;
        }
        default:
          throw new StyleError(`${what}: unknown keyword '${key}'`);
      }
    }
    if (kind! === 'bin') {
      for (const [d, b] of bounds) {
        const s = bins.find((x) => x.dim === d);
        if (!s) throw new StyleError(`${what}: bound given for a dimension with no bins`);
        s.lo = b.lo;
        s.hi = b.hi;
      }
    } else if (bounds.size) {
      throw new StyleError(`${what}: the bound keyword only applies to the binning styles`);
    }
    this.bins = bins;
    this.valueRef = valueRef;
    this.regionId = regionId;
    this.postOps = ops;
    this.idsOnce = idsOnce;
    this.compress = compress;
    this.discard = discard;
    this.units = units;
    this.kind = kind!;
    // docs defaults: nchunk = once for type; for mol style if region is none
    const defaultOnce = kind! === 'type' || (kind! === 'molecule' && regionId === null);
    this.held = nchunkMode === 'once' || (nchunkMode === null && defaultOnce);
    this.scalarFlag = true;
    this.peratomFlag = true;
  }

  /**
   * Used by fix ave/chunk: hold Nchunk constant. The docs: "If *ave* =
   * *running* or *window*, then *Nchunk* is held constant forever".
   */
  holdNchunk(): void {
    this.held = true;
    this.lastEpoch = -1;
  }

  /** Number of bin dimensions (0 for the non-binning styles). */
  get binDims(): number {
    return this.kind === 'bin' ? this.bins.length : 0;
  }

  /** Recomputes chunk IDs when the state changed (computed once for ids once). */
  ensure(): void {
    const n = this.sys.state.n;
    if (this.haveIds && this.ids.length === n) {
      if (this.idsOnce || this.lastEpoch === this.sys.epoch) return;
    }
    this.compute();
    this.lastEpoch = this.sys.epoch;
    this.haveIds = true;
  }

  /** Per-dimension geometry of the bins from the current box; kmin/kmax are fixed once held. */
  private binGeom(): DimGeom[] {
    const s = this.sys.state;
    const reduced = this.units === 'reduced';
    return this.bins.map((spec, j) => {
      const d = spec.dim;
      const boxLo = s.box.lo[d];
      const boxL = s.box.hi[d] - boxLo;
      const sLo = reduced ? 0 : boxLo;
      const sHi = reduced ? 1 : s.box.hi[d];
      const f = this.units === 'lattice' ? (this.sys.lattice?.spacing[d] ?? 1) : 1;
      const num = (v: number) => (reduced ? v : v * f);
      const origin = spec.origin === 'lower' ? sLo : spec.origin === 'upper' ? sHi : spec.origin === 'center' ? (sLo + sHi) / 2 : num(spec.origin);
      const delta = num(spec.delta);
      const loB = spec.lo === 'lower' ? sLo : num(spec.lo);
      const hiB = spec.hi === 'upper' ? sHi : num(spec.hi);
      const kmin = this.heldKmin ? this.heldKmin[j] : Math.floor((loB - origin) / delta);
      const kmax = this.heldKmax ? this.heldKmax[j] : Math.ceil((hiB - origin) / delta) - 1;
      if (kmax < kmin) throw new StyleError(`compute ${this.id} (chunk/atom): no bins in dimension ${'xyz'[d]}`);
      return {
        sLo, sHi, origin, delta, kmin, kmax,
        periodic: s.box.periodic[d],
        loIsLower: spec.lo === 'lower', hiIsUpper: spec.hi === 'upper',
        boxLo, boxL,
      };
    });
  }

  /** Coordinates of chunk c (1-based): bin centres in box units (box, lattice) or reduced units. */
  coordsOf(c: number): number[] {
    if (this.kind !== 'bin') return [];
    const geo = this.binGeom();
    const nb = geo.map((g) => g.kmax - g.kmin + 1);
    const idx = new Array<number>(nb.length).fill(0);
    let rem = (this.origIds ? this.origIds[c - 1] : c) - 1;
    for (let d = nb.length - 1; d >= 0; d--) {
      idx[d] = rem % nb[d];
      rem = Math.floor(rem / nb[d]);
    }
    return geo.map((g, d) => g.origin + (g.kmin + idx[d] + 0.5) * g.delta);
  }

  /** Original (pre-compression) chunk value of chunk c, or c itself without compress. */
  origOf(c: number): number {
    return this.origIds ? this.origIds[c - 1] : c;
  }

  /** Volume of one bin for density values: bin widths times the box in the other dimensions. */
  binVolume(): number {
    const s = this.sys.state;
    const geo = this.binGeom();
    let v = 1;
    for (let d = 0; d < this.sys.dimension; d++) {
      const j = this.bins.findIndex((b) => b.dim === d);
      // a dimension without bins spans the whole box (measured: bin/1d x with delta 3 in a 10x10 box gives 300)
      if (j < 0) v *= s.box.hi[d] - s.box.lo[d];
      else v *= this.units === 'reduced' ? geo[j].delta * geo[j].boxL : geo[j].delta;
    }
    return v;
  }

  private compute(): void {
    const sys = this.sys;
    const s = sys.state;
    const n = s.n;
    const region = this.regionId === null ? null : sys.region(this.regionId);
    const sel = (i: number): boolean =>
      (s.mask[i] & this.groupBit) !== 0 &&
      (region === null || region.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]));
    const raw = new Int32Array(n);
    let nch: number;

    if (this.kind === 'bin') {
      if (this.held && this.heldKmin === null) {
        const g0 = this.binGeom();
        this.heldKmin = g0.map((g) => g.kmin);
        this.heldKmax = g0.map((g) => g.kmax);
      }
      const geo = this.binGeom();
      const nb = geo.map((g) => g.kmax - g.kmin + 1);
      nch = nb.reduce((a, b) => a * b, 1);
      const idx = new Array<number>(geo.length);
      const reduced = this.units === 'reduced';
      for (let i = 0; i < n; i++) {
        raw[i] = 0;
        if (!sel(i)) continue;
        let inside = true;
        for (let j = 0; j < geo.length && inside; j++) {
          const g = geo[j];
          let v = s.x[3 * i + this.bins[j].dim];
          if (reduced) v = (v - g.boxLo) / g.boxL;
          if (g.periodic) {
            const m = g.sHi - g.sLo;
            v = g.sLo + ((((v - g.sLo) % m) + m) % m);
          }
          let k = Math.floor((v - g.origin) / g.delta);
          if (k < g.kmin || k > g.kmax) {
            const low = k < g.kmin;
            // discard yes: 0; no: first/last bin; mixed: first/last only if the bins reach the box boundary
            if (this.discard === 'yes') inside = false;
            else if (this.discard === 'no' || (low ? g.loIsLower : g.hiIsUpper)) k = low ? g.kmin : g.kmax;
            else inside = false;
          }
          idx[j] = k - g.kmin;
        }
        if (!inside) continue;
        let r = 0;
        for (let j = 0; j < geo.length; j++) r = r * nb[j] + idx[j];
        raw[i] = r + 1;
      }
    } else if (this.kind === 'type') {
      nch = s.ntypes;
      for (let i = 0; i < n; i++) raw[i] = sel(i) ? s.type[i] : 0;
    } else if (this.kind === 'molecule') {
      nch = 0;
      for (let i = 0; i < n; i++) {
        raw[i] = sel(i) ? s.molecule[i] : 0;
        if (raw[i] > nch) nch = raw[i];
      }
    } else {
      const vals = peratomValues(sys, this.valueRef!);
      nch = 0;
      for (let i = 0; i < n; i++) {
        raw[i] = sel(i) && vals[i] >= 1 ? Math.trunc(vals[i]) : 0;
        if (raw[i] > nch) nch = raw[i];
      }
    }

    // Nchunk held constant: the first value is kept (bins: kmin/kmax above)
    if (this.kind !== 'bin') {
      if (this.held && this.heldN !== null) nch = this.heldN;
      else if (this.held) this.heldN = nch;
    }

    // IDs above Nchunk: discard yes/mixed -> 0, discard no -> Nchunk
    const dropAbove = (lim: number) => {
      for (let i = 0; i < n; i++) if (raw[i] > lim) raw[i] = this.discard === 'no' ? lim : 0;
    };
    if (this.kind !== 'bin') dropAbove(nch);

    let origIds: number[] | null = null;
    let exactNc: number | null = null;
    for (const op of this.postOps) {
      if (op.op === 'limit') {
        // docs: the binning styles ignore the limit keyword
        if (this.kind === 'bin') continue;
        const newN = op.mode === 'max' ? Math.min(nch, op.nc) : op.nc;
        if (newN < nch) dropAbove(newN);
        nch = newN;
        if (op.mode === 'exact') exactNc = op.nc;
      } else {
        let maxRaw = 0;
        for (let i = 0; i < n; i++) if (raw[i] > maxRaw) maxRaw = raw[i];
        const map = new Int32Array(maxRaw + 1);
        for (let i = 0; i < n; i++) if (raw[i] > 0) map[raw[i]] = 1;
        const list: number[] = [];
        for (let v = 1; v <= maxRaw; v++) if (map[v]) { list.push(v); map[v] = list.length; }
        for (let i = 0; i < n; i++) if (raw[i] > 0) raw[i] = map[raw[i]];
        origIds = list;
        nch = list.length;
        // docs: an exact limit applied before compress still sets Nchunk to Nc
        if (exactNc !== null) nch = exactNc;
      }
    }
    if (this.compress && origIds === null) origIds = [];
    this.ids = raw;
    this.nchunk = nch;
    this.scalar = nch;
    this.origIds = origIds === null ? null : Int32Array.from(origIds);
  }

  protected computeScalar(): number {
    this.ensure();
    return this.nchunk;
  }

  protected computePeratom(): void {
    this.ensure();
    const n = this.sys.state.n;
    if (this.vectorAtom.length !== n) this.vectorAtom = new Float64Array(n);
    for (let i = 0; i < n; i++) this.vectorAtom[i] = this.ids[i];
  }
}

/** Looks up a compute chunk/atom by ID (per-chunk computes and fix ave/chunk). */
export const chunkCompute = (sys: System, id: string, what: string): ComputeChunkAtom => {
  const c = sys.compute(id);
  if (!(c instanceof ComputeChunkAtom)) throw new StyleError(`${what}: compute ${id} is not a compute chunk/atom`);
  return c;
};

/**
 * Mass-weighted, unwrapped centre of mass of each chunk: atoms in the group
 * with chunk ID > 0 (shared by com/chunk and msd/chunk).
 */
const chunkCom = (sys: System, groupBit: number, chunk: ComputeChunkAtom): Float64Array => {
  chunk.ensure();
  const s = sys.state;
  const N = chunk.nchunk;
  const sums = new Float64Array(N * 3);
  const mass = new Float64Array(N);
  const u = [0, 0, 0];
  for (let i = 0; i < s.n; i++) {
    const c = chunk.ids[i];
    if (c <= 0 || c > N || !(s.mask[i] & groupBit)) continue;
    const m = massOf(s, i);
    sys.geom.unwrap(s.x, s.image, i, u);
    for (let d = 0; d < 3; d++) sums[3 * (c - 1) + d] += m * u[d];
    mass[c - 1] += m;
  }
  for (let c = 0; c < N; c++) for (let d = 0; d < 3; d++) sums[3 * c + d] = mass[c] > 0 ? sums[3 * c + d] / mass[c] : 0;
  return sums;
};

/*
 * compute ID group-ID com/chunk chunkID — docs.lammps.org/compute_com_chunk.html:
 *   "Define a computation that calculates the center-of-mass for multiple
 *   chunks of atoms."
 *   "This compute calculates a global array where the number of rows = the"
 *   "number of chunks *Nchunk* as calculated by the specified ... The number of
 *   columns is 3 for the :math:`(x,y,z)` center-of-mass coordinates of each chunk."
 *   "The array values are "intensive"."
 */
export class ComputeComChunk extends Compute {
  readonly style = 'com/chunk';
  private readonly chunk: ComputeChunkAtom;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`usage: compute ${id} group-ID com/chunk chunkID`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (com/chunk)`);
    this.arrayFlag = true;
    this.sizeArrayCols = 3;
  }

  protected computeArray(): void {
    const sums = chunkCom(this.sys, this.groupBit, this.chunk);
    this.sizeArrayRows = this.chunk.nchunk;
    this.array = new Float64Array(sums);
  }
}

/*
 * compute ID group-ID msd/chunk chunkID — docs.lammps.org/compute_msd_chunk.html:
 *   "Four quantities are calculated by this compute for each chunk. The first
 *   3 quantities are the squared *dx*, *dy*, and *dz* displacements of the
 *   center-of-mass. The fourth component is the total squared displacement
 *   (i.e., :math:`dx^2 + dy^2 + dz^2`) of the center-of-mass."
 *   "The displacement of the center-of-mass of the chunk is from its original
 *   center-of-mass position, calculated on the timestep this compute command
 *   was first invoked."
 *   "If *Nchunk* does not remain constant, an error will be generated."
 */
export class ComputeMsdChunk extends Compute {
  readonly style = 'msd/chunk';
  private readonly chunk: ComputeChunkAtom;
  private ref: Float64Array | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`usage: compute ${id} group-ID msd/chunk chunkID`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (msd/chunk)`);
    this.arrayFlag = true;
    this.sizeArrayCols = 4;
  }

  protected computeArray(): void {
    const cur = chunkCom(this.sys, this.groupBit, this.chunk);
    const N = this.chunk.nchunk;
    if (this.ref === null) this.ref = cur.slice();
    else if (this.ref.length !== cur.length) {
      throw new StyleError(`compute ${this.id} (msd/chunk): the number of chunks changed from ${this.ref.length / 3} to ${N}`);
    }
    const out = new Float64Array(N * 4);
    for (let c = 0; c < N; c++) {
      let tot = 0;
      for (let d = 0; d < 3; d++) {
        const dd = cur[3 * c + d] - this.ref[3 * c + d];
        out[4 * c + d] = dd * dd;
        tot += dd * dd;
      }
      out[4 * c + 3] = tot;
    }
    this.sizeArrayRows = N;
    this.array = out;
  }
}
