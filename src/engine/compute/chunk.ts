import { Compute } from './compute';
import { StyleError } from '../force/types';
import { inumeric, logical, missingArgs, numeric } from '../force/util';
import type { System } from '../system';
import { hasMolecule, massOf } from '../atoms';
import { argInfo } from '../argInfo';
import { parseRef, peratomValues, type Ref } from '../refs';
import { gplHook } from '../gplHooks';
import type { ImageObject, ImageShading } from '../output/image_objects';

/*
 * compute chunk/atom, com/chunk and msd/chunk — implemented from the cited
 * docs.lammps.org pages. Quotations below are copied character for character
 * from plans/lammps-docs/compute_chunk_atom.rst, compute_com_chunk.rst and
 * compute_msd_chunk.rst. Behaviour the pages leave open was measured with
 * native LAMMPS (black box; probes under /tmp/haiku-chunk) or checked against
 * its source where cited.
 *
 * compute chunk/atom — docs.lammps.org/compute_chunk_atom.html:
 *   "Define a computation that calculates an integer chunk ID from 1 to
 *   Nchunk for each atom in the group."
 *   "Chunk IDs range from 1 to *Nchunk* inclusive; some chunks may have no
 *   atoms assigned to them. Atoms that do not belong to any chunk are assigned
 *   a value of 0."
 * Styles: bin/1d, bin/2d, bin/3d, bin/sphere, bin/cylinder, type, molecule,
 * c_ID, c_ID[I], f_ID, f_ID[I], v_name.
 * Keywords: region, nchunk, limit, ids, compress, discard,
 * bound, pbc ("The *pbc* keyword only applies to the *bin/sphere* and
 * *bin/cylinder* styles."), units.
 *
 * Measured with native LAMMPS (black box): with box 0..10 and bin/1d delta 3,
 * origin lower gives 4 bins [0,3) .. [9,12); origin center (5) gives bins
 * [-1,2) .. [8,11); origin upper (10) gives bins [-2,1) .. [7,10). Bins
 * extend in both directions from the origin and are kept when they intersect
 * the bounds. An atom exactly on an edge belongs to the bin above it (x = 3.0
 * with delta 3 from 0 is bin 2). bin/2d and bin/3d number the last dimension
 * fastest (the last dimension varies fastest). Density values
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

/*
 * bin/sphere and bin/cylinder — docs.lammps.org/compute_chunk_atom.html:
 *   "*bin/sphere* args = xorig yorig zorig rmin rmax nsbin"
 *   "*bin/cylinder* args = dim origin delta c1 c2 rmin rmax ncbin"
 *   "The *bin/sphere* style defines a set of spherical shell bins around
 *   the origin (\ *xorig*,\ *yorig*,\ *zorig*\ ), using *nsbin* bins with radii
 *   equally spaced between *srmin* and *srmax*\ ."
 *   "For *bin/cylinder*, the numbering varies faster in the dimension
 *   along the cylinder axis and slower in the radial direction."
 *   "If the distance of the atom from the origin is less than *rmin*, it will
 *   be assigned to the first bin.  If the distance of the atom from the origin
 *   is greater than *rmax*, it will be assigned to the last bin."
 *   "Note that for the *bin/sphere* style, the radii *srmin* and *srmax* are
 *   scaled by the lattice spacing or reduced value of the *x* dimension."
 *   "Note that for the *bin/cylinder* style, the radii *crmin* and *crmax*
 *   are scaled by the lattice spacing or reduced value of the first
 *   dimension perpendicular to the cylinder axis"
 * As native LAMMPS does (src/compute_chunk_atom.cpp at b332f28): a reduced
 * centre is taken to box coordinates (x = h lamda + lo) and the reduced radii
 * are multiplied by the box length of that first radial dimension; an atom's
 * coordinates are wrapped into the periodic box before its distance is taken
 * (repeatedly for a sphere, once for a cylinder), and pbc yes folds each
 * periodic component of the distance in the same way; bin = trunc((r - rmin)
 * nbin / (rmax - rmin)), one lower below rmin; the coordinate of a radial bin
 * is the midpoint of its radii, in box units; the density volume of a sphere
 * shell is 4/3 pi (rhi^3 - rlo^3), and of a cylinder bin pi (rhi^2 - rlo^2)
 * times the axial bin width. With pbc yes the sphere's rmax must be at most
 * half of each periodic box length ("Compute chunk/atom bin/sphere radius is
 * too large for periodic box"). Measured with native LAMMPS: bin/cylinder with
 * pbc yes and an outer radius above half the box runs without that error.
 */
interface RadialSpec {
  shape: 'sphere' | 'cylinder';
  /** Sphere centre, or the cylinder axis position in its radial dimensions (the axis entry is 0). */
  origin: [number, number, number];
  rmin: number;
  rmax: number;
  nbin: number;
  /** Dimensions the distance is taken in: x y z for a sphere, the two others than the axis for a cylinder. */
  rdims: number[];
}

/** Radial bins in box units, set up from the box at chunk setup. */
interface RadialGeom {
  origin: number[];
  rmin: number;
  rmax: number;
  inv: number;
}


/** limit / compress, applied in keyword order after the raw chunk IDs are known. */
type PostOp = { op: 'limit'; mode: 'max' | 'exact'; nc: number } | { op: 'compress' };

/** dim origin delta of a bin style, as native readdim reads them (src/compute_chunk_atom.cpp at b332f28). */
const parseBinTriple = (w: string[], k: number, dimension: number): { spec: BinSpec; next: number } => {
  if (w.length < k + 3) throw missingArgs('compute chunk/atom');
  const dim = DIMS[w[k]];
  if (dim === undefined) throw new StyleError(`Illegal compute chunk/atom dimension ${w[k]}`);
  if (dim === 2 && dimension === 2) throw new StyleError('Cannot use compute chunk/atom bin z for 2d model');
  const o = w[k + 1];
  const origin: Origin = o === 'lower' || o === 'center' || o === 'upper' ? o : numeric(o);
  const delta = numeric(w[k + 2]);
  return { spec: { dim, origin, delta, lo: 'lower', hi: 'upper' }, next: k + 3 };
};

/** Chunk-ID source: compute chunk/atom. */
export class ComputeChunkAtom extends Compute {
  readonly style = 'chunk/atom';
  readonly kind: 'bin' | 'type' | 'molecule' | 'value';
  private readonly bins: BinSpec[];
  /** bin/sphere and bin/cylinder (a cylinder's axis bins are bins[0]). */
  private readonly radial: RadialSpec | null;
  private readonly pbc: boolean;
  private radGeo: RadialGeom | null = null;
  private readonly valueRef: Ref | null;
  private readonly regionId: string | null;
  private readonly postOps: PostOp[];
  private readonly idsMode: 'once' | 'nfreq' | 'every';
  /** compress yes (native cchunk->compress): fix ave/chunk counts a column of original chunk IDs. */
  readonly compress: boolean;
  private readonly discard: Discard;
  private readonly units: Units;
  /** nchunk held constant (nchunk once by default for type and molecule, or fix ave/chunk). */
  private held: boolean;
  private heldN: number | null = null;
  private heldKmin: number[] | null = null;
  private heldKmax: number[] | null = null;
  /** The fix holding Nchunk constant for a window of steps (lock()), the window's first step and its last (-1: no end). */
  private lockOwner: object | null = null;
  private lockStart = 0;
  private lockStop = 0;
  /**
   * Native's Nchunk is set up at the start of a run only with nchunk once; with nchunk every it is set up when the
   * compute is first used. A lock taken before that holds Nchunk = 0 (no chunks, every chunk ID 0) until unlocked.
   */
  private readonly setupOnUse: boolean;
  private lockEmpty = false;
  /** ids nfreq: the step on which the window's IDs were last kept. */
  private restoredStep = -1;
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
    // As native LAMMPS does (src/compute_chunk_atom.cpp at b332f28): the style and its words, the keywords, then the
    // checks of the values
    if (!args.length) throw missingArgs('compute chunk/atom');
    const style = args[0];
    let k = 1;
    let discard: Discard = 'yes';
    let units: Units = 'lattice';
    let idsMode: 'once' | 'nfreq' | 'every' = 'every';
    let compress = false;
    let nchunkMode: 'once' | 'every' | null = null;
    const ops: PostOp[] = [];
    const bins: BinSpec[] = [];
    const bounds = new Map<number, { lo: 'lower' | number; hi: 'upper' | number }>();
    let regionId: string | null = null;
    let valueRef: Ref | null = null;
    let radial: RadialSpec | null = null;
    let pbc = false;
    let kind: 'bin' | 'type' | 'molecule' | 'value';

    if (style === 'bin/1d' || style === 'bin/2d' || style === 'bin/3d') {
      kind = 'bin';
      const nd = style === 'bin/1d' ? 1 : style === 'bin/2d' ? 2 : 3;
      for (let d = 0; d < nd; d++) {
        const { spec, next } = parseBinTriple(args, k, sys.dimension);
        bins.push(spec);
        k = next;
      }
      discard = 'mixed';
    } else if (style === 'bin/sphere') {
      kind = 'bin';
      if (k + 6 > args.length) throw missingArgs('compute chunk/atom bin/sphere');
      const o = [0, 1, 2].map((d) => numeric(args[k + d]));
      radial = {
        shape: 'sphere', origin: [o[0], o[1], o[2]],
        rmin: numeric(args[k + 3]),
        rmax: numeric(args[k + 4]),
        nbin: inumeric(args[k + 5]),
        rdims: [0, 1, 2],
      };
      k += 6;
      discard = 'mixed';
    } else if (style === 'bin/cylinder') {
      kind = 'bin';
      const { spec, next } = parseBinTriple(args, k, sys.dimension);
      bins.push(spec);
      k = next;
      if (k + 5 > args.length) throw missingArgs('compute chunk/atom bin/cylinder');
      const rdims = [0, 1, 2].filter((d) => d !== spec.dim);
      const origin: [number, number, number] = [0, 0, 0];
      origin[rdims[0]] = numeric(args[k]);
      origin[rdims[1]] = numeric(args[k + 1]);
      radial = {
        shape: 'cylinder', origin,
        rmin: numeric(args[k + 2]),
        rmax: numeric(args[k + 3]),
        nbin: inumeric(args[k + 4]),
        rdims,
      };
      k += 5;
      discard = 'mixed';
    } else if (style === 'type') {
      kind = 'type';
    } else if (style === 'molecule') {
      kind = 'molecule';
    } else {
      const ai = argInfo(style);
      if (ai.which === 'none' || ai.which === 'unknown' || ai.dim > 1) throw new StyleError(`Invalid compute chunk/atom argument ${style}`);
      kind = 'value';
      valueRef = parseRef(style);
    }
    let discardSet = false;

    while (k < args.length) {
      const key = args[k];
      const need = (n: number): string[] => {
        if (k + n >= args.length) throw missingArgs(`compute chunk/atom ${key}`);
        return args.slice(k + 1, k + 1 + n);
      };
      switch (key) {
        case 'region':
          regionId = need(1)[0];
          if (!sys.regions.has(regionId)) throw new StyleError(`Region ${regionId} for compute chunk/atom does not exist`);
          k += 2;
          break;
        case 'nchunk': {
          const v = need(1)[0];
          if (v !== 'once' && v !== 'every') throw new StyleError(`Unknown compute chunk/atom nchunk argument ${v}`);
          nchunkMode = v;
          k += 2;
          break;
        }
        case 'limit': {
          const nc = inumeric(need(1)[0]);
          if (nc < 0) throw new StyleError(`Illegal compute chunk/atom limit value ${nc}`);
          k += 2;
          if (nc === 0) break;
          if (k >= args.length) throw missingArgs('compute chunk/atom limit');
          const mode = args[k];
          if (mode !== 'max' && mode !== 'exact') throw new StyleError(`Unknown compute chunk/atom limit keyword ${mode}`);
          ops.push({ op: 'limit', mode, nc });
          k += 1;
          break;
        }
        case 'ids': {
          const v = need(1)[0];
          if (v !== 'once' && v !== 'nfreq' && v !== 'every') throw new StyleError(`Unknown compute chunk/atom ids argument ${v}`);
          idsMode = v;
          k += 2;
          break;
        }
        case 'compress': {
          compress = logical(need(1)[0]);
          if (compress) ops.push({ op: 'compress' });
          k += 2;
          break;
        }
        case 'discard': {
          const v = need(1)[0];
          if (v !== 'yes' && v !== 'no' && v !== 'mixed') throw new StyleError(`Unknown compute chunk/atom discard argument ${v}`);
          discard = v;
          discardSet = true;
          k += 2;
          break;
        }
        case 'bound': {
          const [dw, lo, hi] = need(3);
          const d = DIMS[dw];
          if (d === undefined) throw new StyleError(`Unknown compute chunk/atom bound argument ${dw}`);
          bounds.set(d, {
            lo: lo === 'lower' ? 'lower' : numeric(lo),
            hi: hi === 'upper' ? 'upper' : numeric(hi),
          });
          k += 4;
          break;
        }
        case 'pbc': {
          pbc = logical(need(1)[0]);
          k += 2;
          break;
        }
        case 'units': {
          const v = need(1)[0];
          if (v !== 'box' && v !== 'lattice' && v !== 'reduced') throw new StyleError(`Unknown compute chunk/atom units argument ${v}`);
          units = v;
          k += 2;
          break;
        }
        default:
          throw new StyleError(`Unknown compute chunk/atom keyword ${key}`);
      }
    }
    if (!discardSet) discard = kind! === 'bin' ? 'mixed' : 'yes';
    if (kind! === 'molecule' && !hasMolecule(sys.state)) throw new StyleError('Compute chunk/atom molecule for non-molecular system');
    if (kind! !== 'bin' && discard === 'mixed') throw new StyleError('Compute chunk/atom without bins cannot use discard mixed');
    if (kind! === 'bin' && !radial) {
      const nd = bins.length;
      const dimsDiffer = new Set(bins.map((b) => b.dim)).size === nd;
      if (bins.some((b) => b.delta <= 0) || !dimsDiffer) throw new StyleError(`Illegal compute chunk/atom ${style} command`);
    }
    if (radial?.shape === 'cylinder' && bins[0].delta <= 0) throw new StyleError('Illegal compute chunk/atom bin/cylinder command');
    // docs: "The *bound* keyword only applies to the *bin/1d*, *bin/2d*, *bin/3d* styles and to the axis dimension of
    // the *bin/cylinder* style; otherwise it is ignored."
    if (kind! === 'bin') {
      for (const [d, b] of bounds) {
        const s = bins.find((x) => x.dim === d);
        if (!s) continue;
        s.lo = b.lo;
        s.hi = b.hi;
      }
    }
    if (radial) {
      // As native LAMMPS does (src/compute_chunk_atom.cpp at b332f28), with its error texts
      if (radial.shape === 'sphere' && sys.dimension === 2 && radial.origin[2] !== 0) {
        throw new StyleError('Compute chunk/atom sphere z origin must be 0.0 for 2d');
      }
      if (radial.shape === 'cylinder' && sys.dimension === 2 && bins[0].dim !== 2) {
        throw new StyleError('Compute chunk/atom cylinder axis must be z for 2d');
      }
      if (radial.rmin < 0 || radial.rmin >= radial.rmax || radial.nbin < 1) {
        throw new StyleError(`Illegal compute chunk/atom ${style} command`);
      }
      // the lattice spacings apply once, here: the centre per dimension, the radii by x. Measured with native
      // LAMMPS: bin/cylinder takes c1, c2 and its radii as box units with units lattice (the lattice scaling
      // of src/compute_chunk_atom.cpp at b332f28 never reaches its cylinder branch); its axis bins are scaled.
      if (units === 'lattice' && radial.shape === 'sphere') {
        const sp = sys.lattice?.spacing ?? [1, 1, 1];
        for (const d of radial.rdims) radial.origin[d] *= sp[d];
        radial.rmin *= sp[radial.rdims[0]];
        radial.rmax *= sp[radial.rdims[0]];
      }
    }
    // docs: "For non-orthogonal (triclinic) simulation boxes, only the *reduced* option may be used."
    if (kind! === 'bin' && sys.state.box.triclinic && units !== 'reduced') {
      throw new StyleError('Compute chunk/atom for triclinic boxes requires units reduced');
    }
    this.bins = bins;
    this.radial = radial;
    this.pbc = pbc;
    this.valueRef = valueRef;
    this.regionId = regionId;
    this.postOps = ops;
    this.idsMode = idsMode;
    this.compress = compress;
    this.discard = discard;
    this.units = units;
    this.kind = kind!;
    // docs defaults: "nchunk = every, if compress is yes, overriding other defaults listed here";
    // once for type; for mol style if region is none; for binning style if ... units = reduced
    const defaultOnce = kind! === 'type' || (kind! === 'molecule' && regionId === null) || (kind! === 'bin' && units === 'reduced');
    this.held = nchunkMode === 'once' || (nchunkMode === null && !compress && defaultOnce);
    // nchunk every: given, or the default with compress yes, for the c_/f_/v_ styles and for molecule with a region
    // (binning styles in box or lattice units default to once in a static box and every otherwise; taken as once)
    this.setupOnUse = nchunkMode === 'every'
      || (nchunkMode === null && (compress || kind! === 'value' || (kind! === 'molecule' && regionId !== null)));
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

  /** Number of coordinates per chunk: the bin dimensions, 1 for bin/sphere, 2 for bin/cylinder (0 for the non-binning styles). */
  get binDims(): number {
    if (this.kind !== 'bin') return 0;
    return this.radial ? (this.radial.shape === 'sphere' ? 1 : 2) : this.bins.length;
  }

  /**
   * fix ave/chunk holds Nchunk constant for a window of steps (stop = -1: to the end). The docs: "if *ave* =
   * *one*, then for each span of *Nfreq* timesteps, *Nchunk* is held constant between the first timestep when
   * averaging is done (within the Nfreq-length window), and the last timestep when averaging is done"; "the time
   * windows they induce for holding *Nchunk* constant must be identical, else an error will be generated". As
   * native LAMMPS does (src/compute_chunk_atom.cpp at b332f28): the last fix to lock holds the lock, and the bins
   * and Nchunk of the window's first step are kept until it unlocks.
   */
  lock(owner: object, start: number, stop: number): void {
    if (this.lockOwner === null) {
      this.lockOwner = owner;
      this.lockStart = start;
      this.lockStop = stop;
      this.lockEmpty = !this.haveIds && this.setupOnUse;
      // the window's chunks are set up on its first step, also if they were computed earlier on this step
      this.lastEpoch = -1;
      if (!this.held) this.radGeo = null;
      return;
    }
    if (start !== this.lockStart || stop !== this.lockStop) {
      throw new StyleError('Two fix commands using same compute chunk/atom command in incompatible ways');
    }
    this.lockOwner = owner;
  }

  /** Ends the window; only the fix holding the lock can. */
  unlock(owner: object): void {
    if (owner !== this.lockOwner) return;
    this.lockOwner = null;
    this.lockEmpty = false;
    if (!this.held) {
      this.heldN = null;
      this.heldKmin = null;
      this.heldKmax = null;
    }
  }

  /** Nchunk and the bins stay as they are: nchunk once, a hold, or a fix's lock. */
  private get frozen(): boolean {
    return this.held || this.lockOwner !== null;
  }

  /**
   * Recomputes chunk IDs when the state changed (computed once for ids once). ids nfreq — the docs: "in each
   * of the *Nchunk* = constant time windows (discussed above), the chunk ID's assigned to atoms on the first
   * step of the time window will persist until the end of the time window."
   */
  ensure(): void {
    const n = this.sys.state.n;
    if (this.haveIds && this.ids.length === n) {
      if (this.idsMode === 'once' || this.lastEpoch === this.sys.epoch) return;
      // restored IDs are this step's IDs (native sets invoked_ichunk to the step), also for a use after the window
      // ends on this step
      const step = this.sys.state.step;
      if (this.idsMode === 'nfreq' && (this.restoredStep === step || (this.lockOwner !== null && step > this.lockStart))) {
        this.restoredStep = step;
        return;
      }
    }
    this.compute();
    this.lastEpoch = this.sys.epoch;
    this.haveIds = true;
  }

  /**
   * The graphics objects of dump image's "compute" keyword: the walls of the bin/1d, bin/2d and bin/3d bins (the GPL
   * port of ComputeChunkAtom::compute_image, engine-gpl/graphics/chunk_bins.ts); the other styles have none.
   */
  imageObjects(): ImageObject[] {
    if (!this.bins.length || this.radial) return [];
    const s = this.sys.state;
    const reduced = this.units === 'reduced';
    const g = this.sys.geom;
    const binlo: [number, number, number] = reduced ? [0, 0, 0] : [s.box.lo[0], s.box.lo[1], s.box.lo[2]];
    const binhi: [number, number, number] = reduced ? [1, 1, 1] : [s.box.hi[0], s.box.hi[1], s.box.hi[2]];
    const f = (d: number): number => (this.units === 'lattice' ? (this.sys.lattice?.spacing[d] ?? 1) : 1);
    const num = (v: number, d: number): number => (reduced ? v : v * f(d));
    for (const b of this.bins) {
      if (b.lo !== 'lower') binlo[b.dim] = num(b.lo, b.dim);
      if (b.hi !== 'upper') binhi[b.dim] = num(b.hi, b.dim);
    }
    const bins = this.bins.map((b) => {
      const d = b.dim;
      const origin = b.origin === 'lower' ? binlo[d] : b.origin === 'upper' ? binhi[d] : b.origin === 'center' ? 0.5 * (binlo[d] + binhi[d]) : num(b.origin, d);
      return { dim: d, origin, delta: num(b.delta, d) };
    });
    return gplHook<ImageShading>('image').chunkBinObjects({
      dimension: s.dimension === 2 ? 2 : 3,
      ndim: this.bins.length as 1 | 2 | 3,
      bins,
      binlo,
      binhi,
      reduced,
      // As native LAMMPS does (src/domain.cpp at b332f28, Domain::lamda2x): h lambda first, then boxlo
      lamda2x: (p) => [
        g.lx * p[0] + g.xy * p[1] + g.xz * p[2] + g.lo[0],
        g.ly * p[1] + g.yz * p[2] + g.lo[1],
        g.lz * p[2] + g.lo[2],
      ],
    });
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
      const delta = num(spec.delta);
      const loB = spec.lo === 'lower' ? sLo : num(spec.lo);
      const hiB = spec.hi === 'upper' ? sHi : num(spec.hi);
      // As native LAMMPS does (src/compute_chunk_atom.cpp at b332f28): lower, center and upper are taken
      // from the bin bounds, which are the box faces unless the bound keyword sets them.
      const origin = spec.origin === 'lower' ? loB : spec.origin === 'upper' ? hiB : spec.origin === 'center' ? 0.5 * (loB + hiB) : num(spec.origin);
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

  /** The radial bins in box units: set up again at each chunk setup (refresh) unless Nchunk is held. */
  private radialGeom(refresh = false): RadialGeom {
    if (this.radGeo && !(refresh && !this.frozen)) return this.radGeo;
    const r = this.radial!;
    const b = this.sys.state.box;
    let origin: number[];
    let rmin = r.rmin;
    let rmax = r.rmax;
    if (this.units === 'reduced') {
      const l = r.origin;
      const [xy, xz, yz] = b.tilt;
      const lx = b.hi[0] - b.lo[0], ly = b.hi[1] - b.lo[1], lz = b.hi[2] - b.lo[2];
      // x = h lamda + lo, summed in this order
      origin = [lx * l[0] + xy * l[1] + xz * l[2] + b.lo[0], ly * l[1] + yz * l[2] + b.lo[1], lz * l[2] + b.lo[2]];
      const L = b.hi[r.rdims[0]] - b.lo[r.rdims[0]];
      rmin *= L;
      rmax *= L;
    } else {
      origin = [...r.origin];
    }
    if (this.pbc && r.shape === 'sphere') {
      for (let d = 0; d < this.sys.dimension; d++) {
        if (b.periodic[d] && rmax > 0.5 * (b.hi[d] - b.lo[d])) {
          throw new StyleError('Compute chunk/atom bin/sphere radius is too large for periodic box');
        }
      }
    }
    this.radGeo = { origin, rmin, rmax, inv: r.nbin / (rmax - rmin) };
    return this.radGeo;
  }

  /** Inner and outer radius of radial bin i (0-based); the last bin ends at rmax exactly. */
  private radialEdges(i: number): [number, number] {
    const r = this.radial!;
    const g = this.radialGeom();
    const rlo = g.rmin + (i * (g.rmax - g.rmin)) / r.nbin;
    const rhi = i === r.nbin - 1 ? g.rmax : g.rmin + ((i + 1) * (g.rmax - g.rmin)) / r.nbin;
    return [rlo, rhi];
  }

  /** Radial bin (0-based) of atom i, or -1 when discard yes drops it. */
  private radialBin(x: Float64Array, i: number, g: RadialGeom): number {
    const r = this.radial!;
    const b = this.sys.state.box;
    const sphere = r.shape === 'sphere';
    let sum = 0;
    for (const d of r.rdims) {
      const lo = b.lo[d], hi = b.hi[d], prd = hi - lo;
      let v = x[3 * i + d];
      if (b.periodic[d]) {
        if (sphere) {
          while (v < lo) v += prd;
          while (v >= hi) v -= prd;
        } else {
          if (v < lo) v += prd;
          if (v >= hi) v -= prd;
        }
      }
      let dd = v - g.origin[d];
      if (this.pbc && b.periodic[d]) {
        const half = 0.5 * prd;
        if (sphere) while (Math.abs(dd) > half) dd += dd < 0 ? prd : -prd;
        else if (Math.abs(dd) > half) dd += dd < 0 ? prd : -prd;
      }
      sum += dd * dd;
    }
    const rad = Math.sqrt(sum);
    let ib = Math.trunc((rad - g.rmin) * g.inv);
    if (rad < g.rmin) ib--;
    if (this.discard === 'yes') return ib < 0 || ib >= r.nbin ? -1 : ib;
    return Math.min(Math.max(ib, 0), r.nbin - 1);
  }

  /** Coordinates of chunk c (1-based): bin centres in box units (box, lattice) or reduced units. */
  coordsOf(c: number): number[] {
    return this.coordsOfBin(this.origIds ? this.origIds[c - 1] : c);
  }

  /**
   * Coordinates of bin b (1-based, before compression). A radial bin's coordinate is the
   * midpoint of its radii in box units; a cylinder bin has that and its axial bin centre.
   */
  coordsOfBin(b: number): number[] {
    if (this.kind !== 'bin') return [];
    const geo = this.binGeom();
    if (this.radial) {
      const mid = (i: number) => {
        const [lo, hi] = this.radialEdges(i);
        return 0.5 * (lo + hi);
      };
      if (this.radial.shape === 'sphere') return [mid(b - 1)];
      const g = geo[0];
      const nplane = g.kmax - g.kmin + 1;
      return [mid(Math.floor((b - 1) / nplane)), g.origin + (g.kmin + ((b - 1) % nplane) + 0.5) * g.delta];
    }
    const nb = geo.map((g) => g.kmax - g.kmin + 1);
    const idx = new Array<number>(nb.length).fill(0);
    let rem = b - 1;
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

  /**
   * Density volume of chunk m (0-based): the same for all xyz bins, per bin for the radial styles.
   * As native LAMMPS does (src/fix_ave_chunk.cpp at b332f28), with compress yes the radial
   * volumes are still taken by the compressed chunk index.
   */
  chunkVolumes(): (m: number) => number {
    if (!this.radial) {
      const v = this.binVolume();
      return () => v;
    }
    if (this.radial.shape === 'sphere') {
      return (m) => {
        const [rlo, rhi] = this.radialEdges(m);
        return 4.0 / 3.0 * Math.PI * rhi * rhi * rhi - 4.0 / 3.0 * Math.PI * rlo * rlo * rlo;
      };
    }
    const g = this.binGeom()[0];
    const nplane = g.kmax - g.kmin + 1;
    const slab = (g.boxL * g.delta) / (this.units === 'reduced' ? 1 : g.boxL);
    return (m) => {
      const [rlo, rhi] = this.radialEdges(Math.floor(m / nplane));
      return (Math.PI * rhi * rhi - Math.PI * rlo * rlo) * slab;
    };
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
      if (this.frozen && this.heldKmin === null) {
        const g0 = this.binGeom();
        this.heldKmin = g0.map((g) => g.kmin);
        this.heldKmax = g0.map((g) => g.kmax);
      }
      const geo = this.binGeom();
      const nb = geo.map((g) => g.kmax - g.kmin + 1);
      const rad = this.radial;
      const rg = rad ? this.radialGeom(true) : null;
      nch = nb.reduce((a, b) => a * b, 1) * (rad ? rad.nbin : 1);
      const idx = new Array<number>(geo.length);
      const reduced = this.units === 'reduced';
      const lam = [0, 0, 0];
      for (let i = 0; i < n; i++) {
        raw[i] = 0;
        if (!sel(i)) continue;
        if (reduced) sys.geom.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
        let inside = true;
        for (let j = 0; j < geo.length && inside; j++) {
          const g = geo[j];
          let v = reduced ? lam[this.bins[j].dim] : s.x[3 * i + this.bins[j].dim];
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
        if (rad) {
          // the axial bin varies fastest: radial bin times the number of axial bins plus the axial bin
          const rb = this.radialBin(s.x, i, rg!);
          if (rb < 0) continue;
          r += rb * (rad.shape === 'sphere' ? 1 : nb[0]);
        }
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
      if (this.frozen && this.heldN !== null) nch = this.heldN;
      else if (this.frozen) this.heldN = nch;
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
    if (this.lockEmpty) {
      raw.fill(0);
      nch = 0;
      if (this.compress) origIds = [];
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
/**
 * The compute chunk/atom of a chunk compute or fix; `what` is the command and style ("compute com/chunk"), named in
 * native's message (src/compute_chunk.cpp ComputeChunk::init at b332f28).
 */
export const chunkCompute = (sys: System, id: string, what: string): ComputeChunkAtom => {
  const c = sys.computes.find((x) => x.id === id);
  if (!(c instanceof ComputeChunkAtom)) throw new StyleError(`Chunk/atom compute ${id} does not exist or is incorrect style for ${what}`);
  return c;
};

/**
 * Mass-weighted, unwrapped centre of mass of each chunk: atoms in the group
 * with chunk ID > 0 (shared by com/chunk and msd/chunk).
 */
const chunkCom = (sys: System, groupBit: number, chunk: ComputeChunkAtom, wrap = false): Float64Array => {
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
  if (wrap) {
    // Measured with native LAMMPS (black box): a chunk with no atoms keeps its zero centre, also with wrap yes.
    const p = [0, 0, 0];
    for (let c = 0; c < N; c++) {
      if (!(mass[c] > 0)) continue;
      p[0] = sums[3 * c]; p[1] = sums[3 * c + 1]; p[2] = sums[3 * c + 2];
      sys.geom.remapPoint(p);
      sums[3 * c] = p[0]; sums[3 * c + 1] = p[1]; sums[3 * c + 2] = p[2];
    }
  }
  return sums;
};

/**
 * Words after the chunk ID of com/chunk: optional wrap with a logical value.
 * docs.lammps.org/compute_com_chunk.html: "wrap center of mass back to cell if
 * argument is *yes*, *true*, *on* or *1*, otherwise do not (default)". The
 * logical words are accepted in lower case only, and a repeated keyword keeps
 * the last value (measured with native LAMMPS, black box: wrap yes wrap no
 * leaves the centre unwrapped, wrap no wrap yes wraps it; the upper-case word YES
 * is refused with a message that a boolean parameter is expected).
 */
export const parseComChunkWrap = (id: string, words: string[]): boolean => {
  let wrap = false;
  for (let k = 0; k < words.length; k++) {
    // As native LAMMPS does (src/compute_com_chunk.cpp ComputeCOMChunk::ComputeCOMChunk at b332f28): "Unknown compute com/chunk keyword {}"
    if (words[k] !== 'wrap') throw new StyleError(`Unknown compute com/chunk keyword ${words[k]}`);
    if (k + 1 >= words.length) throw new StyleError(`compute ${id} (com/chunk): wrap needs a logical value (yes, true, on, 1, no, false, off or 0)`);
    const v = words[++k];
    if (v === 'yes' || v === 'true' || v === 'on' || v === '1') wrap = true;
    else if (v === 'no' || v === 'false' || v === 'off' || v === '0') wrap = false;
    else throw new StyleError(`compute ${id} (com/chunk): expected a boolean value for wrap, got '${v}'`);
  }
  return wrap;
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
  private readonly wrap: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError(`usage: compute ${id} group-ID com/chunk chunkID [wrap yes|no]`);
    this.chunk = chunkCompute(sys, args[0], 'compute com/chunk');
    this.wrap = parseComChunkWrap(id, args.slice(1));
    this.arrayFlag = true;
    this.sizeArrayCols = 3;
  }

  protected computeArray(): void {
    const sums = chunkCom(this.sys, this.groupBit, this.chunk, this.wrap);
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
    // As native LAMMPS does (src/compute_msd_chunk.cpp ComputeMSDChunk::ComputeMSDChunk at b332f28): narg must be 4, so any extra word is refused
    if (args.length !== 1) throw new StyleError('Incorrect number of arguments for compute msd/chunk');
    this.chunk = chunkCompute(sys, args[0], 'compute msd/chunk');
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
