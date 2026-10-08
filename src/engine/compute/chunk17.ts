import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { hasCharge, massOf } from '../atoms';
import { expandWildcards, globalVector, parseRef, peratomValues, type Ref } from '../refs';
import { ComputeChunkAtom, chunkCompute } from './chunk';
import { ComputeGyration } from './global';
import { pinvSolve3, shapeOf } from './chunk17_math';

/*
 * Wave-17 chunk and shape computes, implemented from the cited
 * docs.lammps.org pages (plans/lammps-docs/compute_<name>.rst) only. Quotations
 * are copied character for character from those pages. Behaviour the pages
 * leave open was measured with native LAMMPS (black box; probes in
 * /tmp/haiku-chunk17).
 *
 * Shared conventions (all chunk computes):
 *   "Note that only atoms in the specified group contribute to the calculation."
 *   Atoms take unwrapped coordinates from their image flags, so every
 *   centre of mass, moment and dipole below is built from unwrapped positions.
 *   Outputs are "intensive" unless a page says otherwise.
 */

/** BIG as LAMMPS uses it for empty min/max chunks; measured: min of an empty chunk prints 1e+20, max -1e+20. */
const BIG = 1e20;

/** Atoms that contribute: in the group, and (for chunk computes) with chunk ID 1..N. */
interface Members {
  /** Number of chunks (1 when the compute has no chunk). */
  N: number;
  /** Number of contributing atoms. */
  n: number;
  /** Local atom index of each contributing atom. */
  atom: Int32Array;
  /** 0-based chunk index of each contributing atom. */
  chunk: Int32Array;
  /** Mass of each contributing atom. */
  mass: Float64Array;
  /** Unwrapped coordinates, 3 per contributing atom. */
  u: Float64Array;
}

const members = (sys: System, groupBit: number, chunk: ComputeChunkAtom | null): Members => {
  const s = sys.state;
  const g = sys.geom;
  let N = 1;
  let ids: Int32Array | null = null;
  if (chunk) {
    chunk.ensure();
    N = chunk.nchunk;
    ids = chunk.ids;
  }
  const atom: number[] = [];
  const chk: number[] = [];
  for (let i = 0; i < s.n; i++) {
    if (!(s.mask[i] & groupBit)) continue;
    let c = 0;
    if (ids) {
      c = ids[i];
      if (c <= 0 || c > N) continue;
    }
    atom.push(i);
    chk.push(ids ? c - 1 : 0);
  }
  const n = atom.length;
  const out: Members = {
    N, n,
    atom: Int32Array.from(atom),
    chunk: Int32Array.from(chk),
    mass: new Float64Array(n),
    u: new Float64Array(3 * n),
  };
  const u = [0, 0, 0];
  for (let k = 0; k < n; k++) {
    const i = atom[k];
    out.mass[k] = massOf(s, i);
    g.unwrap(s.x, s.image, i, u);
    out.u[3 * k] = u[0];
    out.u[3 * k + 1] = u[1];
    out.u[3 * k + 2] = u[2];
  }
  return out;
};

/** Per-chunk mass, mass-weighted centre of mass and geometric centre (unwrapped). */
interface Centres {
  M: Float64Array;
  com: Float64Array;
  geo: Float64Array;
}

const centres = (mem: Members): Centres => {
  const { N, n } = mem;
  const M = new Float64Array(N);
  const com = new Float64Array(3 * N);
  const geo = new Float64Array(3 * N);
  const cnt = new Float64Array(N);
  for (let k = 0; k < n; k++) {
    const c = mem.chunk[k];
    const m = mem.mass[k];
    M[c] += m;
    cnt[c] += 1;
    for (let d = 0; d < 3; d++) {
      com[3 * c + d] += m * mem.u[3 * k + d];
      geo[3 * c + d] += mem.u[3 * k + d];
    }
  }
  for (let c = 0; c < N; c++) {
    for (let d = 0; d < 3; d++) {
      com[3 * c + d] = M[c] > 0 ? com[3 * c + d] / M[c] : 0;
      geo[3 * c + d] = cnt[c] > 0 ? geo[3 * c + d] / cnt[c] : 0;
    }
  }
  return { M, com, geo };
};

/** Angular momentum about each chunk's centre of mass: sum m (r - r_cm) x v. */
const angmomOf = (sys: System, mem: Members, cen: Centres): Float64Array<ArrayBuffer> => {
  const s = sys.state;
  const L = new Float64Array(3 * mem.N);
  for (let k = 0; k < mem.n; k++) {
    const c = mem.chunk[k];
    const i = mem.atom[k];
    const m = mem.mass[k];
    const dx = mem.u[3 * k] - cen.com[3 * c];
    const dy = mem.u[3 * k + 1] - cen.com[3 * c + 1];
    const dz = mem.u[3 * k + 2] - cen.com[3 * c + 2];
    const vx = s.v[3 * i], vy = s.v[3 * i + 1], vz = s.v[3 * i + 2];
    L[3 * c] += m * (dy * vz - dz * vy);
    L[3 * c + 1] += m * (dz * vx - dx * vz);
    L[3 * c + 2] += m * (dx * vy - dy * vx);
  }
  return L;
};

/**
 * Inertia tensor about each chunk's centre of mass, columns Ixx, Iyy, Izz, Ixy,
 * Iyz, Ixz (inertia/chunk ordering). Off-diagonal entries are minus the sum of
 * m dx dy and so on; measured with native LAMMPS (black box) on a 3-atom chunk:
 * Ixy = +1 where sum m x y = -1.
 */
const inertiaOf = (mem: Members, cen: Centres): Float64Array<ArrayBuffer> => {
  const I = new Float64Array(6 * mem.N);
  for (let k = 0; k < mem.n; k++) {
    const c = mem.chunk[k];
    const m = mem.mass[k];
    const dx = mem.u[3 * k] - cen.com[3 * c];
    const dy = mem.u[3 * k + 1] - cen.com[3 * c + 1];
    const dz = mem.u[3 * k + 2] - cen.com[3 * c + 2];
    I[6 * c] += m * (dy * dy + dz * dz);
    I[6 * c + 1] += m * (dx * dx + dz * dz);
    I[6 * c + 2] += m * (dx * dx + dy * dy);
    I[6 * c + 3] -= m * dx * dy;
    I[6 * c + 4] -= m * dy * dz;
    I[6 * c + 5] -= m * dx * dz;
  }
  return I;
};

/** Symmetric 3x3 (row-major) from the inertia/chunk ordering. */
const full3 = (I: Float64Array, c: number): number[] => {
  const o = 6 * c;
  return [I[o], I[o + 3], I[o + 5], I[o + 3], I[o + 1], I[o + 4], I[o + 5], I[o + 4], I[o + 2]];
};

/** Gyration tensor per chunk (xx, yy, zz, xy, xz, yz) = (1/M) sum m dr_a dr_b. */
const gyrationTensorOf = (mem: Members, cen: Centres): Float64Array<ArrayBuffer> => {
  const t = new Float64Array(6 * mem.N);
  for (let k = 0; k < mem.n; k++) {
    const c = mem.chunk[k];
    const m = mem.mass[k];
    const dx = mem.u[3 * k] - cen.com[3 * c];
    const dy = mem.u[3 * k + 1] - cen.com[3 * c + 1];
    const dz = mem.u[3 * k + 2] - cen.com[3 * c + 2];
    t[6 * c] += m * dx * dx;
    t[6 * c + 1] += m * dy * dy;
    t[6 * c + 2] += m * dz * dz;
    t[6 * c + 3] += m * dx * dy;
    t[6 * c + 4] += m * dx * dz;
    t[6 * c + 5] += m * dy * dz;
  }
  for (let c = 0; c < mem.N; c++) {
    for (let j = 0; j < 6; j++) t[6 * c + j] = cen.M[c] > 0 ? t[6 * c + j] / cen.M[c] : 0;
  }
  return t;
};

/** Optional trailing "mass" or "geometry" argument of the dipole computes. */
const dipoleMode = (style: string, words: string[]): 'mass' | 'geometry' => {
  if (words.length === 0) return 'mass';
  if (words.length > 1) throw new StyleError(`compute ${style}: too many arguments (${words.join(' ')})`);
  const w = words[0];
  if (w === 'mass' || w === 'geometry') return w;
  if (w.startsWith('tip4p')) throw new StyleError(`compute ${style}: style dipole/tip4p is not supported by the engine yet`);
  throw new StyleError(`compute ${style}: unknown argument '${w}', expected mass or geometry`);
};

/** Dipole per chunk (dx, dy, dz, |d|): sum q u - Q r_ref, r_ref = centre of mass or geometric centre. */
const dipoleOf = (sys: System, mem: Members, cen: Centres, mode: 'mass' | 'geometry'): Float64Array<ArrayBuffer> => {
  const s = sys.state;
  const out = new Float64Array(4 * mem.N);
  const qsum = new Float64Array(mem.N);
  const qu = new Float64Array(3 * mem.N);
  // compute_dipole.html: "Both per-atom charges and per-atom dipole moments, if present, contribute"
  const mud = new Float64Array(3 * mem.N);
  for (let k = 0; k < mem.n; k++) {
    const c = mem.chunk[k];
    const i = mem.atom[k];
    const q = s.q[i];
    qsum[c] += q;
    for (let d = 0; d < 3; d++) qu[3 * c + d] += q * mem.u[3 * k + d];
    if (s.mu) for (let d = 0; d < 3; d++) mud[3 * c + d] += s.mu[4 * i + d];
  }
  for (let c = 0; c < mem.N; c++) {
    const ref = mode === 'mass' ? cen.com : cen.geo;
    const d = [0, 0, 0];
    for (let a = 0; a < 3; a++) d[a] = qu[3 * c + a] - qsum[c] * ref[3 * c + a] + mud[3 * c + a];
    out[4 * c] = d[0];
    out[4 * c + 1] = d[1];
    out[4 * c + 2] = d[2];
    out[4 * c + 3] = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  }
  return out;
};

/*
 * compute ID group-ID angmom/chunk chunkID — docs.lammps.org/compute_angmom_chunk.html:
 *   "This compute calculates the 3 components of the angular momentum
 *   vector for each chunk, due to the velocity/momentum of the individual
 *   atoms in the chunk around the center-of-mass of the chunk."
 *   "This compute calculates a global array where the number of rows = the
 *   number of chunks"
 *   "The array values are "intensive"."
 */
export class ComputeAngmomChunk extends Compute {
  readonly style = 'angmom/chunk';
  private readonly chunk: ComputeChunkAtom;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`usage: compute ${id} group-ID angmom/chunk chunkID`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (angmom/chunk)`);
    this.arrayFlag = true;
    this.sizeArrayCols = 3;
  }

  protected computeArray(): void {
    const mem = members(this.sys, this.groupBit, this.chunk);
    this.sizeArrayRows = mem.N;
    this.array = angmomOf(this.sys, mem, centres(mem));
  }
}

/*
 * compute ID group-ID omega/chunk chunkID — docs.lammps.org/compute_omega_chunk.html:
 *   "This compute calculates the three components of the angular velocity
 *   vector for each chunk via the formula :math:`\vec L = \mathrm{I}\cdot \vec\omega`"
 * Measured with native LAMMPS (black box): a single-atom chunk and an empty
 * chunk give zero; a linear 2-atom chunk with a singular inertia tensor gives
 * the minimum-norm solution (pinvSolve3 in chunk17_math.ts).
 */
export class ComputeOmegaChunk extends Compute {
  readonly style = 'omega/chunk';
  private readonly chunk: ComputeChunkAtom;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`usage: compute ${id} group-ID omega/chunk chunkID`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (omega/chunk)`);
    this.arrayFlag = true;
    this.sizeArrayCols = 3;
  }

  protected computeArray(): void {
    const mem = members(this.sys, this.groupBit, this.chunk);
    const cen = centres(mem);
    const L = angmomOf(this.sys, mem, cen);
    const I = inertiaOf(mem, cen);
    const out = new Float64Array(3 * mem.N);
    for (let c = 0; c < mem.N; c++) {
      const w = pinvSolve3(full3(I, c), [L[3 * c], L[3 * c + 1], L[3 * c + 2]]);
      out.set(w, 3 * c);
    }
    this.sizeArrayRows = mem.N;
    this.array = out;
  }
}

/*
 * compute ID group-ID inertia/chunk chunkID — docs.lammps.org/compute_inertia_chunk.html:
 *   "This compute calculates the six components of the symmetric inertia
 *   tensor for each chunk, ordered
 *   :math:`I_{xx},I_{yy},I_{zz},I_{xy},I_{yz},I_{xz}`."
 *   The array values are "intensive". They are in mass times distance squared.
 */
export class ComputeInertiaChunk extends Compute {
  readonly style = 'inertia/chunk';
  private readonly chunk: ComputeChunkAtom;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`usage: compute ${id} group-ID inertia/chunk chunkID`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (inertia/chunk)`);
    this.arrayFlag = true;
    this.sizeArrayCols = 6;
  }

  protected computeArray(): void {
    const mem = members(this.sys, this.groupBit, this.chunk);
    this.sizeArrayRows = mem.N;
    this.array = inertiaOf(mem, centres(mem));
  }
}

/*
 * compute ID group-ID gyration/chunk chunkID keyword — docs.lammps.org/compute_gyration_chunk.html:
 *   "This compute calculates a global vector if the *tensor* keyword is not
 *   specified and a global array if it is."
 *   "If the *tensor* keyword is specified, then the scalar :math:`R_g` value is not
 *   calculated, but an :math:`R_g` tensor is instead calculated for each chunk."
 *   "The six components of the tensor are ordered :math:`xx`, :math:`yy`, :math:`zz`,
 *   :math:`xy`, :math:`xz`, :math:`yz`."
 *   "All the vector or array values calculated by this compute are "intensive"."
 * Measured with native LAMMPS (black box): an empty chunk gives zero.
 */
export class ComputeGyrationChunk extends Compute {
  readonly style = 'gyration/chunk';
  readonly tensor: boolean;
  private readonly chunk: ComputeChunkAtom;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError(`usage: compute ${id} group-ID gyration/chunk chunkID [tensor]`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (gyration/chunk)`);
    let tensor = false;
    for (const w of args.slice(1)) {
      if (w !== 'tensor') throw new StyleError(`compute ${id} (gyration/chunk): unknown keyword '${w}'`);
      tensor = true;
    }
    this.tensor = tensor;
    if (tensor) {
      this.arrayFlag = true;
      this.sizeArrayCols = 6;
    } else {
      this.vectorFlag = true;
    }
  }

  protected computeVector(): void {
    const mem = members(this.sys, this.groupBit, this.chunk);
    const t = gyrationTensorOf(mem, centres(mem));
    const out = new Float64Array(mem.N);
    for (let c = 0; c < mem.N; c++) out[c] = Math.sqrt(t[6 * c] + t[6 * c + 1] + t[6 * c + 2]);
    this.sizeVector = mem.N;
    this.vector = out;
  }

  protected computeArray(): void {
    const mem = members(this.sys, this.groupBit, this.chunk);
    this.sizeArrayRows = mem.N;
    this.array = gyrationTensorOf(mem, centres(mem));
  }
}

/*
 * compute ID group-ID gyration/shape compute-ID — docs.lammps.org/compute_gyration_shape.html:
 *   "Define a computation that calculates the eigenvalues of the gyration tensor of a
 *   group of atoms and three shape parameters."
 *   "This compute calculates a global vector of length 6, which can be accessed by
 *   indices 1--6. The first three values are the eigenvalues of the gyration tensor
 *   followed by the asphericity, the acylindricity and the relative shape anisotropy."
 * Measured with native LAMMPS (black box): the eigenvalues come out in descending
 * order, so b = l_1 - (l_2 + l_3) / 2 and c = l_2 - l_3 with l_1 the largest.
 */
export class ComputeGyrationShape extends Compute {
  readonly style = 'gyration/shape';
  private readonly source: ComputeGyration;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`usage: compute ${id} group-ID gyration/shape compute-ID`);
    const c = sys.compute(args[0]);
    if (!(c instanceof ComputeGyration)) throw new StyleError(`compute ${id} (gyration/shape): compute ${args[0]} is not a compute gyration`);
    this.source = c;
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.vector = new Float64Array(6);
  }

  protected computeVector(): void {
    this.vector = Float64Array.from(shapeOf(this.source.vectorValues()));
  }
}

/*
 * compute ID group-ID gyration/shape/chunk compute-ID — docs.lammps.org/compute_gyration_shape_chunk.html:
 *   "The tensor keyword must be specified in the compute gyration/chunk command."
 *   "This compute calculates a global array with six columns, which can be accessed by
 *   indices 1--6. The first three columns are the eigenvalues of the gyration tensor
 *   followed by the asphericity, the acylindricity and the relative shape anisotropy."
 * Measured with native LAMMPS (black box): a chunk with zero eigenvalues (empty or
 * single-atom chunk) gives zero for the eigenvalues and a not-a-number k.
 */
export class ComputeGyrationShapeChunk extends Compute {
  readonly style = 'gyration/shape/chunk';
  private readonly source: ComputeGyrationChunk;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`usage: compute ${id} group-ID gyration/shape/chunk compute-ID`);
    const c = sys.compute(args[0]);
    if (!(c instanceof ComputeGyrationChunk)) throw new StyleError(`compute ${id} (gyration/shape/chunk): compute ${args[0]} is not a compute gyration/chunk`);
    if (!c.tensor) throw new StyleError(`compute ${id} (gyration/shape/chunk): the tensor keyword must be specified in compute ${args[0]}`);
    this.source = c;
    this.arrayFlag = true;
    this.sizeArrayCols = 6;
  }

  protected computeArray(): void {
    const t = this.source.arrayValues();
    const N = this.source.sizeArrayRows;
    const out = new Float64Array(6 * N);
    for (let c = 0; c < N; c++) out.set(shapeOf(t.subarray(6 * c, 6 * c + 6)), 6 * c);
    this.sizeArrayRows = N;
    this.array = out;
  }
}

/*
 * compute ID group-ID momentum — docs.lammps.org/compute_momentum.html:
 *   "It is computed as the sum :math:`\vec{p} = \sum_i m_i \cdot \vec{v}_i`
 *   over all particles in the compute group, where *m* and *v* are
 *   the mass and velocity vector of the particle, respectively."
 *   "The vector value calculated by this compute is "extensive"."
 */
export class ComputeMomentum extends Compute {
  readonly style = 'momentum';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`compute momentum takes no arguments (got ${args.join(' ')})`);
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.extvector = 1;
    this.vector = new Float64Array(3);
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const p = new Float64Array(3);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      for (let d = 0; d < 3; d++) p[d] += m * s.v[3 * i + d];
    }
    this.vector = p;
  }
}

/*
 * compute ID group-ID dipole [mass|geometry] — docs.lammps.org/compute_dipole.html:
 *   "For a group with a net charge the resulting dipole is made position
 *   independent by subtracting the position vector of the center of mass or
 *   geometric center times the net charge from the computed dipole vector."
 *   "Using the center of mass is the default setting for the net charge correction."
 * Measured with native LAMMPS (black box): with a net charge of 0.6 on five
 * atoms, mass and geometry give the vectors (-1.7428571, -1.0285714, -1.2214286)
 * and (-1.64, -0.84, -1.11), the geometric centre being the unweighted mean.
 * The scalar is the magnitude of the vector.
 * dipole/tip4p is not supported by the engine (it needs the pair style's M-site parameters).
 */
export class ComputeDipole extends Compute {
  readonly style = 'dipole';
  private readonly mode: 'mass' | 'geometry';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.mode = dipoleMode('dipole', args);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 3;
    this.vector = new Float64Array(3);
  }

  protected computeVector(): void {
    if (!hasCharge(this.sys.state) && !this.sys.state.mu) throw new StyleError(`compute ${this.id} (dipole) requires atom charges or point dipoles`);
    const mem = members(this.sys, this.groupBit, null);
    const d = dipoleOf(this.sys, mem, centres(mem), this.mode);
    this.vector = d.slice(0, 3);
  }

  protected computeScalar(): number {
    const v = this.vectorValues();
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }
}

/*
 * compute ID group-ID dipole/chunk chunkID [mass|geometry] — docs.lammps.org/compute_dipole_chunk.html:
 *   "These computes calculate the :math:`(x,y,z)` coordinates of the dipole
 *   vector and the total dipole moment for each chunk, which includes all
 *   effects due to atoms passing through periodic boundaries."
 *   "The number of columns is 4 for the :math:`(x,y,z)` dipole vector components and the total dipole of each chunk."
 */
export class ComputeDipoleChunk extends Compute {
  readonly style = 'dipole/chunk';
  private readonly chunk: ComputeChunkAtom;
  private readonly mode: 'mass' | 'geometry';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError(`usage: compute ${id} group-ID dipole/chunk chunkID [mass|geometry]`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (dipole/chunk)`);
    this.mode = dipoleMode('dipole/chunk', args.slice(1));
    this.arrayFlag = true;
    this.sizeArrayCols = 4;
  }

  protected computeArray(): void {
    if (!hasCharge(this.sys.state) && !this.sys.state.mu) throw new StyleError(`compute ${this.id} (dipole/chunk) requires atom charges or point dipoles`);
    const mem = members(this.sys, this.groupBit, this.chunk);
    this.sizeArrayRows = mem.N;
    this.array = dipoleOf(this.sys, mem, centres(mem), this.mode);
  }
}

/** Parses the optional leading value of a per-chunk input list (reduce/chunk and chunk/spread/atom). */
const inputRefs = (sys: System, style: string, words: string[], context: 'vector' | 'peratom'): Ref[] => {
  if (!words.length) throw new StyleError(`compute ${style}: at least one input is required`);
  return expandWildcards(sys, words, context).map((w) => parseRef(w));
};

/*
 * compute ID group-ID reduce/chunk chunkID mode input1 input2 ... — docs.lammps.org/compute_reduce_chunk.html:
 *   "Define a calculation that reduces one or more per-atom vectors into
 *   per-chunk values."
 *   "The *sum* option adds the per-atom values to a per-chunk total. The *min*
 *   or *max* options find the minimum or maximum value of the per-atom values
 *   for each chunk."
 *   "This compute calculates a global vector if a single input value is specified,
 *   otherwise a global array is output."
 *   "The vector or array values are "intensive"."
 * Measured with native LAMMPS (black box): an empty chunk gives 0 for sum,
 * 1e+20 for min and -1e+20 for max.
 */
export class ComputeReduceChunk extends Compute {
  readonly style = 'reduce/chunk';
  private readonly chunk: ComputeChunkAtom;
  private readonly mode: 'sum' | 'min' | 'max';
  private readonly refs: Ref[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError(`usage: compute ${id} group-ID reduce/chunk chunkID mode input1 input2 ...`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (reduce/chunk)`);
    const mode = args[1];
    if (mode !== 'sum' && mode !== 'min' && mode !== 'max') throw new StyleError(`compute ${id} (reduce/chunk): mode must be sum, min or max, got '${mode}'`);
    this.mode = mode;
    this.refs = inputRefs(sys, `${id} (reduce/chunk)`, args.slice(2), 'peratom');
    if (this.refs.length === 1) {
      this.vectorFlag = true;
    } else {
      this.arrayFlag = true;
      this.sizeArrayCols = this.refs.length;
    }
  }

  private reduce(): Float64Array<ArrayBuffer> {
    const mem = members(this.sys, this.groupBit, this.chunk);
    const K = this.refs.length;
    const out = new Float64Array(mem.N * K);
    const init = this.mode === 'sum' ? 0 : this.mode === 'min' ? BIG : -BIG;
    out.fill(init);
    for (let k = 0; k < K; k++) {
      const vals = peratomValues(this.sys, this.refs[k]);
      for (let j = 0; j < mem.n; j++) {
        const c = mem.chunk[j];
        const v = vals[mem.atom[j]];
        const o = K * c + k;
        if (this.mode === 'sum') out[o] += v;
        else if (this.mode === 'min') { if (v < out[o]) out[o] = v; }
        else if (v > out[o]) out[o] = v;
      }
    }
    this.sizeArrayRows = mem.N;
    return out;
  }

  protected computeVector(): void {
    const out = this.reduce();
    this.sizeVector = this.chunk.nchunk;
    this.vector = out;
  }

  protected computeArray(): void {
    this.array = this.reduce();
  }
}

/*
 * compute ID group-ID chunk/spread/atom chunkID input1 input2 ... — docs.lammps.org/compute_chunk_spread_atom.html:
 *   "Define a calculation that "spreads" one or more per-chunk values to
 *   each atom in the chunk."
 *   "The values generated by this compute will be 0.0 for atoms not in the
 *   specified compute group group-ID." They will also be 0.0 if the atom is not in
 *   a chunk, as assigned by the chunkID compute. They will also be 0.0 if the
 *   current chunk ID for the atom is out-of-bounds with respect to the number of
 *   chunks stored by a particular input compute or fix."
 *   "The output is a per-atom vector if a single input value is specified,
 *   otherwise a per-atom array is output."
 */
export class ComputeChunkSpreadAtom extends Compute {
  readonly style = 'chunk/spread/atom';
  private readonly chunk: ComputeChunkAtom;
  private readonly refs: Ref[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError(`usage: compute ${id} group-ID chunk/spread/atom chunkID input1 input2 ...`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (chunk/spread/atom)`);
    this.refs = inputRefs(sys, `${id} (chunk/spread/atom)`, args.slice(1), 'vector');
    this.peratomFlag = true;
    this.sizePeratomCols = this.refs.length === 1 ? 0 : this.refs.length;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    this.chunk.ensure();
    const K = this.refs.length;
    const vals = this.refs.map((r) => globalVector(this.sys, r));
    const out = new Float64Array(s.n * K);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const c = this.chunk.ids[i];
      if (c <= 0) continue;
      for (let k = 0; k < K; k++) {
        if (c <= vals[k].length) out[K * i + k] = vals[k][c - 1];
      }
    }
    if (K === 1) this.vectorAtom = out;
    else this.arrayAtom = out;
  }
}

/*
 * compute ID group-ID property/chunk chunkID input1 input2 ... — docs.lammps.org/compute_property_chunk.html:
 *   "The *count* attribute is the number of atoms in the chunk."
 *   "The *id* attribute stores the original chunk ID for each chunk. It can only be
 *   used if the *compress* keyword was set to *yes*"
 *   "If a single input is specified, a global vector is produced. If two or more
 *   inputs are specified, a global array is produced where the number of columns =
 *   the number of inputs."
 * Measured with native LAMMPS (black box): id on a chunk/atom with compress no is
 * an error (no IDs are stored by compute chunk/atom). coordN is not supported: the
 * bin centres are not exposed by compute chunk/atom in the engine yet.
 */
export class ComputePropertyChunk extends Compute {
  readonly style = 'property/chunk';
  private readonly chunk: ComputeChunkAtom;
  private readonly attrs: string[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError(`usage: compute ${id} group-ID property/chunk chunkID input1 input2 ...`);
    this.chunk = chunkCompute(sys, args[0], `compute ${id} (property/chunk)`);
    this.attrs = args.slice(1);
    for (const a of this.attrs) {
      if (a === 'count' || a === 'id') continue;
      if (/^coord[123]$/.test(a)) throw new StyleError(`compute ${id} (property/chunk): attribute ${a} is not supported by the engine yet`);
      throw new StyleError(`compute ${id} (property/chunk): unknown attribute '${a}', expected count, id, coord1, coord2 or coord3`);
    }
    if (this.attrs.length === 1) {
      this.vectorFlag = true;
    } else {
      this.arrayFlag = true;
      this.sizeArrayCols = this.attrs.length;
    }
  }

  private values(): Float64Array<ArrayBuffer> {
    const mem = members(this.sys, this.groupBit, this.chunk);
    const K = this.attrs.length;
    const out = new Float64Array(mem.N * K);
    const origIds = this.chunk.origIds;
    this.attrs.forEach((a, k) => {
      if (a !== 'id') return;
      if (!origIds) throw new StyleError(`compute ${this.id} (property/chunk): attribute id needs compress yes in compute ${this.chunk.id}`);
      for (let c = 0; c < mem.N; c++) out[K * c + k] = origIds[c];
    });
    for (let j = 0; j < mem.n; j++) {
      const c = mem.chunk[j];
      this.attrs.forEach((a, k) => {
        if (a === 'count') out[K * c + k] += 1;
      });
    }
    this.sizeArrayRows = mem.N;
    return out;
  }

  protected computeVector(): void {
    this.chunk.ensure();
    this.vector = this.values();
    this.sizeVector = this.chunk.nchunk;
  }

  protected computeArray(): void {
    this.chunk.ensure();
    this.array = this.values();
  }
}
