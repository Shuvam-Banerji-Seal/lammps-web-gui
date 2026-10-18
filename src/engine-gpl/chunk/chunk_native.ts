/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/compute_chunk.cpp (ComputeChunk::init,
 * the chunk lookup), src/compute_com_chunk.cpp (ComputeCOMChunk::compute_array, the unwrapped mass sums without wrap),
 * src/compute_omega_chunk.cpp (ComputeOmegaChunk::compute_array, with EPSILON = 1.0e-6) and the MathExtra / MathEigen
 * calls they make (cross3, dot3, angmom_to_omega, jacobi3), at commit b332f28. Copyright (2003) Sandia Corporation
 * (LAMMPS development team). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Per-chunk sums shared by the chunk computes and by fix momentum/chunk. Every sum runs over the atoms in the native
 * storage order (nativeOrder), so the rounding follows the order native LAMMPS adds the atoms in. The engine has one
 * process, so MPI_Allreduce is the identity.
 */
import { ComputeChunkAtom } from '../../engine/compute/chunk';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import { angmomToOmega, jacobi3, type V3 } from '../rigid/math_extra';

/** EPSILON of src/compute_omega_chunk.cpp. */
const EPSILON = 1.0e-6;

/** The compute chunk/atom with this ID (ComputeChunk::init): an error if it is missing or of another style. */
export const chunkOf = (sys: System, id: string, style: string): ComputeChunkAtom => {
  const c = sys.computes.find((x) => x.id === id);
  if (!(c instanceof ComputeChunkAtom)) {
    throw new StyleError(`Chunk/atom compute ${id} does not exist or is incorrect style for compute ${style}`);
  }
  return c;
};

/** ComputeChunk::compute_array: refresh the chunk IDs of the atoms and return the number of chunks. */
export const refreshChunk = (chunk: ComputeChunkAtom): number => {
  chunk.ensure();
  return chunk.nchunk;
};

/** Unwrapped mass sums per chunk over the group: com[3c+d] = sum m u_d (no division), masstotal[c] = sum m. */
export interface ChunkSums {
  sum: Float64Array;
  masstotal: Float64Array;
}

/** The per-chunk COM sums of ComputeCOMChunk::compute_array (before the division by the chunk mass). */
export const comSums = (sys: System, groupBit: number, ichunk: Int32Array, N: number): ChunkSums => {
  const s = sys.state;
  const sum = new Float64Array(3 * N);
  const masstotal = new Float64Array(N);
  const u = [0, 0, 0];
  for (const i of nativeOrder(s)) {
    if (!(s.mask[i] & groupBit)) continue;
    const c = ichunk[i] - 1;
    if (c < 0) continue;
    const m = massOf(s, i);
    sys.geom.unwrap(s.x, s.image, i, u);
    sum[3 * c] += u[0] * m;
    sum[3 * c + 1] += u[1] * m;
    sum[3 * c + 2] += u[2] * m;
    masstotal[c] += m;
  }
  return { sum, masstotal };
};

/** Divides each chunk's COM sums by its mass when the mass is positive; otherwise the chunk COM is zero. */
export const comCentres = (sums: ChunkSums, N: number): Float64Array => {
  const com = new Float64Array(3 * N);
  for (let c = 0; c < N; c++) {
    for (let d = 0; d < 3; d++) {
      com[3 * c + d] = sums.masstotal[c] > 0.0 ? sums.sum[3 * c + d] / sums.masstotal[c] : 0.0;
    }
  }
  return com;
};

/** The per-chunk momentum sums sum m v (ComputeVCMChunk / ComputeVACFChunk), and the chunk masses. */
export const vcmSums = (sys: System, groupBit: number, ichunk: Int32Array, N: number): ChunkSums => {
  const s = sys.state;
  const sum = new Float64Array(3 * N);
  const masstotal = new Float64Array(N);
  for (const i of nativeOrder(s)) {
    if (!(s.mask[i] & groupBit)) continue;
    const c = ichunk[i] - 1;
    if (c < 0) continue;
    const m = massOf(s, i);
    sum[3 * c] += s.v[3 * i] * m;
    sum[3 * c + 1] += s.v[3 * i + 1] * m;
    sum[3 * c + 2] += s.v[3 * i + 2] * m;
    masstotal[c] += m;
  }
  return { sum, masstotal };
};

/**
 * ComputeOmegaChunk::compute_array: the angular velocity of each chunk from its point-mass inertia tensor about the
 * chunk COM and its angular momentum (v relative to the COM). A non-singular inertia tensor is inverted with the
 * determinant; a (nearly) singular one (a 2-atom chunk or a line) takes the principal axes from jacobi3, sets the
 * principal moments below EPSILON times the largest to zero and uses angmom_to_omega. Returns 3 values per chunk.
 */
export const omegaChunk = (sys: System, groupBit: number, ichunk: Int32Array, N: number): Float64Array => {
  const s = sys.state;
  const cen = comCentres(comSums(sys, groupBit, ichunk, N), N);
  const inertia = new Float64Array(6 * N);
  const u = [0, 0, 0];
  for (const i of nativeOrder(s)) {
    if (!(s.mask[i] & groupBit)) continue;
    const c = ichunk[i] - 1;
    if (c < 0) continue;
    const m = massOf(s, i);
    sys.geom.unwrap(s.x, s.image, i, u);
    const dx = u[0] - cen[3 * c], dy = u[1] - cen[3 * c + 1], dz = u[2] - cen[3 * c + 2];
    inertia[6 * c] += m * (dy * dy + dz * dz);
    inertia[6 * c + 1] += m * (dx * dx + dz * dz);
    inertia[6 * c + 2] += m * (dx * dx + dy * dy);
    inertia[6 * c + 3] -= m * dx * dy;
    inertia[6 * c + 4] -= m * dy * dz;
    inertia[6 * c + 5] -= m * dx * dz;
  }
  const angmom = new Float64Array(3 * N);
  for (const i of nativeOrder(s)) {
    if (!(s.mask[i] & groupBit)) continue;
    const c = ichunk[i] - 1;
    if (c < 0) continue;
    const m = massOf(s, i);
    sys.geom.unwrap(s.x, s.image, i, u);
    const dx = u[0] - cen[3 * c], dy = u[1] - cen[3 * c + 1], dz = u[2] - cen[3 * c + 2];
    const vx = s.v[3 * i], vy = s.v[3 * i + 1], vz = s.v[3 * i + 2];
    angmom[3 * c] += m * (dy * vz - dz * vy);
    angmom[3 * c + 1] += m * (dz * vx - dx * vz);
    angmom[3 * c + 2] += m * (dx * vy - dy * vx);
  }
  const omega = new Float64Array(3 * N);
  for (let c = 0; c < N; c++) {
    const w = omegaOfChunk(inertia, 6 * c, [angmom[3 * c], angmom[3 * c + 1], angmom[3 * c + 2]]);
    omega[3 * c] = w[0]; omega[3 * c + 1] = w[1]; omega[3 * c + 2] = w[2];
  }
  return omega;
};

/** The omega solve of ComputeOmegaChunk::compute_array for one chunk (inertia in xx,yy,zz,xy,yz,xz order at off). */
const omegaOfChunk = (inertia: Float64Array, off: number, mall: V3): V3 => {
  const iall = [inertia[off], inertia[off + 1], inertia[off + 2], inertia[off + 3], inertia[off + 4], inertia[off + 5]];
  const determinant = iall[0] * (iall[1] * iall[2] - iall[4] * iall[4]) +
    iall[3] * (iall[4] * iall[5] - iall[3] * iall[2]) +
    iall[5] * (iall[3] * iall[4] - iall[1] * iall[5]);
  const ione: number[][] = [
    [iall[0], iall[3], iall[5]],
    [iall[3], iall[1], iall[4]],
    [iall[5], iall[4], iall[2]],
  ];
  if (determinant > EPSILON) {
    const inverse: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    inverse[0][0] = ione[1][1] * ione[2][2] - ione[1][2] * ione[2][1];
    inverse[0][1] = -(ione[0][1] * ione[2][2] - ione[0][2] * ione[2][1]);
    inverse[0][2] = ione[0][1] * ione[1][2] - ione[0][2] * ione[1][1];
    inverse[1][0] = -(ione[1][0] * ione[2][2] - ione[1][2] * ione[2][0]);
    inverse[1][1] = ione[0][0] * ione[2][2] - ione[0][2] * ione[2][0];
    inverse[1][2] = -(ione[0][0] * ione[1][2] - ione[0][2] * ione[1][0]);
    inverse[2][0] = ione[1][0] * ione[2][1] - ione[1][1] * ione[2][0];
    inverse[2][1] = -(ione[0][0] * ione[2][1] - ione[0][1] * ione[2][0]);
    inverse[2][2] = ione[0][0] * ione[1][1] - ione[0][1] * ione[1][0];
    const invdeterminant = 1.0 / determinant;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) inverse[i][j] *= invdeterminant;
    return [
      inverse[0][0] * mall[0] + inverse[0][1] * mall[1] + inverse[0][2] * mall[2],
      inverse[1][0] * mall[0] + inverse[1][1] * mall[1] + inverse[1][2] * mall[2],
      inverse[2][0] * mall[0] + inverse[2][1] * mall[1] + inverse[2][2] * mall[2],
    ];
  }
  // (nearly) singular I matrix: diagonalize, as native jacobi3 + angmom_to_omega
  const { eval: idiag0, evec } = jacobi3(ione, true);
  const idiag: V3 = [idiag0[0], idiag0[1], idiag0[2]];
  const ex: V3 = [evec[0][0], evec[1][0], evec[2][0]];
  const ey: V3 = [evec[0][1], evec[1][1], evec[2][1]];
  const ez: V3 = [evec[0][2], evec[1][2], evec[2][2]];
  // right-handed coordinate system: flip the third vector if ex x ey points against it
  const cross: V3 = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
  if (cross[0] * ez[0] + cross[1] * ez[1] + cross[2] * ez[2] < 0.0) for (let k = 0; k < 3; k++) ez[k] = -ez[k];
  // principal moments below EPSILON times the largest one are set to zero
  let max = Math.max(idiag[0], idiag[1]);
  max = Math.max(max, idiag[2]);
  for (let k = 0; k < 3; k++) if (idiag[k] < EPSILON * max) idiag[k] = 0.0;
  const w: V3 = [0, 0, 0];
  angmomToOmega(mall, ex, ey, ez, idiag, w);
  return w;
};
