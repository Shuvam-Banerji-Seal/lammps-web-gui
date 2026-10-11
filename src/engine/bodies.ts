import { StyleError } from './force/types';

/*
 * atom_style body nparticle: per-body storage shared by the atom code (read_data, the atom arrays), the
 * body pair style and fix nve/body.
 *
 * Howto_body.rst (nparticle): "The 6 moments of inertia (ixx,iyy,izz,ixy,ixz,iyz) should be the values
 * consistent with the current orientation of the rigid body around its center of mass. The values are with
 * respect to the simulation box XYZ axes, not with respect to the principal axes of the rigid body itself.
 * LAMMPS performs the latter calculation internally." and "The coordinates of each sub-particle are specified
 * as its x,y,z displacement from the center-of-mass of the body particle."
 *
 * The inertia tensor is the symmetric matrix [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]. Measured with
 * native LAMMPS (black box): a body with ixy 0.5 and angular momentum (1,1,0) has the kinetic energy 0.5 L.I^-1.L of
 * that matrix, not of ixy = -0.5 (temp/body with the rotational dof gives the same numbers). Its eigen-decomposition
 * gives the principal moments and the orientation quaternion of the principal axes. The engine keeps every body's
 * sub-particle displacements in the body frame (d_body = E^T d_space), so that the space displacement is R d_body
 * at every step.
 */

/** Name of the per-atom body record in SimState.custom (the '#' keeps it out of user property names). */
export const BODY_RECORD = '#body';

/**
 * Record layout per atom. nparticle, stride 4 + 3 Nmax: [nsub, I1, I2, I3, then per sub-particle k the body-frame
 * displacement (dx, dy, dz) at 4 + 3k]. rounded/polygon, stride 5 + 3 Nmax: [nvert, I1, I2, I3, diameter, then per
 * vertex k the body-frame displacement (dx, dy, dz) at 5 + 3k]. Point particles (bodyflag 0) have a zero count.
 */
export const bodyRecordStride = (nmax: number, bstyle: string = 'nparticle'): number => bodyRecordOffset(bstyle) + 3 * nmax;

/** Index of the first sub-particle (vertex) displacement in the record of a body style. */
export const bodyRecordOffset = (bstyle: string): number => (bstyle === 'rounded/polygon' ? 5 : 4);

/**
 * atom_style body nparticle Nmin Nmax | rounded/polygon Nmin Nmax: the body style and its bounds, or null for
 * other styles. Howto_body.rst: "atom_style body rounded/polygon Nmin Nmax" with "Nmin = minimum # of vertices in
 * any body in the system".
 */
export const parseBodyStyle = (st: string): { bstyle: string; nmin: number; nmax: number } | null => {
  if (!st.startsWith('body ')) return null;
  const w = st.split(/\s+/);
  return { bstyle: w[1], nmin: Number(w[2]), nmax: Number(w[3]) };
};

/** Eigen-decomposition of a symmetric 3x3 matrix (row-major, 9 entries) by cyclic Jacobi rotations. */
export const eigenSym3 = (m: ArrayLike<number>): { values: [number, number, number]; vectors: number[] } => {
  const A = Array.from(m);
  let V = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const mul = (a: number[], b: number[]): number[] => {
    const o = new Array<number>(9).fill(0);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
    return o;
  };
  const tr = (a: number[]): number[] => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
  for (let sweep = 0; sweep < 100; sweep++) {
    const off = A[1] * A[1] + A[2] * A[2] + A[5] * A[5];
    const scale = A[0] * A[0] + A[4] * A[4] + A[8] * A[8] + 2 * off;
    if (off <= 1e-34 * scale || off === 0) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      const apq = A[3 * p + q];
      if (apq === 0) continue;
      const theta = (A[3 * q + q] - A[3 * p + p]) / (2 * apq);
      const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      const J = [1, 0, 0, 0, 1, 0, 0, 0, 1];
      J[3 * p + p] = c; J[3 * q + q] = c; J[3 * p + q] = s; J[3 * q + p] = -s;
      const JA = mul(tr(J), A);
      A.splice(0, 9, ...mul(JA, J));
      V = mul(V, J);
    }
  }
  const values: [number, number, number] = [A[0], A[4], A[8]];
  // a proper rotation: det(V) = +1 (flip the last axis otherwise)
  const det = V[0] * (V[4] * V[8] - V[5] * V[7]) - V[1] * (V[3] * V[8] - V[5] * V[6]) + V[2] * (V[3] * V[7] - V[4] * V[6]);
  if (det < 0) { V[2] = -V[2]; V[5] = -V[5]; V[8] = -V[8]; }
  return { values, vectors: V };
};

/** Unit quaternion (w, i, j, k) of a proper rotation matrix (row-major): the inverse of quatToMatrix. */
export const rotationToQuat = (R: ArrayLike<number>): [number, number, number, number] => {
  const tr = R[0] + R[4] + R[8];
  let w: number, x: number, y: number, z: number;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s; x = (R[7] - R[5]) / s; y = (R[2] - R[6]) / s; z = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
    w = (R[7] - R[5]) / s; x = 0.25 * s; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
    w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = 0.25 * s; z = (R[5] + R[7]) / s;
  } else {
    const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
    w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z);
  if (w < 0) return [-w / n, -x / n, -y / n, -z / n];
  return [w / n, x / n, y / n, z / n];
};

/** Principal frame of a body from its 6 inertia values (ixx iyy izz ixy ixz iyz). */
export const principalFrame = (I6: ArrayLike<number>): { moments: [number, number, number]; E: number[]; quat: [number, number, number, number] } => {
  const M = [I6[0], I6[3], I6[4], I6[3], I6[1], I6[5], I6[4], I6[5], I6[2]];
  const { values, vectors } = eigenSym3(M);
  // A rigid body's principal moments are sums of m r^2, so they are not negative. A rod (two sub-particles on one
  // line) has a zero moment about its axis, which the data file rounds to a value of order 1e-7 of either sign
  // (examples/body/data.body): only a clearly negative moment is refused.
  const big = Math.max(Math.abs(values[0]), Math.abs(values[1]), Math.abs(values[2]));
  for (const v of values) {
    if (v < -1e-6 * big) throw new StyleError('read_data Bodies: the inertia tensor of a body must not have a negative principal moment');
  }
  return { moments: values, E: vectors, quat: rotationToQuat(vectors) };
};
