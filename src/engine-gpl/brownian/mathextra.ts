/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the vector, matrix and quaternion
 * operations of src/math_extra.h and src/math_extra.cpp that the BROWNIAN fixes call (scale3, scaleadd3,
 * snormalize3, len3, dot3, cross3, matvec, transpose_matvec, quatvec, qnormalize, quat_to_mat,
 * quat_to_mat_trans). The operations keep the source's order of floating-point steps.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

/** m[row][col] */
export type Mat3 = number[][];

/** MathExtra::scale3(s, v, ans): ans = s * v. */
export const scale3 = (s: number, v: ArrayLike<number>, ans: number[]): void => {
  ans[0] = s * v[0];
  ans[1] = s * v[1];
  ans[2] = s * v[2];
};

/** MathExtra::scaleadd3(s, v1, v2, ans): ans = s * v1 + v2. */
export const scaleadd3 = (s: number, v1: ArrayLike<number>, v2: ArrayLike<number>, ans: number[]): void => {
  ans[0] = s * v1[0] + v2[0];
  ans[1] = s * v1[1] + v2[1];
  ans[2] = s * v1[2] + v2[2];
};

/** MathExtra::scaleadd3(s1, v1, s2, v2, ans): ans = s1 * v1 + s2 * v2. */
export const scaleadd3Two = (s1: number, v1: ArrayLike<number>, s2: number, v2: ArrayLike<number>, ans: number[]): void => {
  ans[0] = s1 * v1[0] + s2 * v2[0];
  ans[1] = s1 * v1[1] + s2 * v2[1];
  ans[2] = s1 * v1[2] + s2 * v2[2];
};

/** MathExtra::snormalize3(length, v, ans): ans = length * v / |v| (ans is left alone when v is zero). */
export const snormalize3 = (length: number, v: ArrayLike<number>, ans: number[]): void => {
  const val = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (val > 0.0) {
    const scale = length / Math.sqrt(val);
    ans[0] = v[0] * scale;
    ans[1] = v[1] * scale;
    ans[2] = v[2] * scale;
  }
};

export const len3 = (v: ArrayLike<number>): number => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

export const dot3 = (a: ArrayLike<number>, b: ArrayLike<number>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export const cross3 = (a: ArrayLike<number>, b: ArrayLike<number>, ans: number[]): void => {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  ans[0] = x;
  ans[1] = y;
  ans[2] = z;
};

/** MathExtra::matvec(m, v, ans): ans = m v. */
export const matvec = (m: Mat3, v: ArrayLike<number>, ans: number[]): void => {
  const a0 = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2];
  const a1 = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2];
  const a2 = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2];
  ans[0] = a0;
  ans[1] = a1;
  ans[2] = a2;
};

/** MathExtra::transpose_matvec(m, v, ans): ans = m^T v. */
export const transposeMatvec = (m: Mat3, v: ArrayLike<number>, ans: number[]): void => {
  const a0 = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2];
  const a1 = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2];
  const a2 = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2];
  ans[0] = a0;
  ans[1] = a1;
  ans[2] = a2;
};

/** MathExtra::qnormalize(q), in place. */
export const qnormalize = (q: number[]): void => {
  const norm = 1.0 / Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  q[0] *= norm;
  q[1] *= norm;
  q[2] *= norm;
  q[3] *= norm;
};

/** MathExtra::quatvec(a, b, c): c = a * (0, b), quaternion times vector (c is a 4-vector). */
export const quatvec = (a: ArrayLike<number>, b: ArrayLike<number>, c: number[]): void => {
  const c0 = -a[1] * b[0] - a[2] * b[1] - a[3] * b[2];
  const c1 = a[0] * b[0] + a[2] * b[2] - a[3] * b[1];
  const c2 = a[0] * b[1] + a[3] * b[0] - a[1] * b[2];
  const c3 = a[0] * b[2] + a[1] * b[1] - a[2] * b[0];
  c[0] = c0;
  c[1] = c1;
  c[2] = c2;
  c[3] = c3;
};

/** MathExtra::quat_to_mat(quat, mat): rotation matrix of the quaternion quat = (w, i, j, k). */
export const quatToMat = (quat: ArrayLike<number>, mat: Mat3): void => {
  const w2 = quat[0] * quat[0];
  const i2 = quat[1] * quat[1];
  const j2 = quat[2] * quat[2];
  const k2 = quat[3] * quat[3];
  const twoij = 2.0 * quat[1] * quat[2];
  const twoik = 2.0 * quat[1] * quat[3];
  const twojk = 2.0 * quat[2] * quat[3];
  const twoiw = 2.0 * quat[1] * quat[0];
  const twojw = 2.0 * quat[2] * quat[0];
  const twokw = 2.0 * quat[3] * quat[0];
  mat[0][0] = w2 + i2 - j2 - k2;
  mat[0][1] = twoij - twokw;
  mat[0][2] = twojw + twoik;
  mat[1][0] = twoij + twokw;
  mat[1][1] = w2 - i2 + j2 - k2;
  mat[1][2] = twojk - twoiw;
  mat[2][0] = twoik - twojw;
  mat[2][1] = twojk + twoiw;
  mat[2][2] = w2 - i2 - j2 + k2;
};

/** MathExtra::quat_to_mat_trans(quat, mat): the transpose of quat_to_mat (the conjugate rotation). */
export const quatToMatTrans = (quat: ArrayLike<number>, mat: Mat3): void => {
  const w2 = quat[0] * quat[0];
  const i2 = quat[1] * quat[1];
  const j2 = quat[2] * quat[2];
  const k2 = quat[3] * quat[3];
  const twoij = 2.0 * quat[1] * quat[2];
  const twoik = 2.0 * quat[1] * quat[3];
  const twojk = 2.0 * quat[2] * quat[3];
  const twoiw = 2.0 * quat[1] * quat[0];
  const twojw = 2.0 * quat[2] * quat[0];
  const twokw = 2.0 * quat[3] * quat[0];
  mat[0][0] = w2 + i2 - j2 - k2;
  mat[1][0] = twoij - twokw;
  mat[2][0] = twojw + twoik;
  mat[0][1] = twoij + twokw;
  mat[1][1] = w2 - i2 + j2 - k2;
  mat[2][1] = twojk - twoiw;
  mat[0][2] = twoik - twojw;
  mat[1][2] = twojk + twoiw;
  mat[2][2] = w2 - i2 - j2 + k2;
};
