/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/math_extra.h and src/math_extra.cpp at commit b332f28 (the MathExtra helpers used by
 * src/ASPHERE/pair_gayberne.cpp: quat_to_mat_trans, diag_times3, transpose_times3, plus3, det3, mldivide3,
 * normalize3, dot3, cross3, vecmat). Copyright (2003) Sandia Corporation, distributed under the GNU General Public
 * License (the LAMMPS file headers). Original authors: Mike Brown (SNL) and the LAMMPS development team
 * (developers@lammps.org). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Matrices are 3 rows of 3 entries, indexed exactly as the C arrays m[row][col] (the operation order of every
 * helper is kept, so the results match the native code bit for bit).
 */

/** A 3x3 matrix as C double m[3][3]: m[row][col]. */
export type Mat3 = Float64Array[];

export const newMat3 = (): Mat3 => [new Float64Array(3), new Float64Array(3), new Float64Array(3)];

/** MathExtra::quat_to_mat_trans: quat = (w, i, j, k), no normalization (the caller's quaternion is used as stored). */
export function quatToMatTrans(quat: ArrayLike<number>, mat: Mat3): void {
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
}

/** MathExtra::diag_times3: ans = diag(d) * m (ans must not alias m). */
export function diagTimes3(d: ArrayLike<number>, m: Mat3, ans: Mat3): void {
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) ans[r][c] = d[r] * m[r][c];
}

/** MathExtra::transpose_times3: ans = transpose(m) * m2 (ans must not alias m or m2). */
export function transposeTimes3(m: Mat3, m2: Mat3, ans: Mat3): void {
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) ans[a][b] = m[0][a] * m2[0][b] + m[1][a] * m2[1][b] + m[2][a] * m2[2][b];
  }
}

/** MathExtra::plus3: ans = m + m2 (elementwise). */
export function plus3(m: Mat3, m2: Mat3, ans: Mat3): void {
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) ans[r][c] = m[r][c] + m2[r][c];
}

/** MathExtra::det3. */
export function det3(m: Mat3): number {
  return m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1] -
    m[1][0] * m[0][1] * m[2][2] + m[1][0] * m[0][2] * m[2][1] + m[2][0] * m[0][1] * m[1][2] -
    m[2][0] * m[0][2] * m[1][1];
}

/** MathExtra::normalize3: ans = v / |v| (ans is left unchanged for a zero vector, as native). */
export function normalize3(v: ArrayLike<number>, ans: Float64Array): void {
  const val = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (val > 0.0) {
    const scale = 1.0 / Math.sqrt(val);
    ans[0] = v[0] * scale;
    ans[1] = v[1] * scale;
    ans[2] = v[2] * scale;
  }
}

/** MathExtra::dot3. */
export function dot3(v1: ArrayLike<number>, v2: ArrayLike<number>): number {
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

/** MathExtra::cross3 (ans must not alias v1 or v2). */
export function cross3(v1: ArrayLike<number>, v2: ArrayLike<number>, ans: Float64Array): void {
  ans[0] = v1[1] * v2[2] - v1[2] * v2[1];
  ans[1] = v1[2] * v2[0] - v1[0] * v2[2];
  ans[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

/** MathExtra::vecmat: ans = v^T m (row vector times matrix). */
export function vecmat(v: ArrayLike<number>, m: Mat3, ans: Float64Array): void {
  ans[0] = v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0];
  ans[1] = v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1];
  ans[2] = v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2];
}

/**
 * MathExtra::mldivide3: solve M ans = v by Gaussian elimination with the native partial pivoting. Returns 0 on
 * success and 1 for a singular matrix (the caller raises the native "Bad matrix inversion" error).
 */
export function mldivide3(m: Mat3, v: ArrayLike<number>, ans: Float64Array): number {
  // create augmented matrix for pivoting
  const aug: Float64Array[] = [new Float64Array(4), new Float64Array(4), new Float64Array(4)];
  for (let i = 0; i < 3; i++) {
    aug[i][3] = v[i];
    for (let j = 0; j < 3; j++) aug[i][j] = m[i][j];
  }

  for (let i = 0; i < 2; i++) {
    let p = i;
    for (let j = i + 1; j < 3; j++) {
      if (Math.abs(aug[j][i]) > Math.abs(aug[i][i])) {
        const tempv = Float64Array.from(aug[i]);
        aug[i].set(aug[j]);
        aug[j].set(tempv);
      }
    }

    while (p < 3 && aug[p][i] === 0.0) p++;

    if (p === 3) return 1;
    if (p !== i) {
      const tempv = Float64Array.from(aug[i]);
      aug[i].set(aug[p]);
      aug[p].set(tempv);
    }

    for (let j = i + 1; j < 3; j++) {
      const n = aug[j][i] / aug[i][i];
      for (let k = i + 1; k < 4; k++) aug[j][k] -= n * aug[i][k];
    }
  }

  if (aug[2][2] === 0.0) return 1;

  // back substitution
  ans[2] = aug[2][3] / aug[2][2];
  for (let i = 1; i >= 0; i--) {
    let sumax = 0.0;
    for (let j = i + 1; j < 3; j++) sumax += aug[i][j] * ans[j];
    ans[i] = (aug[i][3] - sumax) / aug[i][i];
  }
  return 0;
}

/**
 * PairGayBerne::compute_eta_torque (src/ASPHERE/pair_gayberne.cpp at b332f28): the trace in the torque derivative
 * of eta, from the symbolic solver dump of LAMMPS (the expressions are kept verbatim). m is g12, m2 is a_i, s is
 * the shape (shape2) of the particle.
 */
export function computeEtaTorque(m: Mat3, m2: Mat3, s: ArrayLike<number>, ans: Mat3): void {
  const den = m[1][0]*m[0][2]*m[2][1]-m[0][0]*m[1][2]*m[2][1]-
    m[0][2]*m[2][0]*m[1][1]+m[0][1]*m[2][0]*m[1][2]-
    m[1][0]*m[0][1]*m[2][2]+m[0][0]*m[1][1]*m[2][2];

  ans[0][0] = s[0]*(m[1][2]*m[0][1]*m2[0][2]+2.0*m[1][1]*m[2][2]*m2[0][0]-
                    m[1][1]*m2[0][2]*m[0][2]-2.0*m[1][2]*m2[0][0]*m[2][1]+
                    m2[0][1]*m[0][2]*m[2][1]-m2[0][1]*m[0][1]*m[2][2]-
                    m[1][0]*m[2][2]*m2[0][1]+m[2][0]*m[1][2]*m2[0][1]+
                    m[1][0]*m2[0][2]*m[2][1]-m2[0][2]*m[2][0]*m[1][1])/den;

  ans[0][1] = s[0]*(m[0][2]*m2[0][0]*m[2][1]-m[2][2]*m2[0][0]*m[0][1]+
                    2.0*m[0][0]*m[2][2]*m2[0][1]-m[0][0]*m2[0][2]*m[1][2]-
                    2.0*m[2][0]*m[0][2]*m2[0][1]+m2[0][2]*m[1][0]*m[0][2]-
                    m[2][2]*m[1][0]*m2[0][0]+m[2][0]*m2[0][0]*m[1][2]+
                    m[2][0]*m2[0][2]*m[0][1]-m2[0][2]*m[0][0]*m[2][1])/den;

  ans[0][2] = s[0]*(m[0][1]*m[1][2]*m2[0][0]-m[0][2]*m2[0][0]*m[1][1]-
                    m[0][0]*m[1][2]*m2[0][1]+m[1][0]*m[0][2]*m2[0][1]-
                    m2[0][1]*m[0][0]*m[2][1]-m[2][0]*m[1][1]*m2[0][0]+
                    2.0*m[1][1]*m[0][0]*m2[0][2]-2.0*m[1][0]*m2[0][2]*m[0][1]+
                    m[1][0]*m[2][1]*m2[0][0]+m[2][0]*m2[0][1]*m[0][1])/den;

  ans[1][0] = s[1]*(-m[1][1]*m2[1][2]*m[0][2]+2.0*m[1][1]*m[2][2]*m2[1][0]+
                    m[1][2]*m[0][1]*m2[1][2]-2.0*m[1][2]*m2[1][0]*m[2][1]+
                    m2[1][1]*m[0][2]*m[2][1]-m2[1][1]*m[0][1]*m[2][2]-
                    m[1][0]*m[2][2]*m2[1][1]+m[2][0]*m[1][2]*m2[1][1]-
                    m2[1][2]*m[2][0]*m[1][1]+m[1][0]*m2[1][2]*m[2][1])/den;

  ans[1][1] = s[1]*(m[0][2]*m2[1][0]*m[2][1]-m[0][1]*m[2][2]*m2[1][0]+
                    2.0*m[2][2]*m[0][0]*m2[1][1]-m2[1][2]*m[0][0]*m[1][2]-
                    2.0*m[2][0]*m2[1][1]*m[0][2]-m[1][0]*m[2][2]*m2[1][0]+
                    m[2][0]*m[1][2]*m2[1][0]+m[1][0]*m2[1][2]*m[0][2]-
                    m[0][0]*m2[1][2]*m[2][1]+m2[1][2]*m[0][1]*m[2][0])/den;

  ans[1][2] = s[1]*(m[0][1]*m[1][2]*m2[1][0]-m[0][2]*m2[1][0]*m[1][1]-
                    m[0][0]*m[1][2]*m2[1][1]+m[1][0]*m[0][2]*m2[1][1]+
                    2.0*m[1][1]*m[0][0]*m2[1][2]-m[0][0]*m2[1][1]*m[2][1]+
                    m[0][1]*m[2][0]*m2[1][1]-m2[1][0]*m[2][0]*m[1][1]-
                    2.0*m[1][0]*m[0][1]*m2[1][2]+m[1][0]*m2[1][0]*m[2][1])/den;

  ans[2][0] = s[2]*(-m[1][1]*m[0][2]*m2[2][2]+m[0][1]*m[1][2]*m2[2][2]+
                    2.0*m[1][1]*m2[2][0]*m[2][2]-m[0][1]*m2[2][1]*m[2][2]+
                    m[0][2]*m[2][1]*m2[2][1]-2.0*m2[2][0]*m[2][1]*m[1][2]-
                    m[1][0]*m2[2][1]*m[2][2]+m[1][2]*m[2][0]*m2[2][1]-
                    m[1][1]*m[2][0]*m2[2][2]+m[2][1]*m[1][0]*m2[2][2])/den;

  ans[2][1] = s[2]*-(m[0][1]*m[2][2]*m2[2][0]-m[0][2]*m2[2][0]*m[2][1]-
                     2.0*m2[2][1]*m[0][0]*m[2][2]+m[1][2]*m2[2][2]*m[0][0]+
                     2.0*m2[2][1]*m[0][2]*m[2][0]+m[1][0]*m2[2][0]*m[2][2]-
                     m[1][0]*m[0][2]*m2[2][2]-m[1][2]*m[2][0]*m2[2][0]+
                     m[0][0]*m2[2][2]*m[2][1]-m2[2][2]*m[0][1]*m[2][0])/den;

  ans[2][2] = s[2]*(m[0][1]*m[1][2]*m2[2][0]-m[0][2]*m2[2][0]*m[1][1]-
                    m[0][0]*m[1][2]*m2[2][1]+m[1][0]*m[0][2]*m2[2][1]-
                    m[1][1]*m[2][0]*m2[2][0]-m[2][1]*m2[2][1]*m[0][0]+
                    2.0*m[1][1]*m2[2][2]*m[0][0]+m[2][1]*m[1][0]*m2[2][0]+
                    m[2][0]*m[0][1]*m2[2][1]-2.0*m2[2][2]*m[1][0]*m[0][1])/den;
}

