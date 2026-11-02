/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MACHDYN/smd_math.h (namespace SMD_Math: Deviator, PolDec, pseudo_inverse_SVD) and the Eigen calls the
 * MACHDYN styles make on 3x3 matrices (Eigen::JacobiSVD, Eigen::Matrix3d::inverse, determinant, trace). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE). Copyright (2014) Georg C. Ganzenmueller, Fraunhofer Ernst-Mach
 * Institute for High-Speed Dynamics, EMI (the MACHDYN package author).
 *
 * The matrices are 3x3 row-major Float64Array(9) (M[3*r + c] = M(r, c)). Eigen's JacobiSVD is replaced by a one-sided
 * Jacobi SVD with the singular values sorted in decreasing order, as Eigen returns them. The rotation of a polar
 * decomposition is unique for a non-singular matrix, so the result does not depend on the SVD variant; the
 * pseudo-inverse of the MACHDYN code uses only U and the singular values, which is the same for a symmetric matrix
 * whatever the sign choice of the singular vectors.
 */

/**
 * Plain arrays (not typed arrays): the styles make millions of 3x3 temporaries per run, and a small array is much
 * cheaper to allocate than a Float64Array.
 */
export type Mat3 = number[];

export const mat3 = (): Mat3 => [0, 0, 0, 0, 0, 0, 0, 0, 0];
export const identity3 = (): Mat3 => {
  const m = mat3();
  m[0] = m[4] = m[8] = 1;
  return m;
};
/** Copies the entries of src into dst (both 3x3). */
export const setM = (dst: Mat3, src: ArrayLike<number>): void => {
  for (let k = 0; k < 9; k++) dst[k] = src[k];
};
export const mul3 = (a: Mat3, b: Mat3, out: Mat3 = mat3()): Mat3 => {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
    }
  }
  return out;
};
export const transpose3 = (a: Mat3, out: Mat3 = mat3()): Mat3 => {
  // read before writing: out may be a
  const a1 = a[1], a2 = a[2], a3 = a[3], a5 = a[5], a6 = a[6], a7 = a[7];
  out[0] = a[0]; out[1] = a3; out[2] = a6;
  out[3] = a1; out[4] = a[4]; out[5] = a7;
  out[6] = a2; out[7] = a5; out[8] = a[8];
  return out;
};
export const add3 = (a: Mat3, b: Mat3, out: Mat3 = mat3()): Mat3 => {
  for (let k = 0; k < 9; k++) out[k] = a[k] + b[k];
  return out;
};
export const sub3 = (a: Mat3, b: Mat3, out: Mat3 = mat3()): Mat3 => {
  for (let k = 0; k < 9; k++) out[k] = a[k] - b[k];
  return out;
};
export const scale3 = (a: Mat3, s: number, out: Mat3 = mat3()): Mat3 => {
  for (let k = 0; k < 9; k++) out[k] = a[k] * s;
  return out;
};
export const trace3 = (a: Mat3): number => a[0] + a[4] + a[8];
export const det3 = (a: Mat3): number =>
  a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
/** Inverse by the adjugate (Eigen's 3x3 inverse is also the cofactor formula). Singular input gives infinities. */
export const inverse3 = (a: Mat3, out: Mat3 = mat3()): Mat3 => {
  const d = det3(a);
  // read before writing: out may be a
  const c0 = a[4] * a[8] - a[5] * a[7], c1 = a[2] * a[7] - a[1] * a[8], c2 = a[1] * a[5] - a[2] * a[4];
  const c3 = a[5] * a[6] - a[3] * a[8], c4 = a[0] * a[8] - a[2] * a[6], c5 = a[2] * a[3] - a[0] * a[5];
  const c6 = a[3] * a[7] - a[4] * a[6], c7 = a[1] * a[6] - a[0] * a[7], c8 = a[0] * a[4] - a[1] * a[3];
  out[0] = c0 / d; out[1] = c1 / d; out[2] = c2 / d;
  out[3] = c3 / d; out[4] = c4 / d; out[5] = c5 / d;
  out[6] = c6 / d; out[7] = c7 / d; out[8] = c8 / d;
  return out;
};
/** SMD_Math::Deviator: M minus its isotropic part. */
export const deviator3 = (m: Mat3, out: Mat3 = mat3()): Mat3 => {
  const p = trace3(m) / 3;
  for (let k = 0; k < 9; k++) out[k] = m[k];
  out[0] -= p; out[4] -= p; out[8] -= p;
  return out;
};
/** Frobenius norm (Eigen's Matrix::norm). */
export const norm3 = (a: Mat3): number => Math.sqrt(a.reduce((s, x) => s + x * x, 0));

/** Column pairs of one Jacobi sweep (p, q flattened). */
const JACOBI_PQ = [0, 1, 0, 2, 1, 2];

/** Singular value decomposition M = U diag(s) V^T of a 3x3 matrix (one-sided Jacobi), s sorted in decreasing order. */
export interface SVD3 { U: Mat3; s: Float64Array; V: Mat3 }
export const svd3 = (M: Mat3): SVD3 => {
  const U = mat3(), V = mat3(), s = new Float64Array(3);
  svd3Into(M, U, s, V);
  return { U, s, V };
};

/** svd3's working copies of the columns and of the rotations. */
const svdA = new Float64Array(9), svdV = new Float64Array(9);

/** svd3 into the given U, s and V (none of them M), with no allocation. */
const svd3Into = (M: ArrayLike<number>, U: Mat3, s: Float64Array, V2: Mat3): void => {
  // columns of A are rotated until they are mutually orthogonal; V accumulates the rotations
  const A = svdA, V = svdV;
  for (let k = 0; k < 9; k++) { A[k] = M[k]; V[k] = 0; }
  V[0] = V[4] = V[8] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let pq = 0; pq < 6; pq += 2) {
      const p = JACOBI_PQ[pq], q = JACOBI_PQ[pq + 1];
      let alpha = 0, beta = 0, gamma = 0;
      for (let r = 0; r < 3; r++) {
        alpha += A[3 * r + p] * A[3 * r + p];
        beta += A[3 * r + q] * A[3 * r + q];
        gamma += A[3 * r + p] * A[3 * r + q];
      }
      if (gamma === 0 || Math.abs(gamma) <= 1e-300) continue;
      const conv = Math.abs(gamma) / Math.sqrt(alpha * beta);
      if (!(conv > 1e-17)) continue;
      off = Math.max(off, conv);
      const zeta = (beta - alpha) / (2 * gamma);
      const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
      const c = 1 / Math.sqrt(1 + t * t), sn = c * t;
      for (let r = 0; r < 3; r++) {
        const ap = A[3 * r + p], aq = A[3 * r + q];
        A[3 * r + p] = c * ap - sn * aq;
        A[3 * r + q] = sn * ap + c * aq;
        const vp = V[3 * r + p], vq = V[3 * r + q];
        V[3 * r + p] = c * vp - sn * vq;
        V[3 * r + q] = sn * vp + c * vq;
      }
    }
    if (off <= 1e-16) break;
  }
  const sv0 = Math.hypot(A[0], A[3], A[6]), sv1 = Math.hypot(A[1], A[4], A[7]), sv2 = Math.hypot(A[2], A[5], A[8]);
  // the columns in decreasing order of their norm, equal ones in index order (a stable sort: the order that sorting
  // [0, 1, 2] with the comparator (x, y) => sv[y] - sv[x] gives)
  const sv = svdSv;
  sv[0] = sv0; sv[1] = sv1; sv[2] = sv2;
  const order = svdOrder;
  order[0] = 0; order[1] = 1; order[2] = 2;
  for (let i = 1; i < 3; i++) {
    const pivot = order[i];
    let left = 0, right = i;
    while (left < right) {
      const mid = (left + right) >> 1;
      if (sv[order[mid]] - sv[pivot] < 0) right = mid; else left = mid + 1;
    }
    for (let k = i; k > left; k--) order[k] = order[k - 1];
    order[left] = pivot;
  }
  for (let dst = 0; dst < 3; dst++) {
    const src = order[dst];
    s[dst] = sv[src];
    for (let r = 0; r < 3; r++) {
      V2[3 * r + dst] = V[3 * r + src];
      U[3 * r + dst] = sv[src] > 0 ? A[3 * r + src] / sv[src] : 0;
    }
  }
  completeOrthonormal(U, s);
};
const svdSv = new Float64Array(3), svdOrder = new Int32Array(3);

/**
 * Replaces the columns of U with zero singular values by unit vectors orthogonal to the other columns (their value only
 * enters a pseudo-inverse with a zero factor, so any orthonormal completion gives the same result).
 */
const completeOrthonormal = (U: Mat3, s: Float64Array): void => {
  if (s[0] > 0 && s[1] > 0 && s[2] > 0) return;
  const done = [s[0] > 0, s[1] > 0, s[2] > 0];
  for (let c = 0; c < 3; c++) {
    if (done[c]) continue;
    for (const e of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
      const v = [e[0], e[1], e[2]];
      for (let k = 0; k < 3; k++) {
        if (!done[k]) continue;
        const d = U[k] * v[0] + U[3 + k] * v[1] + U[6 + k] * v[2];
        v[0] -= d * U[k]; v[1] -= d * U[3 + k]; v[2] -= d * U[6 + k];
      }
      const n = Math.hypot(v[0], v[1], v[2]);
      if (n > 1e-8) {
        U[c] = v[0] / n; U[3 + c] = v[1] / n; U[6 + c] = v[2] / n;
        break;
      }
    }
    done[c] = true;
  }
};

/**
 * SMD_Math::PolDec (no scaling of the stretch): M = R T with R a rotation (U V^T) and T = V S V^T. If R is improper
 * (det < 0), the smallest singular value is flipped in sign and R is recomputed from M V S^-1 V^T. Returns
 * det(R) > 0. R and T are written to the output arrays.
 */
export const polDec = (M: Mat3, R: Mat3, T: Mat3): boolean => {
  const U = pdU, s = pdS, V = pdV, S = pdSm, Vt = pdVt;
  svd3Into(M, U, s, V);
  S.fill(0);
  S[0] = s[0]; S[4] = s[1]; S[8] = s[2];
  transpose3(V, Vt);
  mul3(U, Vt, R);
  mul3(mul3(V, S, pdT1), Vt, T);
  if (det3(R) < 0) {
    // the smallest entry of S is the one flipped (Eigen minCoeff: first index of the minimum)
    let imin = 0;
    for (let k = 1; k < 3; k++) if (s[k] < s[imin]) imin = k;
    S[4 * imin] *= -1;
    mul3(mul3(mul3(M, V, pdT1), inverse3(S, pdT2), pdT3), Vt, R);
  }
  return det3(R) > 0;
};
const pdU = mat3(), pdV = mat3(), pdSm = mat3(), pdVt = mat3(), pdT1 = mat3(), pdT2 = mat3(), pdT3 = mat3();
const pdS = new Float64Array(3);

/** SMD_Math::pseudo_inverse_SVD: M <- U diag(1/s) U^T with s <= 1e-16 treated as zero (in place). */
export const pseudoInverseSVD = (M: Mat3): void => {
  const U = piU, s = piS;
  svd3Into(M, U, s, piV);
  const invS = piInv;
  for (let r = 0; r < 3; r++) invS[r] = s[r] > 1.0e-16 ? 1 / s[r] : 0;
  const Ut = transpose3(U, piUt);
  const D = piD;
  D.fill(0);
  D[0] = invS[0]; D[4] = invS[1]; D[8] = invS[2];
  mul3(mul3(U, D, piT1), Ut, M);
};
const piU = mat3(), piV = mat3(), piUt = mat3(), piD = mat3(), piT1 = mat3();
const piS = new Float64Array(3), piInv = new Float64Array(3);
