/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/math_extra.cpp and src/math_extra.h
 * (quatvec, invquatvec, matvec, transpose_matvec, angmom_to_omega, exyz_to_q, q_to_exyz, quat_to_mat,
 * no_squish_rotate) and src/math_eigen.cpp / src/math_eigen_impl.h (MathEigen::jacobi3, the cyclic Jacobi
 * rotations of Andrew Jewett, Scripps Research, with SORT_INCREASING_EVALS).
 * Contributing author of src/math_extra.cpp: Mike Brown (SNL). Contributing author of src/math_eigen.cpp (the Jacobi
 * rotations): Andrew Jewett (Scripps Research).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Vectors are 3-element arrays, quaternions are (w, x, y, z) 4-element arrays, matrices are arrays of rows.
 * Every operation keeps LAMMPS's order of floating-point operations so the rigid-body trajectories match.
 */

export type V3 = number[];
export type Q4 = number[];

export const quatvec = (a: Q4, b: V3, c: number[]): void => {
  c[0] = -a[1] * b[0] - a[2] * b[1] - a[3] * b[2];
  c[1] = a[0] * b[0] + a[2] * b[2] - a[3] * b[1];
  c[2] = a[0] * b[1] + a[3] * b[0] - a[1] * b[2];
  c[3] = a[0] * b[2] + a[1] * b[1] - a[2] * b[0];
};

export const invquatvec = (a: Q4, b: number[], c: V3): void => {
  c[0] = -a[1] * b[0] + a[0] * b[1] + a[3] * b[2] - a[2] * b[3];
  c[1] = -a[2] * b[0] - a[3] * b[1] + a[0] * b[2] + a[1] * b[3];
  c[2] = -a[3] * b[0] + a[2] * b[1] - a[1] * b[2] + a[0] * b[3];
};

/** ans = [ex ey ez] * vec (columns ex, ey, ez). */
export const matvec = (ex: V3, ey: V3, ez: V3, vec: V3, ans: V3): void => {
  ans[0] = ex[0] * vec[0] + ey[0] * vec[1] + ez[0] * vec[2];
  ans[1] = ex[1] * vec[0] + ey[1] * vec[1] + ez[1] * vec[2];
  ans[2] = ex[2] * vec[0] + ey[2] * vec[1] + ez[2] * vec[2];
};

/** ans = [ex ey ez]^T * v. */
export const transposeMatvec = (ex: V3, ey: V3, ez: V3, v: V3, ans: V3): void => {
  ans[0] = ex[0] * v[0] + ex[1] * v[1] + ex[2] * v[2];
  ans[1] = ey[0] * v[0] + ey[1] * v[1] + ey[2] * v[2];
  ans[2] = ez[0] * v[0] + ez[1] * v[1] + ez[2] * v[2];
};

/** w = omega (space frame) from m = angular momentum (space frame), axes ex ey ez and principal moments idiag. */
export const angmomToOmega = (m: V3, ex: V3, ey: V3, ez: V3, idiag: V3, w: V3): void => {
  const wb = [0, 0, 0];
  wb[0] = idiag[0] === 0 ? 0 : (m[0] * ex[0] + m[1] * ex[1] + m[2] * ex[2]) / idiag[0];
  wb[1] = idiag[1] === 0 ? 0 : (m[0] * ey[0] + m[1] * ey[1] + m[2] * ey[2]) / idiag[1];
  wb[2] = idiag[2] === 0 ? 0 : (m[0] * ez[0] + m[1] * ez[1] + m[2] * ez[2]) / idiag[2];
  w[0] = wb[0] * ex[0] + wb[1] * ey[0] + wb[2] * ez[0];
  w[1] = wb[0] * ex[1] + wb[1] * ey[1] + wb[2] * ez[1];
  w[2] = wb[0] * ex[2] + wb[1] * ey[2] + wb[2] * ez[2];
};

const qnormalize = (q: Q4): void => {
  const norm = 1.0 / Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (let k = 0; k < 4; k++) q[k] *= norm;
};

/** Unit quaternion from the columns ex, ey, ez of a rotation matrix (exyz_to_q). */
export const exyzToQ = (ex: V3, ey: V3, ez: V3): Q4 => {
  const q: Q4 = [0, 0, 0, 0];
  const q0sq = 0.25 * (ex[0] + ey[1] + ez[2] + 1.0);
  const q1sq = q0sq - 0.5 * (ey[1] + ez[2]);
  const q2sq = q0sq - 0.5 * (ex[0] + ez[2]);
  const q3sq = q0sq - 0.5 * (ex[0] + ey[1]);
  if (q0sq >= 0.25) {
    q[0] = Math.sqrt(q0sq);
    q[1] = (ey[2] - ez[1]) / (4.0 * q[0]);
    q[2] = (ez[0] - ex[2]) / (4.0 * q[0]);
    q[3] = (ex[1] - ey[0]) / (4.0 * q[0]);
  } else if (q1sq >= 0.25) {
    q[1] = Math.sqrt(q1sq);
    q[0] = (ey[2] - ez[1]) / (4.0 * q[1]);
    q[2] = (ey[0] + ex[1]) / (4.0 * q[1]);
    q[3] = (ex[2] + ez[0]) / (4.0 * q[1]);
  } else if (q2sq >= 0.25) {
    q[2] = Math.sqrt(q2sq);
    q[0] = (ez[0] - ex[2]) / (4.0 * q[2]);
    q[1] = (ey[0] + ex[1]) / (4.0 * q[2]);
    q[3] = (ez[1] + ey[2]) / (4.0 * q[2]);
  } else if (q3sq >= 0.25) {
    q[3] = Math.sqrt(q3sq);
    q[0] = (ex[1] - ey[0]) / (4.0 * q[3]);
    q[1] = (ez[0] + ex[2]) / (4.0 * q[3]);
    q[2] = (ez[1] + ey[2]) / (4.0 * q[3]);
  }
  qnormalize(q);
  return q;
};

/** Space-frame axes ex, ey, ez from a unit quaternion (q_to_exyz). */
export const qToExyz = (q: Q4): { ex: V3; ey: V3; ez: V3 } => ({
  ex: [q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3], 2.0 * (q[1] * q[2] + q[0] * q[3]), 2.0 * (q[1] * q[3] - q[0] * q[2])],
  ey: [2.0 * (q[1] * q[2] - q[0] * q[3]), q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3], 2.0 * (q[2] * q[3] + q[0] * q[1])],
  ez: [2.0 * (q[1] * q[3] + q[0] * q[2]), 2.0 * (q[2] * q[3] - q[0] * q[1]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]],
});

/** Rotation matrix (rows) of a unit quaternion (quat_to_mat). */
export const quatToMat = (q: Q4): number[][] => {
  const w2 = q[0] * q[0], i2 = q[1] * q[1], j2 = q[2] * q[2], k2 = q[3] * q[3];
  const twoij = 2.0 * q[1] * q[2], twoik = 2.0 * q[1] * q[3], twojk = 2.0 * q[2] * q[3];
  const twoiw = 2.0 * q[1] * q[0], twojw = 2.0 * q[2] * q[0], twokw = 2.0 * q[3] * q[0];
  return [
    [w2 + i2 - j2 - k2, twoij - twokw, twojw + twoik],
    [twoij + twokw, w2 - i2 + j2 - k2, twojk - twoiw],
    [twoik - twojw, twojk + twoiw, w2 - i2 - j2 + k2],
  ];
};

/**
 * One NO_SQUISH free-rotor sub-step about body axis k (1, 2 or 3) on the conjugate momentum p and the
 * quaternion q, in place (no_squish_rotate). Miller et al., J. Chem. Phys. 116, 8649 (2002).
 */
export const noSquishRotate = (k: number, p: Q4, q: Q4, inertia: V3, dt: number): void => {
  const kq = [0, 0, 0, 0], kp = [0, 0, 0, 0];
  if (k === 1) {
    kq[0] = -q[1]; kp[0] = -p[1];
    kq[1] = q[0]; kp[1] = p[0];
    kq[2] = q[3]; kp[2] = p[3];
    kq[3] = -q[2]; kp[3] = -p[2];
  } else if (k === 2) {
    kq[0] = -q[2]; kp[0] = -p[2];
    kq[1] = -q[3]; kp[1] = -p[3];
    kq[2] = q[0]; kp[2] = p[0];
    kq[3] = q[1]; kp[3] = p[1];
  } else if (k === 3) {
    kq[0] = -q[3]; kp[0] = -p[3];
    kq[1] = q[2]; kp[1] = p[2];
    kq[2] = -q[1]; kp[2] = -p[1];
    kq[3] = q[0]; kp[3] = p[0];
  }
  let phi = p[0] * kq[0] + p[1] * kq[1] + p[2] * kq[2] + p[3] * kq[3];
  if (inertia[k - 1] === 0.0) phi = 0.0;
  else phi /= 4.0 * inertia[k - 1];
  const c = Math.cos(dt * phi), s = Math.sin(dt * phi);
  for (let m = 0; m < 4; m++) p[m] = c * p[m] + s * kp[m];
  for (let m = 0; m < 4; m++) q[m] = c * q[m] + s * kq[m];
};

/**
 * Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations, sorted by increasing eigenvalue
 * (MathEigen::jacobi3 with sort = 1). Returns the eigenvalues and the eigenvectors as columns.
 * Throws if the rotations do not converge (the source returns an error code).
 */
export const jacobi3 = (mat: number[][]): { eval: V3; evec: number[][] } => {
  const n = 3;
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) M[i][j] = mat[i][j];
  // E holds the eigenvectors in rows while rotating (ApplyRotLeft); transposed at the end
  const E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const maxIdx = [0, 0, 0];
  const maxEntryRow = (i: number): number => {
    let jmax = i + 1;
    for (let j = i + 2; j < n; j++) if (Math.abs(M[i][j]) > Math.abs(M[i][jmax])) jmax = j;
    return jmax;
  };
  for (let i = 0; i < n - 1; i++) maxIdx[i] = maxEntryRow(i);
  const maxIters = 50 * n * (n - 1) / 2;
  let c = 1, s = 0, t = 0;
  let converged = false;
  for (let it = 0; it < maxIters; it++) {
    // MaxEntry: the largest off-diagonal entry, scanned through the stored row maxima
    let i = 0, j = maxIdx[0];
    let maxE = Math.abs(M[i][j]);
    for (let r = 1; r < n - 1; r++) {
      const jr = maxIdx[r];
      if (Math.abs(M[r][jr]) > maxE) { maxE = Math.abs(M[r][jr]); i = r; j = jr; }
    }
    if (M[i][i] + M[i][j] === M[i][i] && M[j][j] + M[i][j] === M[j][j]) {
      M[i][j] = 0.0;
      maxIdx[i] = maxEntryRow(i);
    }
    if (M[i][j] === 0.0) { converged = true; break; }
    // CalcRot
    t = 1.0;
    const diff = M[j][j] - M[i][i];
    if (diff !== 0.0) {
      let kappa = diff;
      t = 0.0;
      const Mij = M[i][j];
      if (Mij !== 0.0) {
        kappa /= 2.0 * Mij;
        t = 1.0 / (Math.sqrt(1 + kappa * kappa) + Math.abs(kappa));
        if (kappa < 0.0) t = -t;
      }
    }
    c = 1.0 / Math.sqrt(1 + t * t);
    s = c * t;
    // ApplyRot
    M[i][i] -= t * M[i][j];
    M[j][j] += t * M[i][j];
    M[i][j] = 0.0;
    for (let w = 0; w < i; w++) {
      M[i][w] = M[w][i];
      M[w][i] = c * M[w][i] - s * M[w][j];
      if (i === maxIdx[w]) maxIdx[w] = maxEntryRow2(M, n, w);
      else if (Math.abs(M[w][i]) > Math.abs(M[w][maxIdx[w]])) maxIdx[w] = i;
    }
    for (let w = i + 1; w < j; w++) {
      M[w][i] = M[i][w];
      M[i][w] = c * M[i][w] - s * M[w][j];
    }
    for (let w = j + 1; w < n; w++) {
      M[w][i] = M[i][w];
      M[i][w] = c * M[i][w] - s * M[j][w];
    }
    maxIdx[i] = maxEntryRow2(M, n, i);
    for (let w = 0; w < i; w++) {
      M[w][j] = s * M[i][w] + c * M[w][j];
      if (j === maxIdx[w]) maxIdx[w] = maxEntryRow2(M, n, w);
      else if (Math.abs(M[w][j]) > Math.abs(M[w][maxIdx[w]])) maxIdx[w] = j;
    }
    for (let w = i + 1; w < j; w++) {
      M[w][j] = s * M[w][i] + c * M[w][j];
      if (j === maxIdx[w]) maxIdx[w] = maxEntryRow2(M, n, w);
      else if (Math.abs(M[w][j]) > Math.abs(M[w][maxIdx[w]])) maxIdx[w] = j;
    }
    for (let w = j + 1; w < n; w++) M[j][w] = s * M[w][i] + c * M[j][w];
    maxIdx[j] = maxEntryRow2(M, n, j);
    // ApplyRotLeft
    for (let v = 0; v < n; v++) {
      const Eiv = E[i][v];
      E[i][v] = c * E[i][v] - s * E[j][v];
      E[j][v] = s * Eiv + c * E[j][v];
    }
  }
  if (!converged) throw new Error('Insufficient Jacobi rotations for rigid body');
  const ev: V3 = [M[0][0], M[1][1], M[2][2]];
  // SortRows, increasing eigenvalues
  for (let a = 0; a < n - 1; a++) {
    let imax = a;
    for (let b = a + 1; b < n; b++) if (ev[b] < ev[imax]) imax = b;
    [ev[a], ev[imax]] = [ev[imax], ev[a]];
    [E[a], E[imax]] = [E[imax], E[a]];
  }
  const evec = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) evec[a][b] = E[b][a];
  return { eval: ev, evec };
};

/** MaxEntryRow on the current matrix (the source's MaxEntryRow, with its out-of-range result for the last row unused). */
const maxEntryRow2 = (M: number[][], n: number, i: number): number => {
  let jmax = i + 1;
  for (let j = i + 2; j < n; j++) if (Math.abs(M[i][j]) > Math.abs(M[i][jmax])) jmax = j;
  return jmax;
};
