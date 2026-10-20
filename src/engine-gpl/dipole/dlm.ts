/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the update dipole/dlm loop of src/fix_nve_sphere.cpp (FixNVESphere::initial_integrate), src/fix_nh_sphere.cpp
 * (FixNHSphere::nve_x) and src/BPM/fix_nve_bpm_sphere.cpp (FixNVEBPMSphere::initial_integrate) at commit b332f28 (the
 * three carry the same loop), with MathExtra::BuildRxMatrix, BuildRyMatrix, BuildRzMatrix (src/math_extra.cpp) and the
 * inline matvec, transpose_matvec and transpose_times3 (src/math_extra.h). Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License (the LAMMPS file headers). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The Dullweber-Leimkuhler-McLachlan splitting: the dipole defines a body frame Q whose z axis is mu/|mu|; the angular
 * velocity in that frame is rotated by five small-angle rotations about x, y, z (double), y, x, each a Cayley
 * approximation of angle h w_k, and Q is carried along; omega and mu are then read back from the final frame. The
 * operation order of every step is kept, so the results match the native code bit for bit.
 */
import { newMat3, transposeTimes3, type Mat3 } from '../asphere/math_extra';

/** MathExtra::BuildRxMatrix / BuildRyMatrix / BuildRzMatrix: the small-angle rotation about axis 0, 1 or 2. */
const buildR = (R: Mat3, axis: 0 | 1 | 2, angle: number): void => {
  const angleSq = angle * angle;
  const c = (1.0 - angleSq * 0.25) / (1.0 + angleSq * 0.25);
  const s = angle / (1.0 + angleSq * 0.25);
  if (axis === 0) {
    R[0][0] = 1.0; R[0][1] = 0.0; R[0][2] = 0.0;
    R[1][0] = 0.0; R[1][1] = c; R[1][2] = -s;
    R[2][0] = 0.0; R[2][1] = s; R[2][2] = c;
  } else if (axis === 1) {
    R[0][0] = c; R[0][1] = 0.0; R[0][2] = s;
    R[1][0] = 0.0; R[1][1] = 1.0; R[1][2] = 0.0;
    R[2][0] = -s; R[2][1] = 0.0; R[2][2] = c;
  } else {
    R[0][0] = c; R[0][1] = -s; R[0][2] = 0.0;
    R[1][0] = s; R[1][1] = c; R[1][2] = 0.0;
    R[2][0] = 0.0; R[2][1] = 0.0; R[2][2] = 1.0;
  }
};

/** MathExtra::matvec: ans = m v. */
const matvec = (m: Mat3, v: Float64Array, ans: Float64Array): void => {
  ans[0] = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2];
  ans[1] = m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2];
  ans[2] = m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2];
};

/** MathExtra::transpose_matvec: ans = m^T v. */
const transposeMatvec = (m: Mat3, v: Float64Array, ans: Float64Array): void => {
  ans[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2];
  ans[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2];
  ans[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2];
};

/**
 * The update dipole/dlm loop: for each atom i < n in the group (mask & groupBit) with mu[4i+3] > 0, rotates omega
 * (3 per atom) and the dipole mu (4 per atom, the length last) by the DLM splitting over dtf / ftm2v (dtf =
 * dt/2 ftm2v, the fix's half-step factor).
 */
export const dlmUpdate = (n: number, mask: ArrayLike<number>, groupBit: number, omega: Float64Array, mu: Float64Array, dtf: number, ftm2v: number): void => {
  const w = new Float64Array(3), wTemp = new Float64Array(3);
  const Q = newMat3(), QTemp = newMat3(), R = newMat3();
  for (let i = 0; i < n; i++) {
    if (!(mask[i] & groupBit) || !(mu[4 * i + 3] > 0.0)) continue;
    // Q rotates the space frame into the body frame, in which the unit dipole lies along z
    const invLenMu = 1.0 / mu[4 * i + 3];
    const a0 = mu[4 * i] * invLenMu, a1 = mu[4 * i + 1] * invLenMu, a2 = mu[4 * i + 2] * invLenMu;
    const s2 = a0 * a0 + a1 * a1;
    if (s2 !== 0.0) {
      const scale = (1.0 - a2) / s2;
      Q[0][0] = 1.0 - scale * a0 * a0; Q[0][1] = -scale * a0 * a1; Q[0][2] = -a0;
      Q[1][0] = -scale * a0 * a1; Q[1][1] = 1.0 - scale * a1 * a1; Q[1][2] = -a1;
      Q[2][0] = a0; Q[2][1] = a1; Q[2][2] = 1.0 - scale * (a0 * a0 + a1 * a1);
    } else {
      // parallel to z: I or -I
      Q[0][0] = 1.0 / a2; Q[0][1] = 0.0; Q[0][2] = 0.0;
      Q[1][0] = 0.0; Q[1][1] = 1.0 / a2; Q[1][2] = 0.0;
      Q[2][0] = 0.0; Q[2][1] = 0.0; Q[2][2] = 1.0 / a2;
    }
    w[0] = omega[3 * i]; w[1] = omega[3 * i + 1]; w[2] = omega[3 * i + 2];
    matvec(Q, w, wTemp);
    // R1 about x
    buildR(R, 0, dtf / ftm2v * wTemp[0]);
    matvec(R, wTemp, w);
    transposeTimes3(R, Q, QTemp);
    // R2 about y
    buildR(R, 1, dtf / ftm2v * w[1]);
    matvec(R, w, wTemp);
    transposeTimes3(R, QTemp, Q);
    // R3 about z, the full step
    buildR(R, 2, 2.0 * dtf / ftm2v * wTemp[2]);
    matvec(R, wTemp, w);
    transposeTimes3(R, Q, QTemp);
    // R4 about y
    buildR(R, 1, dtf / ftm2v * w[1]);
    matvec(R, w, wTemp);
    transposeTimes3(R, QTemp, Q);
    // R5 about x
    buildR(R, 0, dtf / ftm2v * wTemp[0]);
    matvec(R, wTemp, w);
    transposeTimes3(R, Q, QTemp);
    // back to the space frame: omega = Q^T w, mu = Q^T (0, 0, 1) |mu|
    transposeMatvec(QTemp, w, wTemp);
    omega[3 * i] = wTemp[0]; omega[3 * i + 1] = wTemp[1]; omega[3 * i + 2] = wTemp[2];
    mu[4 * i] = QTemp[2][0] * mu[4 * i + 3];
    mu[4 * i + 1] = QTemp[2][1] * mu[4 * i + 3];
    mu[4 * i + 2] = QTemp[2][2] * mu[4 * i + 3];
  }
};
