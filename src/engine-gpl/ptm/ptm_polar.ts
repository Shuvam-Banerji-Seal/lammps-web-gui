/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_polar.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

import { quaternion_to_rotation_matrix } from './ptm_quat';

// std::max(a, b) as libstdc++ defines it: (a < b) ? b : a
function std_max(a: number, b: number): number {
        return a < b ? b : a;
}

export function matmul_3x3(A: number[], x: number[], b: number[]): void {
        b[0] = A[0] * x[0] + A[1] * x[3] + A[2] * x[6];
        b[3] = A[3] * x[0] + A[4] * x[3] + A[5] * x[6];
        b[6] = A[6] * x[0] + A[7] * x[3] + A[8] * x[6];

        b[1] = A[0] * x[1] + A[1] * x[4] + A[2] * x[7];
        b[4] = A[3] * x[1] + A[4] * x[4] + A[5] * x[7];
        b[7] = A[6] * x[1] + A[7] * x[4] + A[8] * x[7];

        b[2] = A[0] * x[2] + A[1] * x[5] + A[2] * x[8];
        b[5] = A[3] * x[2] + A[4] * x[5] + A[5] * x[8];
        b[8] = A[6] * x[2] + A[7] * x[5] + A[8] * x[8];
}

export function matrix_determinant_3x3(A: number[]): number {
        return A[0] * (A[4] * A[8] - A[5] * A[7])
                - A[1] * (A[3] * A[8] - A[5] * A[6])
                + A[2] * (A[3] * A[7] - A[4] * A[6]);
}

export function flip_matrix(A: number[]): void {
        for (let i = 0; i < 9; i++) {
                A[i] = -A[i];
        }
}

// Writes the optimal quaternion into qopt. The C out-parameter p_nrmsdsq is
// returned as nrmsdsq (it is set on every path, as in the C code); the C bool
// return value is returned as ok.
export function optimal_quaternion(A: number[], polar: boolean, E0: number, qopt: number[]): { ok: boolean; nrmsdsq: number } {
        const evecprec = 1e-6;
        const evalprec = 1e-11;

        const Sxx = A[0], Sxy = A[1], Sxz = A[2],
                Syx = A[3], Syy = A[4], Syz = A[5],
                Szx = A[6], Szy = A[7], Szz = A[8];

        const Sxx2 = Sxx * Sxx, Syy2 = Syy * Syy, Szz2 = Szz * Szz,
                Sxy2 = Sxy * Sxy, Syz2 = Syz * Syz, Sxz2 = Sxz * Sxz,
                Syx2 = Syx * Syx, Szy2 = Szy * Szy, Szx2 = Szx * Szx;

        const fnorm_squared = Sxx2 + Syy2 + Szz2 + Sxy2 + Syz2 + Sxz2 + Syx2 + Szy2 + Szx2;

        const SyzSzymSyySzz2 = 2.0 * (Syz * Szy - Syy * Szz);
        const Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2;
        const SxzpSzx = Sxz + Szx;
        const SyzpSzy = Syz + Szy;
        const SxypSyx = Sxy + Syx;
        const SyzmSzy = Syz - Szy;
        const SxzmSzx = Sxz - Szx;
        const SxymSyx = Sxy - Syx;
        const SxxpSyy = Sxx + Syy;
        const SxxmSyy = Sxx - Syy;
        const Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2;

        const C = [0, 0, 0];
        C[0] = Sxy2Sxz2Syx2Szx2 * Sxy2Sxz2Syx2Szx2
          + (Sxx2Syy2Szz2Syz2Szy2 + SyzSzymSyySzz2) * (Sxx2Syy2Szz2Syz2Szy2 - SyzSzymSyySzz2)
          + (-SxzpSzx * SyzmSzy + SxymSyx * (SxxmSyy - Szz)) * (-SxzmSzx * SyzpSzy + SxymSyx * (SxxmSyy + Szz))
          + (-SxzpSzx * SyzpSzy - SxypSyx * (SxxpSyy - Szz)) * (-SxzmSzx * SyzmSzy - SxypSyx * (SxxpSyy + Szz))
          + (SxypSyx * SyzpSzy + SxzpSzx * (SxxmSyy + Szz)) * (-SxymSyx * SyzmSzy + SxzpSzx * (SxxpSyy + Szz))
          + (SxypSyx * SyzmSzy + SxzmSzx * (SxxmSyy - Szz)) * (-SxymSyx * SyzpSzy + SxzmSzx * (SxxpSyy - Szz));

        C[1] = 8.0 * (Sxx * Syz * Szy + Syy * Szx * Sxz + Szz * Sxy * Syx - Sxx * Syy * Szz - Syz * Szx * Sxy - Szy * Syx * Sxz);
        C[2] = -2.0 * fnorm_squared;

        //Newton-Raphson
        let mxEigenV = polar ? Math.sqrt(3 * fnorm_squared) : E0;
        if (mxEigenV > evalprec) {
                for (let i = 0; i < 50; i++) {
                        const oldg = mxEigenV;
                        const x2 = mxEigenV * mxEigenV;
                        const b = (x2 + C[2]) * mxEigenV;
                        const a = b + C[1];
                        const delta = ((a * mxEigenV + C[0]) / (2 * x2 * mxEigenV + b + a));
                        mxEigenV -= delta;
                        if (Math.abs(mxEigenV - oldg) < Math.abs(evalprec * mxEigenV)) {
                                break;
                        }
                }
        } else {
                mxEigenV = 0.0;
        }

        const nrmsdsq = std_max(0.0, 2.0 * (E0 - mxEigenV));

        const a11 = SxxpSyy + Szz - mxEigenV;
        const a12 = SyzmSzy;
        const a13 = -SxzmSzx;
        const a14 = SxymSyx;

        const a21 = SyzmSzy;
        const a22 = SxxmSyy - Szz - mxEigenV;
        const a23 = SxypSyx;
        const a24 = SxzpSzx;

        const a31 = a13;
        const a32 = a23;
        const a33 = Syy - Sxx - Szz - mxEigenV;
        const a34 = SyzpSzy;

        const a41 = a14;
        const a42 = a24;
        const a43 = a34;
        const a44 = Szz - SxxpSyy - mxEigenV;

        const a3344_4334 = a33 * a44 - a43 * a34;
        const a3244_4234 = a32 * a44 - a42 * a34;
        const a3243_4233 = a32 * a43 - a42 * a33;
        const a3143_4133 = a31 * a43 - a41 * a33;
        const a3144_4134 = a31 * a44 - a41 * a34;
        const a3142_4132 = a31 * a42 - a41 * a32;
        const a1324_1423 = a13 * a24 - a14 * a23;
        const a1224_1422 = a12 * a24 - a14 * a22;
        const a1223_1322 = a12 * a23 - a13 * a22;
        const a1124_1421 = a11 * a24 - a14 * a21;
        const a1123_1321 = a11 * a23 - a13 * a21;
        const a1122_1221 = a11 * a22 - a12 * a21;

        const q: number[][] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
        q[0][0] = a12 * a3344_4334 - a13 * a3244_4234 + a14 * a3243_4233;
        q[0][1] = -a11 * a3344_4334 + a13 * a3144_4134 - a14 * a3143_4133;
        q[0][2] = a11 * a3244_4234 - a12 * a3144_4134 + a14 * a3142_4132;
        q[0][3] = -a11 * a3243_4233 + a12 * a3143_4133 - a13 * a3142_4132;

        q[1][0] = a22 * a3344_4334 - a23 * a3244_4234 + a24 * a3243_4233;
        q[1][1] = -a21 * a3344_4334 + a23 * a3144_4134 - a24 * a3143_4133;
        q[1][2] = a21 * a3244_4234 - a22 * a3144_4134 + a24 * a3142_4132;
        q[1][3] = -a21 * a3243_4233 + a22 * a3143_4133 - a23 * a3142_4132;

        q[2][0] = a32 * a1324_1423 - a33 * a1224_1422 + a34 * a1223_1322;
        q[2][1] = -a31 * a1324_1423 + a33 * a1124_1421 - a34 * a1123_1321;
        q[2][2] = a31 * a1224_1422 - a32 * a1124_1421 + a34 * a1122_1221;
        q[2][3] = -a31 * a1223_1322 + a32 * a1123_1321 - a33 * a1122_1221;

        q[3][0] = a42 * a1324_1423 - a43 * a1224_1422 + a44 * a1223_1322;
        q[3][1] = -a41 * a1324_1423 + a43 * a1124_1421 - a44 * a1123_1321;
        q[3][2] = a41 * a1224_1422 - a42 * a1124_1421 + a44 * a1122_1221;
        q[3][3] = -a41 * a1223_1322 + a42 * a1123_1321 - a43 * a1122_1221;

        const qsqr = [0, 0, 0, 0];
        for (let i = 0; i < 4; i++) {
                qsqr[i] = q[i][0] * q[i][0] + q[i][1] * q[i][1] + q[i][2] * q[i][2] + q[i][3] * q[i][3];
        }

        let bi = 0;
        let max = 0;
        for (let i = 0; i < 4; i++) {
                if (qsqr[i] > max) {
                        bi = i;
                        max = qsqr[i];
                }
        }

        let too_small = false;
        if (qsqr[bi] < evecprec) {
                //if qsqr is still too small, return the identity rotation.
                q[bi][0] = 1;
                q[bi][1] = 0;
                q[bi][2] = 0;
                q[bi][3] = 0;
                too_small = true;
        } else {
                const normq = Math.sqrt(qsqr[bi]);
                q[bi][0] /= normq;
                q[bi][1] /= normq;
                q[bi][2] /= normq;
                q[bi][3] /= normq;
        }

        for (let i = 0; i < 4; i++) {
                qopt[i] = q[bi][i];
        }
        return { ok: !too_small, nrmsdsq };
}

export function polar_decomposition_3x3(_A: number[], right_sided: boolean, U: number[], P: number[]): number {
        const A = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        for (let i = 0; i < 9; i++) {
                A[i] = _A[i];
        }

        const det = matrix_determinant_3x3(A);
        if (det < 0) {
                flip_matrix(A);
        }

        const q = [0, 0, 0, 0];
        optimal_quaternion(A, true, -1, q);
        q[0] = -q[0];
        quaternion_to_rotation_matrix(q, U);

        if (det < 0) {
                flip_matrix(U);
        }

        const UT = [U[0], U[3], U[6], U[1], U[4], U[7], U[2], U[5], U[8]];

        if (right_sided) {
                matmul_3x3(UT, _A, P);
        } else {
                matmul_3x3(_A, UT, P);
        }

        return 0;
}

export function InnerProduct(A: number[], num: number, coords1: number[][], coords2: number[][], permutation: number[]): void {
        A[0] = A[1] = A[2] = A[3] = A[4] = A[5] = A[6] = A[7] = A[8] = 0.0;

        for (let i = 0; i < num; ++i) {
                const x1 = coords1[i][0];
                const y1 = coords1[i][1];
                const z1 = coords1[i][2];

                const x2 = coords2[permutation[i]][0];
                const y2 = coords2[permutation[i]][1];
                const z2 = coords2[permutation[i]][2];

                A[0] += x1 * x2;
                A[1] += x1 * y2;
                A[2] += x1 * z2;

                A[3] += y1 * x2;
                A[4] += y1 * y2;
                A[5] += y1 * z2;

                A[6] += z1 * x2;
                A[7] += z1 * y2;
                A[8] += z1 * z2;
        }
}

export function FastCalcRMSDAndRotation(A: number[], E0: number, q: number[], U: number[]): { ret: number; nrmsdsq: number } {
        const r = optimal_quaternion(A, false, E0, q);
        quaternion_to_rotation_matrix(q, U);
        return { ret: 0, nrmsdsq: r.nrmsdsq };
}
