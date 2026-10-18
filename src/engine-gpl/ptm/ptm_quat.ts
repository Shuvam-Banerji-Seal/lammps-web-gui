/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_quat.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { acos as libmAcos } from '../libm/acos';

// Generator tables (ptm_quat.cpp), one row per symmetry operation, [w, x, y, z].
const generator_cubic: number[][] = [
        [          1,          0,          0,          0 ],
        [  Math.sqrt(2.0)/2,  Math.sqrt(2.0)/2,          0,          0 ],
        [  Math.sqrt(2.0)/2,          0,  Math.sqrt(2.0)/2,          0 ],
        [  Math.sqrt(2.0)/2,          0,          0,  Math.sqrt(2.0)/2 ],
        [  Math.sqrt(2.0)/2,          0,          0, -Math.sqrt(2.0)/2 ],
        [  Math.sqrt(2.0)/2,          0, -Math.sqrt(2.0)/2,          0 ],
        [  Math.sqrt(2.0)/2, -Math.sqrt(2.0)/2,         -0,         -0 ],
        [        0.5,        0.5,        0.5,        0.5 ],
        [        0.5,        0.5,        0.5,       -0.5 ],
        [        0.5,        0.5,       -0.5,        0.5 ],
        [        0.5,        0.5,       -0.5,       -0.5 ],
        [        0.5,       -0.5,        0.5,        0.5 ],
        [        0.5,       -0.5,        0.5,       -0.5 ],
        [        0.5,       -0.5,       -0.5,        0.5 ],
        [        0.5,       -0.5,       -0.5,       -0.5 ],
        [          0,          1,          0,          0 ],
        [          0,  Math.sqrt(2.0)/2,  Math.sqrt(2.0)/2,          0 ],
        [          0,  Math.sqrt(2.0)/2,          0,  Math.sqrt(2.0)/2 ],
        [          0,  Math.sqrt(2.0)/2,          0, -Math.sqrt(2.0)/2 ],
        [          0,  Math.sqrt(2.0)/2, -Math.sqrt(2.0)/2,          0 ],
        [          0,          0,          1,          0 ],
        [          0,          0,  Math.sqrt(2.0)/2,  Math.sqrt(2.0)/2 ],
        [          0,          0,  Math.sqrt(2.0)/2, -Math.sqrt(2.0)/2 ],
        [          0,          0,          0,          1 ],
];

const generator_diamond_cubic: number[][] = [
        [    1,    0,    0,    0 ],
        [  0.5,  0.5,  0.5,  0.5 ],
        [  0.5,  0.5,  0.5, -0.5 ],
        [  0.5,  0.5, -0.5,  0.5 ],
        [  0.5,  0.5, -0.5, -0.5 ],
        [  0.5, -0.5,  0.5,  0.5 ],
        [  0.5, -0.5,  0.5, -0.5 ],
        [  0.5, -0.5, -0.5,  0.5 ],
        [  0.5, -0.5, -0.5, -0.5 ],
        [    0,    1,    0,    0 ],
        [    0,    0,    1,    0 ],
        [    0,    0,    0,    1 ],
];

const generator_hcp: number[][] = [
        [          1,          0,          0,          0 ],
        [        0.5,          0,          0,  Math.sqrt(3.0)/2 ],
        [        0.5,          0,          0, -Math.sqrt(3.0)/2 ],
        [          0,  Math.sqrt(3.0)/2,        0.5,          0 ],
        [          0,  Math.sqrt(3.0)/2,       -0.5,          0 ],
        [          0,          0,          1,          0 ],
];


const generator_hcp_conventional: number[][] = [
        [          1,          0,          0,          0 ],
        [  Math.sqrt(3.0)/2,          0,          0,        0.5 ],
        [  Math.sqrt(3.0)/2,          0,          0,       -0.5 ],
        [        0.5,          0,          0,  Math.sqrt(3.0)/2 ],
        [        0.5,          0,          0, -Math.sqrt(3.0)/2 ],
        [          0,          1,          0,          0 ],
        [          0,  Math.sqrt(3.0)/2,        0.5,          0 ],
        [          0,  Math.sqrt(3.0)/2,       -0.5,          0 ],
        [          0,        0.5,  Math.sqrt(3.0)/2,          0 ],
        [          0,        0.5, -Math.sqrt(3.0)/2,          0 ],
        [          0,          0,          1,          0 ],
        [          0,          0,          0,          1 ],
];

const generator_diamond_hexagonal: number[][] = [
        [          1,          0,          0,          0 ],
        [        0.5,          0,          0,  Math.sqrt(3.0)/2 ],
        [        0.5,          0,          0, -Math.sqrt(3.0)/2 ],
];

const generator_icosahedral: number[][] = [
        [                        1,                        0,                        0,                        0 ],
        [            (1+Math.sqrt(5.0))/4,                      0.5,   Math.sqrt(25-10*Math.sqrt(5.0))/10,   Math.sqrt(50-10*Math.sqrt(5.0))/20 ],
        [            (1+Math.sqrt(5.0))/4,                      0.5,  -Math.sqrt(25-10*Math.sqrt(5.0))/10,  -Math.sqrt(50-10*Math.sqrt(5.0))/20 ],
        [            (1+Math.sqrt(5.0))/4,            1/(1+Math.sqrt(5.0)),   Math.sqrt(10*Math.sqrt(5.0)+50)/20,  -Math.sqrt(50-10*Math.sqrt(5.0))/20 ],
        [            (1+Math.sqrt(5.0))/4,            1/(1+Math.sqrt(5.0)),  -Math.sqrt(10*Math.sqrt(5.0)+50)/20,   Math.sqrt(50-10*Math.sqrt(5.0))/20 ],
        [            (1+Math.sqrt(5.0))/4,                        0,   Math.sqrt(50-10*Math.sqrt(5.0))/10,   Math.sqrt(50-10*Math.sqrt(5.0))/20 ],
        [            (1+Math.sqrt(5.0))/4,                        0,                        0,     Math.sqrt(5./8-Math.sqrt(5.0)/8) ],
        [            (1+Math.sqrt(5.0))/4,                        0,                        0,    -Math.sqrt(5./8-Math.sqrt(5.0)/8) ],
        [            (1+Math.sqrt(5.0))/4,                        0,  -Math.sqrt(50-10*Math.sqrt(5.0))/10,  -Math.sqrt(50-10*Math.sqrt(5.0))/20 ],
        [            (1+Math.sqrt(5.0))/4,           -1/(1+Math.sqrt(5.0)),   Math.sqrt(10*Math.sqrt(5.0)+50)/20,  -Math.sqrt(50-10*Math.sqrt(5.0))/20 ],
        [            (1+Math.sqrt(5.0))/4,           -1/(1+Math.sqrt(5.0)),  -Math.sqrt(10*Math.sqrt(5.0)+50)/20,   Math.sqrt(50-10*Math.sqrt(5.0))/20 ],
        [            (1+Math.sqrt(5.0))/4,                     -0.5,   Math.sqrt(25-10*Math.sqrt(5.0))/10,   Math.sqrt(50-10*Math.sqrt(5.0))/20 ],
        [            (1+Math.sqrt(5.0))/4,                     -0.5,  -Math.sqrt(25-10*Math.sqrt(5.0))/10,  -Math.sqrt(50-10*Math.sqrt(5.0))/20 ],
        [                      0.5,            (1+Math.sqrt(5.0))/4,   Math.sqrt(50-10*Math.sqrt(5.0))/20,  -Math.sqrt(25-10*Math.sqrt(5.0))/10 ],
        [                      0.5,            (1+Math.sqrt(5.0))/4,  -Math.sqrt(50-10*Math.sqrt(5.0))/20,   Math.sqrt(25-10*Math.sqrt(5.0))/10 ],
        [                      0.5,                      0.5,   Math.sqrt((5+2*Math.sqrt(5.0))/20),   Math.sqrt(25-10*Math.sqrt(5.0))/10 ],
        [                      0.5,                      0.5,   Math.sqrt(25-10*Math.sqrt(5.0))/10,  -Math.sqrt((5+2*Math.sqrt(5.0))/20) ],
        [                      0.5,                      0.5,  -Math.sqrt(25-10*Math.sqrt(5.0))/10,   Math.sqrt((5+2*Math.sqrt(5.0))/20) ],
        [                      0.5,                      0.5,  -Math.sqrt((5+2*Math.sqrt(5.0))/20),  -Math.sqrt(25-10*Math.sqrt(5.0))/10 ],
        [                      0.5,            1/(1+Math.sqrt(5.0)),   Math.sqrt(10*Math.sqrt(5.0)+50)/20,   Math.sqrt((5+2*Math.sqrt(5.0))/20) ],
        [                      0.5,            1/(1+Math.sqrt(5.0)),  -Math.sqrt(10*Math.sqrt(5.0)+50)/20,  -Math.sqrt((5+2*Math.sqrt(5.0))/20) ],
        [                      0.5,                        0,     Math.sqrt((5+Math.sqrt(5.0))/10),  -Math.sqrt(25-10*Math.sqrt(5.0))/10 ],
        [                      0.5,                        0,   Math.sqrt(50-10*Math.sqrt(5.0))/10,  -Math.sqrt((5+2*Math.sqrt(5.0))/20) ],
        [                      0.5,                        0,  -Math.sqrt(50-10*Math.sqrt(5.0))/10,   Math.sqrt((5+2*Math.sqrt(5.0))/20) ],
        [                      0.5,                        0,    -Math.sqrt((5+Math.sqrt(5.0))/10),   Math.sqrt(25-10*Math.sqrt(5.0))/10 ],
        [                      0.5,           -1/(1+Math.sqrt(5.0)),   Math.sqrt(10*Math.sqrt(5.0)+50)/20,   Math.sqrt((5+2*Math.sqrt(5.0))/20) ],
        [                      0.5,           -1/(1+Math.sqrt(5.0)),  -Math.sqrt(10*Math.sqrt(5.0)+50)/20,  -Math.sqrt((5+2*Math.sqrt(5.0))/20) ],
        [                      0.5,                     -0.5,   Math.sqrt((5+2*Math.sqrt(5.0))/20),   Math.sqrt(25-10*Math.sqrt(5.0))/10 ],
        [                      0.5,                     -0.5,   Math.sqrt(25-10*Math.sqrt(5.0))/10,  -Math.sqrt((5+2*Math.sqrt(5.0))/20) ],
        [                      0.5,                     -0.5,  -Math.sqrt(25-10*Math.sqrt(5.0))/10,   Math.sqrt((5+2*Math.sqrt(5.0))/20) ],
        [                      0.5,                     -0.5,  -Math.sqrt((5+2*Math.sqrt(5.0))/20),  -Math.sqrt(25-10*Math.sqrt(5.0))/10 ],
        [                      0.5,           -(1+Math.sqrt(5.0))/4,   Math.sqrt(50-10*Math.sqrt(5.0))/20,  -Math.sqrt(25-10*Math.sqrt(5.0))/10 ],
        [                      0.5,           -(1+Math.sqrt(5.0))/4,  -Math.sqrt(50-10*Math.sqrt(5.0))/20,   Math.sqrt(25-10*Math.sqrt(5.0))/10 ],
        [            1/(1+Math.sqrt(5.0)),            (1+Math.sqrt(5.0))/4,   Math.sqrt(50-10*Math.sqrt(5.0))/20,   Math.sqrt(10*Math.sqrt(5.0)+50)/20 ],
        [            1/(1+Math.sqrt(5.0)),            (1+Math.sqrt(5.0))/4,  -Math.sqrt(50-10*Math.sqrt(5.0))/20,  -Math.sqrt(10*Math.sqrt(5.0)+50)/20 ],
        [            1/(1+Math.sqrt(5.0)),                      0.5,   Math.sqrt((5+2*Math.sqrt(5.0))/20),  -Math.sqrt(10*Math.sqrt(5.0)+50)/20 ],
        [            1/(1+Math.sqrt(5.0)),                      0.5,  -Math.sqrt((5+2*Math.sqrt(5.0))/20),   Math.sqrt(10*Math.sqrt(5.0)+50)/20 ],
        [            1/(1+Math.sqrt(5.0)),                        0,     Math.sqrt((5+Math.sqrt(5.0))/10),   Math.sqrt(10*Math.sqrt(5.0)+50)/20 ],
        [            1/(1+Math.sqrt(5.0)),                        0,                        0,  Math.sqrt(1-1/(2*Math.sqrt(5.0)+6)) ],
        [            1/(1+Math.sqrt(5.0)),                        0,                        0, -Math.sqrt(1-1/(2*Math.sqrt(5.0)+6)) ],
        [            1/(1+Math.sqrt(5.0)),                        0,    -Math.sqrt((5+Math.sqrt(5.0))/10),  -Math.sqrt(10*Math.sqrt(5.0)+50)/20 ],
        [            1/(1+Math.sqrt(5.0)),                     -0.5,   Math.sqrt((5+2*Math.sqrt(5.0))/20),  -Math.sqrt(10*Math.sqrt(5.0)+50)/20 ],
        [            1/(1+Math.sqrt(5.0)),                     -0.5,  -Math.sqrt((5+2*Math.sqrt(5.0))/20),   Math.sqrt(10*Math.sqrt(5.0)+50)/20 ],
        [            1/(1+Math.sqrt(5.0)),           -(1+Math.sqrt(5.0))/4,   Math.sqrt(50-10*Math.sqrt(5.0))/20,   Math.sqrt(10*Math.sqrt(5.0)+50)/20 ],
        [            1/(1+Math.sqrt(5.0)),           -(1+Math.sqrt(5.0))/4,  -Math.sqrt(50-10*Math.sqrt(5.0))/20,  -Math.sqrt(10*Math.sqrt(5.0)+50)/20 ],
        [                        0,                        1,                        0,                        0 ],
        [                        0,            (1+Math.sqrt(5.0))/4,     Math.sqrt(5./8-Math.sqrt(5.0)/8),                        0 ],
        [                        0,            (1+Math.sqrt(5.0))/4,   Math.sqrt(50-10*Math.sqrt(5.0))/20,  -Math.sqrt(50-10*Math.sqrt(5.0))/10 ],
        [                        0,            (1+Math.sqrt(5.0))/4,  -Math.sqrt(50-10*Math.sqrt(5.0))/20,   Math.sqrt(50-10*Math.sqrt(5.0))/10 ],
        [                        0,            (1+Math.sqrt(5.0))/4,    -Math.sqrt(5./8-Math.sqrt(5.0)/8),                        0 ],
        [                        0,                      0.5,   Math.sqrt((5+2*Math.sqrt(5.0))/20),   Math.sqrt(50-10*Math.sqrt(5.0))/10 ],
        [                        0,                      0.5,   Math.sqrt(25-10*Math.sqrt(5.0))/10,     Math.sqrt((5+Math.sqrt(5.0))/10) ],
        [                        0,                      0.5,  -Math.sqrt(25-10*Math.sqrt(5.0))/10,    -Math.sqrt((5+Math.sqrt(5.0))/10) ],
        [                        0,                      0.5,  -Math.sqrt((5+2*Math.sqrt(5.0))/20),  -Math.sqrt(50-10*Math.sqrt(5.0))/10 ],
        [                        0,            1/(1+Math.sqrt(5.0)),  Math.sqrt(1-1/(2*Math.sqrt(5.0)+6)),                        0 ],
        [                        0,            1/(1+Math.sqrt(5.0)),   Math.sqrt(10*Math.sqrt(5.0)+50)/20,    -Math.sqrt((5+Math.sqrt(5.0))/10) ],
        [                        0,            1/(1+Math.sqrt(5.0)),  -Math.sqrt(10*Math.sqrt(5.0)+50)/20,     Math.sqrt((5+Math.sqrt(5.0))/10) ],
        [                        0,            1/(1+Math.sqrt(5.0)), -Math.sqrt(1-1/(2*Math.sqrt(5.0)+6)),                        0 ],
        [                        0,                        0,     Math.sqrt((5+Math.sqrt(5.0))/10),  -Math.sqrt(50-10*Math.sqrt(5.0))/10 ],
        [                        0,                        0,   Math.sqrt(50-10*Math.sqrt(5.0))/10,     Math.sqrt((5+Math.sqrt(5.0))/10) ],
];



function SIGN(x: number): number {
        return x >= 0 ? 1 : -1;
}

// std::max(a, b) as libstdc++ defines it: (a < b) ? b : a
function std_max(a: number, b: number): number {
        return a < b ? b : a;
}

// std::min(a, b) as libstdc++ defines it: (b < a) ? b : a
function std_min(a: number, b: number): number {
        return b < a ? b : a;
}

export function quat_rot(r: number[], a: number[], b: number[]): void {
        b[0] = (r[0] * a[0] - r[1] * a[1] - r[2] * a[2] - r[3] * a[3]);
        b[1] = (r[0] * a[1] + r[1] * a[0] + r[2] * a[3] - r[3] * a[2]);
        b[2] = (r[0] * a[2] - r[1] * a[3] + r[2] * a[0] + r[3] * a[1]);
        b[3] = (r[0] * a[3] + r[1] * a[2] - r[2] * a[1] + r[3] * a[0]);
}

export function rotate_quaternion_into_fundamental_zone(num_generators: number, generator: number[][], q: number[]): number {
        let max = 0.0;
        let i = 0;
        let bi = -1;
        for (i = 0; i < num_generators; i++) {
                const g = generator[i];
                const t = Math.abs(q[0] * g[0] - q[1] * g[1] - q[2] * g[2] - q[3] * g[3]);
                if (t > max) {
                        max = t;
                        bi = i;
                }
        }

        const f = [0, 0, 0, 0];
        quat_rot(q, generator[bi], f);
        for (let k = 0; k < 4; k++) {
                q[k] = f[k];
        }
        if (q[0] < 0) {
                q[0] = -q[0];
                q[1] = -q[1];
                q[2] = -q[2];
                q[3] = -q[3];
        }

        return bi;
}

export function rotate_quaternion_into_cubic_fundamental_zone(q: number[]): number {
        return rotate_quaternion_into_fundamental_zone(24, generator_cubic, q);
}

export function rotate_quaternion_into_diamond_cubic_fundamental_zone(q: number[]): number {
        return rotate_quaternion_into_fundamental_zone(12, generator_diamond_cubic, q);
}

export function rotate_quaternion_into_icosahedral_fundamental_zone(q: number[]): number {
        return rotate_quaternion_into_fundamental_zone(60, generator_icosahedral, q);
}

export function rotate_quaternion_into_hcp_fundamental_zone(q: number[]): number {
        return rotate_quaternion_into_fundamental_zone(6, generator_hcp, q);
}

export function rotate_quaternion_into_hcp_conventional_fundamental_zone(q: number[]): number {
        return rotate_quaternion_into_fundamental_zone(12, generator_hcp_conventional, q);
}

export function rotate_quaternion_into_diamond_hexagonal_fundamental_zone(q: number[]): number {
        return rotate_quaternion_into_fundamental_zone(3, generator_diamond_hexagonal, q);
}

export function quat_dot(a: number[], b: number[]): number {
        return a[0] * b[0]
                + a[1] * b[1]
                + a[2] * b[2]
                + a[3] * b[3];
}

export function quat_size(q: number[]): number {
        return Math.sqrt(quat_dot(q, q));
}

export function normalize_quaternion(q: number[]): void {
        const size = quat_size(q);

        q[0] /= size;
        q[1] /= size;
        q[2] /= size;
        q[3] /= size;
}

export function quaternion_to_rotation_matrix(q: number[], U: number[]): void {
        const a = q[0];
        const b = q[1];
        const c = q[2];
        const d = q[3];

        U[0] = a * a + b * b - c * c - d * d;
        U[1] = 2 * b * c - 2 * a * d;
        U[2] = 2 * b * d + 2 * a * c;

        U[3] = 2 * b * c + 2 * a * d;
        U[4] = a * a - b * b + c * c - d * d;
        U[5] = 2 * c * d - 2 * a * b;

        U[6] = 2 * b * d - 2 * a * c;
        U[7] = 2 * c * d + 2 * a * b;
        U[8] = a * a - b * b - c * c + d * d;
}

export function rotation_matrix_to_quaternion(u: number[], q: number[]): void {
        const r11 = u[0];
        const r12 = u[1];
        const r13 = u[2];
        const r21 = u[3];
        const r22 = u[4];
        const r23 = u[5];
        const r31 = u[6];
        const r32 = u[7];
        const r33 = u[8];

        q[0] = (1.0 + r11 + r22 + r33) / 4.0;
        q[1] = (1.0 + r11 - r22 - r33) / 4.0;
        q[2] = (1.0 - r11 + r22 - r33) / 4.0;
        q[3] = (1.0 - r11 - r22 + r33) / 4.0;

        q[0] = Math.sqrt(std_max(0., q[0]));
        q[1] = Math.sqrt(std_max(0., q[1]));
        q[2] = Math.sqrt(std_max(0., q[2]));
        q[3] = Math.sqrt(std_max(0., q[3]));

        const m0 = std_max(q[0], q[1]);
        const m1 = std_max(q[2], q[3]);
        const max = std_max(m0, m1);

        let i = 0;
        for (i = 0; i < 4; i++) {
                if (q[i] === max) {
                        break;
                }
        }

        if (i === 0) {
                q[1] *= SIGN(r32 - r23);
                q[2] *= SIGN(r13 - r31);
                q[3] *= SIGN(r21 - r12);
        } else if (i === 1) {
                q[0] *= SIGN(r32 - r23);
                q[2] *= SIGN(r21 + r12);
                q[3] *= SIGN(r13 + r31);
        } else if (i === 2) {
                q[0] *= SIGN(r13 - r31);
                q[1] *= SIGN(r21 + r12);
                q[3] *= SIGN(r32 + r23);
        } else if (i === 3) {
                q[0] *= SIGN(r21 - r12);
                q[1] *= SIGN(r31 + r13);
                q[2] *= SIGN(r32 + r23);
        }

        normalize_quaternion(q);
}

export function quat_quick_misorientation(q1: number[], q2: number[]): number {
        let t = quat_dot(q1, q2);
        t = std_min(1., std_max(-1., t));
        return 2 * t * t - 1;
}

export function quat_misorientation(q1: number[], q2: number[]): number {
        return libmAcos(quat_quick_misorientation(q1, q2));
}
