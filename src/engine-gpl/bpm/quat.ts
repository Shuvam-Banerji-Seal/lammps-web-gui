/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/math_extra.h (MathExtra::quatquat, qconjugate, qnormalize, quatrotvec, and the 3-vector helpers),
 * used by src/BPM/bond_bpm_rotational.cpp, bond_bpm_peri.cpp and bond_bpm_spring_plastic.cpp at commit b332f28.
 * Copyright (2003) Sandia Corporation (GNU General Public License). Authors of the BPM sources (Joel Clemmer, SNL).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Quaternions are [w, i, j, k] with unit norm for atom orientations. Vectors are 3-element arrays.
 */

export type Vec3 = number[];
export type Quat = number[];

export const dot3 = (a: readonly number[], b: readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const len3 = (a: readonly number[]): number => Math.sqrt(dot3(a, a));
export const cross3 = (a: readonly number[], b: readonly number[]): Vec3 => [
  a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
];
export const add3 = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub3 = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale3 = (s: number, a: readonly number[]): Vec3 => [s * a[0], s * a[1], s * a[2]];
/** Unit vector a/|a|; a zero vector stays zero (MathExtra::norm3 and normalize3 skip val = 0). */
export const normalize3 = (a: readonly number[]): Vec3 => {
  const n = len3(a);
  return n > 0 ? scale3(1 / n, a) : [0, 0, 0];
};

/** Hamilton product c = a b (MathExtra::quatquat). */
export const quatquat = (a: readonly number[], b: readonly number[]): Quat => [
  a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
  a[0] * b[1] + b[0] * a[1] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] + b[0] * a[2] + a[3] * b[1] - a[1] * b[3],
  a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1],
];

export const qconjugate = (q: readonly number[]): Quat => [q[0], -q[1], -q[2], -q[3]];

/** Unit quaternion q/|q| (MathExtra::qnormalize). */
export const qnormalize = (q: readonly number[]): Quat => {
  const n = 1 / Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return [q[0] * n, q[1] * n, q[2] * n, q[3] * n];
};

/** Rotation of v by quaternion a: c = (a (0, v)) conj(a) (MathExtra::quatrotvec; unit a rotates v). */
export const quatrotvec = (a: readonly number[], b: readonly number[]): Vec3 => {
  const t0 = -a[1] * b[0] - a[2] * b[1] - a[3] * b[2];
  const t1 = a[0] * b[0] + a[2] * b[2] - a[3] * b[1];
  const t2 = a[0] * b[1] + a[3] * b[0] - a[1] * b[2];
  const t3 = a[0] * b[2] + a[1] * b[1] - a[2] * b[0];
  return [
    -a[1] * t0 + a[0] * t1 - a[3] * t2 + a[2] * t3,
    -a[2] * t0 + a[3] * t1 + a[0] * t2 - a[1] * t3,
    -a[3] * t0 - a[2] * t1 + a[1] * t2 + a[0] * t3,
  ];
};
