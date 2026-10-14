/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/group.cpp (Group::xcm, Group::angmom, Group::inertia and Group::omega, with EPSILON = 1.0e-6), the
 * callers that use the same sequence: src/velocity.cpp (Velocity::zero_rotation), src/fix_momentum.cpp
 * (FixMomentum::end_of_step, angular) and src/variable.cpp (the omega() group function), and src/math_extra.cpp
 * (MathExtra::cross3, dot3, angmom_to_omega), at commit b332f28. Copyright Sandia Corporation (LAMMPS development team,
 * GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entries 'velocity', 'momentum' and 'omega' (lazyStyles.ts group gpl_group_omega, registry/group_omega.ts)
 * carry this routine into the core: the core reads it through gplHook() (src/engine/groupMotion.ts). The point-mass
 * sums, the inverse of a non-singular inertia tensor and the Jacobi path of a singular one are kept in the order of the
 * source so the results agree to rounding.
 */
import { angmomToOmega, jacobi3, type V3 } from '../rigid/math_extra';

/** EPSILON of src/group.cpp: an absolute determinant bound and a bound scaled by the largest principal moment. */
const EPSILON = 1.0e-6;

export interface GroupMotion {
  /** Group::mass: the sum of the masses, in atom order. */
  masstotal: number;
  /** Group::xcm: the centre of mass of the unwrapped positions (divided by masstotal when it is positive). */
  xcm: V3;
  /** Group::omega of Group::angmom and Group::inertia about xcm. */
  omega: V3;
}

/**
 * Group::omega(angmom, inertia, w) (src/group.cpp at b332f28): a non-singular inertia tensor (determinant > EPSILON,
 * absolute) is inverted with the cofactors and w = I^-1 L, which zeroes L exactly; otherwise I is diagonalised with
 * jacobi3 (decreasing eigenvalues), the eigenvectors are made right-handed, principal moments below EPSILON times the
 * largest are set to zero, and angmom_to_omega gives w (the pseudo-inverse along the non-degenerate axes).
 */
export const groupOmega = (angmom: V3, I: number[][]): V3 => {
  const determinant = I[0][0] * I[1][1] * I[2][2] +
    I[0][1] * I[1][2] * I[2][0] +
    I[0][2] * I[1][0] * I[2][1] -
    I[0][0] * I[1][2] * I[2][1] -
    I[0][1] * I[1][0] * I[2][2] -
    I[2][0] * I[1][1] * I[0][2];
  if (determinant > EPSILON) {
    const inv = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    inv[0][0] = I[1][1] * I[2][2] - I[1][2] * I[2][1];
    inv[0][1] = -(I[0][1] * I[2][2] - I[0][2] * I[2][1]);
    inv[0][2] = I[0][1] * I[1][2] - I[0][2] * I[1][1];
    inv[1][0] = -(I[1][0] * I[2][2] - I[1][2] * I[2][0]);
    inv[1][1] = I[0][0] * I[2][2] - I[0][2] * I[2][0];
    inv[1][2] = -(I[0][0] * I[1][2] - I[0][2] * I[1][0]);
    inv[2][0] = I[1][0] * I[2][1] - I[1][1] * I[2][0];
    inv[2][1] = -(I[0][0] * I[2][1] - I[0][1] * I[2][0]);
    inv[2][2] = I[0][0] * I[1][1] - I[0][1] * I[1][0];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) inv[i][j] /= determinant;
    return [0, 1, 2].map((r) => inv[r][0] * angmom[0] + inv[r][1] * angmom[1] + inv[r][2] * angmom[2]);
  }
  // (nearly) singular: a 2-atom group or a linear molecule
  const { eval: idiag, evec } = jacobi3(I, true);
  const ex: V3 = [evec[0][0], evec[1][0], evec[2][0]];
  const ey: V3 = [evec[0][1], evec[1][1], evec[2][1]];
  const ez: V3 = [evec[0][2], evec[1][2], evec[2][2]];
  // right-handed: flip the third vector if ex x ey points against it
  const cross: V3 = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
  if (cross[0] * ez[0] + cross[1] * ez[1] + cross[2] * ez[2] < 0.0) for (let k = 0; k < 3; k++) ez[k] = -ez[k];
  // principal moments below EPSILON times the largest are zeroed
  const max = Math.max(idiag[0], idiag[1], idiag[2]);
  for (let k = 0; k < 3; k++) if (idiag[k] < EPSILON * max) idiag[k] = 0.0;
  const w: V3 = [0, 0, 0];
  angmomToOmega(angmom, ex, ey, ez, idiag, w);
  return w;
};

/**
 * Group::xcm, Group::angmom, Group::inertia and Group::omega for one group (point masses), from the unwrapped positions
 * pos[k], the velocities vel[k] and the masses mass[k] of its members in atom order. The sums run in the source's order
 * (mass times unwrapped position; mass times the cross product of the offset from xcm and the velocity; the inertia
 * sums of src/group.cpp Group::inertia).
 */
export const groupMotion = (pos: V3[], vel: V3[], mass: number[]): GroupMotion => {
  let masstotal = 0;
  for (let k = 0; k < mass.length; k++) masstotal += mass[k];
  const xcm: V3 = [0, 0, 0];
  for (let k = 0; k < mass.length; k++) {
    xcm[0] += pos[k][0] * mass[k];
    xcm[1] += pos[k][1] * mass[k];
    xcm[2] += pos[k][2] * mass[k];
  }
  if (masstotal > 0.0) {
    xcm[0] /= masstotal;
    xcm[1] /= masstotal;
    xcm[2] /= masstotal;
  }
  const L: V3 = [0, 0, 0];
  const I = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let k = 0; k < mass.length; k++) {
    const dx = pos[k][0] - xcm[0], dy = pos[k][1] - xcm[1], dz = pos[k][2] - xcm[2];
    const vx = vel[k][0], vy = vel[k][1], vz = vel[k][2];
    const m = mass[k];
    L[0] += m * (dy * vz - dz * vy);
    L[1] += m * (dz * vx - dx * vz);
    L[2] += m * (dx * vy - dy * vx);
    I[0][0] += m * (dy * dy + dz * dz);
    I[1][1] += m * (dx * dx + dz * dz);
    I[2][2] += m * (dx * dx + dy * dy);
    I[0][1] -= m * dx * dy;
    I[1][2] -= m * dy * dz;
    I[0][2] -= m * dx * dz;
  }
  I[1][0] = I[0][1];
  I[2][1] = I[1][2];
  I[2][0] = I[0][2];
  return { masstotal, xcm, omega: groupOmega(L, I) };
};
