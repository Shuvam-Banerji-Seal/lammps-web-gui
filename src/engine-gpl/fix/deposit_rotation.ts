/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/fix_deposit.cpp (the molecule orientation: norm3, axisangle_to_quat) and src/math_extra.h (quat_to_mat),
 * at commit b332f28. Copyright Sandia Corporation (LAMMPS development team, GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The rotation matrix of a deposited molecule. The core FixDeposit (src/engine/fix/deposit.ts) reaches it through the
 * lazy HOOKS entry 'deposit' (lazyStyles.ts group gpl_deposit, registry/deposit.ts).
 */
import { cos as libmCos, sin as libmSin } from '../libm/sincos';

/** Rotation matrix of fix_deposit.cpp: norm3(r), axisangle_to_quat(r, theta), quat_to_mat (math_extra.h, b332f28),
 * which gives the rounding of native LAMMPS for the molecule orientation (a molecule's atoms at exactly the cutoff
 * distance are counted the same way only with the same rounding). */
export const nativeRotation = (r: number[], theta: number): number[][] => {
  const v = [r[0], r[1], r[2]];
  const val = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (val > 0.0) {
    const scale = 1.0 / Math.sqrt(val);
    v[0] *= scale; v[1] *= scale; v[2] *= scale;
  }
  const halfa = 0.5 * theta;
  // the C library's sin and cos (bit-exact glibc ports, ../libm/sincos.ts) rather than Math.sin / Math.cos
  const sina = libmSin(halfa);
  const q = [libmCos(halfa), v[0] * sina, v[1] * sina, v[2] * sina];
  const w2 = q[0] * q[0], i2 = q[1] * q[1], j2 = q[2] * q[2], k2 = q[3] * q[3];
  const twoij = 2.0 * q[1] * q[2], twoik = 2.0 * q[1] * q[3], twojk = 2.0 * q[2] * q[3];
  const twoiw = 2.0 * q[1] * q[0], twojw = 2.0 * q[2] * q[0], twokw = 2.0 * q[3] * q[0];
  return [
    [w2 + i2 - j2 - k2, twoij - twokw, twojw + twoik],
    [twoij + twokw, w2 - i2 + j2 - k2, twojk - twoiw],
    [twoik - twojw, twojk + twoiw, w2 - i2 - j2 + k2],
  ];
};
