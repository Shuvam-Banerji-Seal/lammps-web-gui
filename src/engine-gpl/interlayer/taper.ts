/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/interlayer_taper.h at commit b332f28 (the taper polynomial shared by the interlayer
 * styles). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 */

/** Coefficients of the seventh-order taper polynomial in r/Rcut (Tap_coeff of the C++ header). */
const TAP_COEFF = [1.0, 0.0, 0.0, 0.0, -35.0, 84.0, -70.0, 20.0];

/** The taper function: 1 at r = 0, 0 at r >= Rcut (Horner evaluation as in the C++ source). */
export const calcTap = (rij: number, rcut: number): number => {
  const r = rij / rcut;
  if (r >= 1.0) return 0.0;
  let tap = TAP_COEFF[7] * r + TAP_COEFF[6];
  tap = tap * r + TAP_COEFF[5];
  tap = tap * r + TAP_COEFF[4];
  tap = tap * r + TAP_COEFF[3];
  tap = tap * r + TAP_COEFF[2];
  tap = tap * r + TAP_COEFF[1];
  tap = tap * r + TAP_COEFF[0];
  return tap;
};

/** Derivative of the taper function with respect to r. */
export const calcDTap = (rij: number, rcut: number): number => {
  const r = rij / rcut;
  if (r >= 1.0) return 0.0;
  let dtap = 7.0 * TAP_COEFF[7] * r + 6.0 * TAP_COEFF[6];
  dtap = dtap * r + 5.0 * TAP_COEFF[5];
  dtap = dtap * r + 4.0 * TAP_COEFF[4];
  dtap = dtap * r + 3.0 * TAP_COEFF[3];
  dtap = dtap * r + 2.0 * TAP_COEFF[2];
  dtap = dtap * r + TAP_COEFF[1];
  return dtap / rcut;
};
