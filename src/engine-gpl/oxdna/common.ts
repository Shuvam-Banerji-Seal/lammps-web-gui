/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the CG-DNA package's shared pieces,
 * src/CG-DNA/constants_oxdna.cpp and constants_oxdna.h (oxDNA1 site offsets in lj and real units),
 * src/CG-DNA/nucleotide_oxdna.h (NucleotideOxdna1::backbone_site, base_site<0>), src/CG-DNA/mf_oxdna.h (the
 * modulation functions F1, DF1, F2, DF2, F3, F4, DF4, F5, DF5, F6, DF6 and is_3pto5p) and the local reference
 * frame of src/CG-DNA/fix_oxdna_lrf.cpp (compute_lrf) with MathExtra::q_to_exyz from src/math_extra.cpp.
 * Original authors (from the headers): Oliver Henrich (University of Strathclyde, Glasgow) and Kierran Falloon
 * (University of Strathclyde, Glasgow); the package is described by Ouldridge, Louis and Doye (J Chem Phys 2011).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

/** The oxDNA1 site offsets (constants_oxdna.cpp: "oxDNA1 parameters"), in the units of the simulation. */
export interface OxdnaConstants {
  readonly dx_cbk_oxdna1: number;
  readonly dx_cstk_oxdna1: number;
  readonly dx_cbs_oxdna1: number;
  /** electrostatic prefactors (constants_oxdna.cpp, "electrostatic parameters") */
  readonly lambda_dh_one_prefactor: number;
  readonly qeff_dh_pf_one_prefactor: number;
}

/** constants_oxdna.cpp: "default to lj units" (the values without set_real_units). */
const LJ: OxdnaConstants = {
  dx_cbk_oxdna1: -0.4,
  dx_cstk_oxdna1: +0.34,
  dx_cbs_oxdna1: +0.4,
  lambda_dh_one_prefactor: +0.3616455075438555,
  qeff_dh_pf_one_prefactor: +0.08173808693529228,
};

/** constants_oxdna.cpp, set_real_units(): "oxDNA1 parameters in real units" and the electrostatic prefactors. */
const REAL: OxdnaConstants = {
  dx_cbk_oxdna1: -3.4072,
  dx_cstk_oxdna1: +2.89612,
  dx_cbs_oxdna1: +3.4072,
  lambda_dh_one_prefactor: +0.05624154892,
  qeff_dh_pf_one_prefactor: +4.15079634587587,
};

/**
 * ConstantsOxdna: "real_flag = utils::strmatch(units, "^real")" selects the real-unit values, anything else
 * (lj) the default ones.
 */
export const oxdnaConstants = (unitStyle: string): OxdnaConstants => (/^real/.test(unitStyle) ? REAL : LJ);

/** NucleotideOxdna1::backbone_site: rbk = dx_cbk * e1 (e1 is the local x axis). */
export const backboneSite = (dxCbk: number, e1: ArrayLike<number>, out: number[]): void => {
  out[0] = dxCbk * e1[0]; out[1] = dxCbk * e1[1]; out[2] = dxCbk * e1[2];
};

/** NucleotideOxdna1::base_site<0>: rbs = dx_cbs * e1. */
export const baseSite = (dxCbs: number, e1: ArrayLike<number>, out: number[]): void => {
  out[0] = dxCbs * e1[0]; out[1] = dxCbs * e1[1]; out[2] = dxCbs * e1[2];
};

/** NucleotideOxdna1::stacking_site: rstk = dx_cstk * e1. */
export const stackingSite = (dxCstk: number, e1: ArrayLike<number>, out: number[]): void => {
  out[0] = dxCstk * e1[0]; out[1] = dxCstk * e1[1]; out[2] = dxCstk * e1[2];
};

/**
 * fix_oxdna_lrf.cpp compute_lrf: the three Cartesian unit vectors of an ellipsoid in the lab frame, stored as
 * nxyz = ex, ey, ez (9 values) from its quaternion q = w i j k (MathExtra::q_to_exyz). Atoms that are not
 * ellipsoids keep zeros ("skip non-ellipsoid atoms").
 */
export const lrfOf = (q: ArrayLike<number>, qo: number, out: Float64Array, o: number): void => {
  const q0 = q[qo], q1 = q[qo + 1], q2 = q[qo + 2], q3 = q[qo + 3];
  out[o] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
  out[o + 1] = 2.0 * (q1 * q2 + q0 * q3);
  out[o + 2] = 2.0 * (q1 * q3 - q0 * q2);
  out[o + 3] = 2.0 * (q1 * q2 - q0 * q3);
  out[o + 4] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3;
  out[o + 5] = 2.0 * (q2 * q3 + q0 * q1);
  out[o + 6] = 2.0 * (q1 * q3 + q0 * q2);
  out[o + 7] = 2.0 * (q2 * q3 - q0 * q1);
  out[o + 8] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
};

/** MFOxdna::F1: modulation factor of the stacking term (r, eps, a, cut_0, cut_lc, cut_hc, cut_lo, cut_hi, b_lo, b_hi, shift). */
export const F1 = (r: number, eps: number, a: number, cut0: number, cutLc: number, cutHc: number, cutLo: number, cutHi: number, bLo: number, bHi: number, shift: number): number => {
  if (r > cutHc) return 0.0;
  if (r > cutHi) return eps * bHi * (r - cutHc) * (r - cutHc);
  if (r > cutLo) {
    const tmp = 1 - Math.exp(-(r - cut0) * a);
    return eps * tmp * tmp - shift;
  }
  if (r > cutLc) return eps * bLo * (r - cutLc) * (r - cutLc);
  return 0.0;
};

/** MFOxdna::DF1: derivative of F1 (no shift argument, as in the C++). */
export const DF1 = (r: number, eps: number, a: number, cut0: number, cutLc: number, cutHc: number, cutLo: number, cutHi: number, bLo: number, bHi: number): number => {
  if (r > cutHc) return 0.0;
  if (r > cutHi) return 2 * eps * bHi * (1 - cutHc / r);
  if (r > cutLo) {
    const tmp = Math.exp(-(r - cut0) * a);
    return 2 * eps * (1 - tmp) * tmp * a / r;
  }
  if (r > cutLc) return 2 * eps * bLo * (1 - cutLc / r);
  return 0.0;
};

/** MFOxdna::F2: (r, k, cut_0, cut_lc, cut_hc, cut_lo, cut_hi, b_lo, b_hi, cut_c). */
export const F2 = (r: number, k: number, cut0: number, cutLc: number, cutHc: number, cutLo: number, cutHi: number, bLo: number, bHi: number, cutC: number): number => {
  if (r < cutLc || r > cutHc) return 0;
  if (r < cutLo) return k * bLo * (cutLc - r) * (cutLc - r);
  if (r < cutHi) return k * 0.5 * ((r - cut0) * (r - cut0) - (cut0 - cutC) * (cut0 - cutC));
  return k * bHi * (cutHc - r) * (cutHc - r);
};

/** MFOxdna::DF2. */
export const DF2 = (r: number, k: number, cut0: number, cutLc: number, cutHc: number, cutLo: number, cutHi: number, bLo: number, bHi: number): number => {
  if (r < cutLc || r > cutHc) return 0;
  if (r < cutLo) return 2 * k * bLo * (r - cutLc);
  if (r < cutHi) return k * (r - cut0);
  return 2 * k * bHi * (r - cutHc);
};

/**
 * MFOxdna::F3: the excluded-volume-like term. Returns the energy and sets fpair (force over r) through out[0].
 * Inside cut_ast it is the LJ form with lj1 = 4 eps sigma^12 and lj2 = 4 eps sigma^6, outside it is the
 * quadratic tail towards cut_c.
 */
export const F3 = (rsq: number, cutsqAst: number, cutC: number, lj1: number, lj2: number, eps: number, b: number, fpair: number[]): number => {
  if (rsq < cutsqAst) {
    const r2inv = 1.0 / rsq;
    const r6inv = r2inv * r2inv * r2inv;
    fpair[0] = r2inv * r6inv * (12 * lj1 * r6inv - 6 * lj2);
    return r6inv * (lj1 * r6inv - lj2);
  }
  const r = Math.sqrt(rsq);
  const rinv = 1.0 / r;
  fpair[0] = 2 * eps * b * (cutC * rinv - 1);
  return eps * b * (cutC - r) * (cutC - r);
};

/** MFOxdna::F4 (theta, a, theta_0, dtheta_ast, b, dtheta_c). */
export const F4 = (theta: number, a: number, theta0: number, dthetaAst: number, b: number, dthetaC: number): number => {
  const dtheta = theta - theta0;
  if (Math.abs(dtheta) > dthetaC) return 0.0;
  if (dtheta > dthetaAst) return b * (dtheta - dthetaC) * (dtheta - dthetaC);
  if (dtheta > -dthetaAst) return 1 - a * dtheta * dtheta;
  return b * (dtheta + dthetaC) * (dtheta + dthetaC);
};

/** MFOxdna::DF4 (the sin(theta) factor is applied by the caller, as in the C++). */
export const DF4 = (theta: number, a: number, theta0: number, dthetaAst: number, b: number, dthetaC: number): number => {
  const dtheta = theta - theta0;
  if (Math.abs(dtheta) > dthetaC) return 0.0;
  if (dtheta > dthetaAst) return 2 * b * (dtheta - dthetaC);
  if (dtheta > -dthetaAst) return -2 * a * dtheta;
  return 2 * b * (dtheta + dthetaC);
};

/** MFOxdna::F5 (x, a, x_ast, b, x_c). */
export const F5 = (x: number, a: number, xAst: number, b: number, xC: number): number => {
  if (x >= 0) return 1.0;
  if (x > xAst) return 1 - a * x * x;
  if (x > xC) return b * (x - xC) * (x - xC);
  return 0.0;
};

/** MFOxdna::DF5. */
export const DF5 = (x: number, a: number, xAst: number, b: number, xC: number): number => {
  if (x >= 0) return 0.0;
  if (x > xAst) return -2 * a * x;
  if (x > xC) return 2 * b * (x - xC);
  return 0.0;
};

/** MFOxdna::F6 (theta, a, b). */
export const F6 = (theta: number, a: number, b: number): number => (theta < b ? 0.0 : 0.5 * a * (theta - b) * (theta - b));

/** MFOxdna::DF6. */
export const DF6 = (theta: number, a: number, b: number): number => (theta < b ? 0.0 : a * (theta - b));

/** MFOxdna::is_3pto5p: copysign(1, delr . n), i.e. +1 or -1 (the sign bit of a zero dot product is kept). */
export const is3pto5p = (delr: readonly number[], n: readonly number[]): number => {
  const d = delr[0] * n[0] + delr[1] * n[1] + delr[2] * n[2];
  return Object.is(d, -0) || d < 0 ? -1 : 1;
};
