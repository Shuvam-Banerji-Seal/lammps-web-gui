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
import { exp as libmExp } from '../libm/exp';

/** The oxDNA1 site offsets (constants_oxdna.cpp: "oxDNA1 parameters"), in the units of the simulation. */
export interface OxdnaConstants {
  readonly dx_cbk_oxdna1: number;
  readonly dx_cstk_oxdna1: number;
  readonly dx_cbs_oxdna1: number;
  /** oxDNA2 backbone site offsets along e1 and e2 */
  readonly dx_cbk_oxdna2: number;
  readonly dy_cbk_oxdna2: number;
  /** oxRNA2 backbone site offsets along e1 and e3, and the stacking offsets of the 3' and 5' nucleotides */
  readonly dx_cbk_oxrna2: number;
  readonly dz_cbk_oxrna2: number;
  readonly dx_cstk_3p_oxrna2: number;
  readonly dy_cstk_3p_oxrna2: number;
  readonly dx_cstk_5p_oxrna2: number;
  readonly dy_cstk_5p_oxrna2: number;
  /** electrostatic prefactors (constants_oxdna.cpp, "electrostatic parameters") */
  readonly lambda_dh_one_prefactor: number;
  readonly qeff_dh_pf_one_prefactor: number;
  /** oxDNA3 stacking offset and base-site offsets of the purines and pyrimidines (constants_oxdna.cpp, "oxDNA3 parameters") */
  readonly dx_cstk_oxdna3: number;
  readonly dx_cbs_pur_oxdna3: number;
  readonly dx_cbs_pyr_oxdna3: number;
}

/** constants_oxdna.cpp: "default to lj units" (the values without set_real_units). */
const LJ: OxdnaConstants = {
  dx_cbk_oxdna1: -0.4,
  dx_cstk_oxdna1: +0.34,
  dx_cbs_oxdna1: +0.4,
  dx_cbk_oxdna2: -0.34,
  dy_cbk_oxdna2: +0.3408,
  dx_cbk_oxrna2: -0.4,
  dz_cbk_oxrna2: +0.2,
  dx_cstk_3p_oxrna2: +0.4,
  dy_cstk_3p_oxrna2: +0.1,
  dx_cstk_5p_oxrna2: +0.124906078525,
  dy_cstk_5p_oxrna2: -0.00866274917473,
  lambda_dh_one_prefactor: +0.3616455075438555,
  qeff_dh_pf_one_prefactor: +0.08173808693529228,
  dx_cstk_oxdna3: +0.37,
  dx_cbs_pur_oxdna3: +0.43,
  dx_cbs_pyr_oxdna3: +0.37,
};

/** constants_oxdna.cpp, set_real_units(): "oxDNA1 parameters in real units" and the electrostatic prefactors. */
const REAL: OxdnaConstants = {
  dx_cbk_oxdna1: -3.4072,
  dx_cstk_oxdna1: +2.89612,
  dx_cbs_oxdna1: +3.4072,
  dx_cbk_oxdna2: -2.89612,
  dy_cbk_oxdna2: +2.9029344,
  dx_cbk_oxrna2: -3.4072,
  dz_cbk_oxrna2: +1.7036,
  dx_cstk_3p_oxrna2: +3.4072,
  dy_cstk_3p_oxrna2: +0.8518,
  dx_cstk_5p_oxrna2: +1.063949977,
  dy_cstk_5p_oxrna2: -0.07378929747,
  lambda_dh_one_prefactor: +0.05624154892,
  qeff_dh_pf_one_prefactor: +4.15079634587587,
  dx_cstk_oxdna3: +3.15166,
  dx_cbs_pur_oxdna3: +3.66274,
  dx_cbs_pyr_oxdna3: +3.15166,
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

/** A backbone site rule: out = site(e1, e2, e3) with the constants of the unit style (the C++ nucleotide classes). */
export type BackboneRule = (c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]) => void;

/** NucleotideOxdna1::backbone_site. */
export const backboneOxdna1: BackboneRule = (c, e1, _e2, _e3, out) => backboneSite(c.dx_cbk_oxdna1, e1, out);

/** NucleotideOxdna2::backbone_site: rbk = dx_cbk e1 + dy_cbk e2. */
export const backboneOxdna2: BackboneRule = (c, e1, e2, _e3, out) => {
  for (let k = 0; k < 3; k++) out[k] = c.dx_cbk_oxdna2 * e1[k] + c.dy_cbk_oxdna2 * e2[k];
};

/** NucleotideOxrna2::backbone_site: rbk = dx_cbk e1 + dz_cbk e3. */
export const backboneOxrna2: BackboneRule = (c, e1, _e2, e3, out) => {
  for (let k = 0; k < 3; k++) out[k] = c.dx_cbk_oxrna2 * e1[k] + c.dz_cbk_oxrna2 * e3[k];
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
    const tmp = 1 - libmExp(-(r - cut0) * a);
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
    const tmp = libmExp(-(r - cut0) * a);
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

/** NucleotideOxrna2 dh site as written in pair_oxrna2_dh.cpp: dx_cbk_oxdna1 e1 + dz_cbk_oxrna2 e3 (the oxDNA1 x offset). */
export const backboneOxrna2Dh: BackboneRule = (c, e1, _e2, e3, out) => {
  for (let k = 0; k < 3; k++) out[k] = c.dx_cbk_oxdna1 * e1[k] + c.dz_cbk_oxrna2 * e3[k];
};

/** NucleotideOxdna3::stacking_site: rstk = dx_cstk_oxdna3 * e1. */
export const stackingSiteOxdna3: StackRule = (c, e1, _e2, _e3, out) => stackingSite(c.dx_cstk_oxdna3, e1, out);

/** NucleotideOxdna1::stacking_site (the rule of the oxDNA and oxDNA2 styles): rstk = dx_cstk_oxdna1 * e1. */
export const stackingSiteOxdna1: StackRule = (c, e1, _e2, _e3, out) => stackingSite(c.dx_cstk_oxdna1, e1, out);

/**
 * NucleotideOxdna3::base_site<N> with N = type % 4 (the C++ callers pass atype % 4): N = 1 and 3 (A, G) are the
 * purines (dx_cbs_pur), N = 0 and 2 (T, C) the pyrimidines (dx_cbs_pyr).
 */
export const baseSiteOxdna3 = (c: OxdnaConstants, type: number, e1: ArrayLike<number>, out: number[]): void => {
  const dx = type % 2 === 1 ? c.dx_cbs_pur_oxdna3 : c.dx_cbs_pyr_oxdna3;
  out[0] = dx * e1[0]; out[1] = dx * e1[1]; out[2] = dx * e1[2];
};

/** NucleotideOxdna1::base_site<0> as a type-indexed rule (the oxDNA and oxDNA2 styles ignore the type). */
export const baseSiteOxdna1 = (c: OxdnaConstants, _type: number, e1: ArrayLike<number>, out: number[]): void => baseSite(c.dx_cbs_oxdna1, e1, out);

/**
 * A stacking-site rule: out = site(constants, e1). The oxDNA3 base-site rule also takes the nucleotide type.
 */
export type StackRule = (c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]) => void;

/**
 * The C++ sequence-dependent tables (Delta, r0, sigma4, cut4, cut_st_0, ... in the oxDNA3 styles) are four-index
 * arrays [i][j][k][l] with i, j, k, l = 0..n, where type 0 stands for "no neighbour" (the terminal base). The file
 * gives the n^4 values for 1 <= i, j, k, l <= n in the order i, j, k, l (l fastest). The C++ coeff() then fills
 *   T[i][j][k][0] += T[i][j][k][l]      (sum over the 5' neighbour l)
 *   T[0][j][k][l] += T[i][j][k][l]      (sum over the 3' neighbour i)
 *   T[0][j][k][0] += T[i][j][k][l]      (sum over both)
 * and then divides the sums by n, n and n^2 ("calculate sequence-averaged parameters for terminal base step j-k").
 * The result is returned flat with the index (((i * m + j) * m + k) * m + l), m = n + 1.
 */
export const tetramerTable = (vals: ArrayLike<number>, n: number): Float64Array => {
  const m = n + 1;
  const t = new Float64Array(m * m * m * m);
  const at = (i: number, j: number, k: number, l: number) => ((i * m + j) * m + k) * m + l;
  let v = 0;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= n; j++) {
      for (let k = 1; k <= n; k++) {
        for (let l = 1; l <= n; l++) {
          const x = vals[v++];
          t[at(i, j, k, l)] = x;
          t[at(i, j, k, 0)] += x;
          t[at(0, j, k, l)] += x;
          t[at(0, j, k, 0)] += x;
        }
      }
    }
  }
  for (let i = 1; i <= n; i++) for (let j = 1; j <= n; j++) for (let k = 1; k <= n; k++) t[at(i, j, k, 0)] /= n;
  for (let j = 1; j <= n; j++) for (let k = 1; k <= n; k++) for (let l = 1; l <= n; l++) t[at(0, j, k, l)] /= n;
  for (let j = 1; j <= n; j++) for (let k = 1; k <= n; k++) t[at(0, j, k, 0)] /= n * n;
  return t;
};

/** Flat index of tetramerTable's layout for m = n + 1 entries per axis. */
export const tetIndex = (m: number, i: number, j: number, k: number, l: number): number => ((i * m + j) * m + k) * m + l;

/**
 * The word lists of the lines of a potential file whose key words match the coefficient words and whose name word
 * (the word after the keys) is `name`. A key word matches when it is equal to the coefficient word or is the
 * wildcard "*": the C++ compares with the words of the command, and the engine's hybrid styles pass each type
 * number where the command had "*", so a "*" line must match every type. Lines that match exactly come first, so
 * an explicit "1 4 hbond" line is used for the pair 1 4 before a "* * hbond" line of the same file.
 */
export const potentialLines = (text: string, keys: readonly string[], name: string): string[][] => {
  const exact: string[][] = [], wild: string[][] = [];
  for (const line of text.split(/\r?\n/)) {
    const w = line.replace(/#.*$/, '').trim().split(/\s+/).filter((x) => x.length);
    if (w.length < keys.length + 1 || w[keys.length] !== name) continue;
    let isExact = true, isWild = true;
    for (let k = 0; k < keys.length; k++) {
      if (w[k] !== keys[k]) isExact = false;
      if (w[k] !== keys[k] && w[k] !== '*') isWild = false;
    }
    if (isExact) exact.push(w);
    else if (isWild) wild.push(w);
  }
  return [...exact, ...wild];
};

/**
 * Per-atom virial of the CG-DNA styles' ev_tally_xyz with newton on (pair.cpp and bond.cpp ev_tally_xyz, and the
 * copy in pair_oxdna_stk.cpp): v = d (x) f, half to each of i and j, in the order xx, yy, zz, xy, xz, yz.
 */
export const halfVirial = (vatom: Float64Array, i: number, j: number, d: ArrayLike<number>, f: ArrayLike<number>): void => {
  const v0 = 0.5 * d[0] * f[0], v1 = 0.5 * d[1] * f[1], v2 = 0.5 * d[2] * f[2];
  const v3 = 0.5 * d[0] * f[1], v4 = 0.5 * d[0] * f[2], v5 = 0.5 * d[1] * f[2];
  vatom[6 * i] += v0; vatom[6 * i + 1] += v1; vatom[6 * i + 2] += v2; vatom[6 * i + 3] += v3; vatom[6 * i + 4] += v4; vatom[6 * i + 5] += v5;
  vatom[6 * j] += v0; vatom[6 * j + 1] += v1; vatom[6 * j + 2] += v2; vatom[6 * j + 3] += v3; vatom[6 * j + 4] += v4; vatom[6 * j + 5] += v5;
};
