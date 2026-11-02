/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/gran_sub_mod_normal.cpp (GranSubModNormalJKR::calculate_contact_radius, touch and
 * pulloff_distance, and its constants PI27SQ, THREEROOT3, SIXROOT6, INVROOT6 and JKRPREFIX), at commit b332f28.
 * Contributing authors credited in the C++ file: Dan Bolintineanu (SNL), Joel Clemmer (SNL), Ishan Srivastava (SNL),
 * Jeremy Lechman (SNL), Leo Silbert (SNL), Gary Grest (SNL); copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The JKR normal model of pair_style granular and fix wall/gran granular: the contact radius as the closed-form root of
 * the JKR relation, and the pull-off distance beyond contact, with native's operations in native's order (the core pair
 * reaches them through the lazy hook 'jkr', lazyStyles.ts group gpl_granular_jkr).
 */
import { cbrt } from '../libm/cbrt';

/** gran_sub_mod_normal.cpp: "PI27SQ = 266.479318829412648029", "THREEROOT3 = 5.19615242270663202362", ... */
const PI27SQ = 266.479318829412648029; // 27*PI^2
const THREEROOT3 = 5.19615242270663202362; // 3*sqrt(3)
const SIXROOT6 = 14.69693845669906728801; // 6*sqrt(6)
const INVROOT6 = 0.40824829046386307274; // 1/sqrt(6)
const JKRPREFIX = 1.2277228507842888; // cbrt(3*PI**2/16)
const MY_PI = 3.14159265358979323846;
/** The MAX(0, x) of src/pointers.h ("#define MAX(A,B) ((A) > (B) ? (A) : (B))"): x itself for -0 and NaN, as native. */
const max0 = (x: number): number => (0 > x ? 0 : x);

/** GranSubModNormalJKR::calculate_contact_radius: the contact radius for dR = delta Reff. */
export const jkrContactRadius = (dR: number, Reff: number, Emix: number, cohesion: number): number => {
  const R2 = Reff * Reff;
  const dR2 = dR * dR;
  const t0 = cohesion * cohesion * R2 * R2 * Emix;
  const t1 = PI27SQ * t0;
  const t2 = 8.0 * dR * dR2 * Emix * Emix * Emix;
  const t3 = 4.0 * dR2 * Emix;
  // in case sqrt(0) < 0 due to precision issues
  const sqrt1 = max0(t0 * (t1 + 2.0 * t2));
  const t4 = cbrt(t1 + t2 + THREEROOT3 * MY_PI * Math.sqrt(sqrt1));
  const t5 = t3 / t4 + t4 / Emix;
  const sqrt2 = max0(2.0 * dR + t5);
  const t6 = Math.sqrt(sqrt2);
  const sqrt3 = max0(4.0 * dR - t5 + SIXROOT6 * cohesion * MY_PI * R2 / (Emix * t6));
  return INVROOT6 * (t6 + Math.sqrt(sqrt3));
};

/** GranSubModNormalJKR::pulloff_distance and touch: the overlap beyond contact (positive) at which a contact ends. */
export const jkrPulloff = (Reff: number, Emix: number, cohesion: number): number =>
  JKRPREFIX * cbrt(Reff * cohesion * cohesion / (Emix * Emix));
