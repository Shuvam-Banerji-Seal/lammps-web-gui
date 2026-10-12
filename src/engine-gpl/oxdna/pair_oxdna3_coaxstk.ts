/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna3_coaxstk.cpp and
 * pair_oxdna3_coaxstk.h (class PairOxdna3Coaxstk, pair_style oxdna3/coaxstk). It is the oxDNA2 coaxial stacking
 * (PairOxdna2Coaxstk) with the oxDNA3 stacking site (NucleotideOxdna3::stacking_site) and the sequence-specific
 * coaxial strength eta_cxst (the constructor's table, A:0 C:1 G:2 T:3, applied to k_cxst in coeff()). Original
 * author (from the headers): Oliver Henrich (University of Strathclyde, Glasgow). LAMMPS is Copyright (2003)
 * Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 */
import { PairOxdna2Coaxstk } from './pair_oxdna2_coaxstk';
import { type StackRule, stackingSiteOxdna3 } from './common';

/**
 * eta_cxst[r][c] of pair_oxdna3_coaxstk.cpp (the constructor): the first index is the 3' base type of the pair,
 * the second the 5' one (row = imod4 - 1, column = jmod4 - 1).
 */
const ETA_CXST_OXDNA3: number[][] = [
  [1.1217958408368172, 1.1217958408368172, 1.1217958408368172, 0.9300223683636719],
  [1.0712851690057155, 0.7892685731520542, 0.9896542231533637, 0.7694592613578328],
  [1.1161603311902566, 1.1022201982984874, 1.1088392608169480, 1.0007533199170144],
  [1.0052361315065244, 0.8658975520778347, 1.1217958408368172, 0.8593983791552220],
];

/** pair_style oxdna3/coaxstk (PairOxdna3Coaxstk). */
export class PairOxdna3Coaxstk extends PairOxdna2Coaxstk {
  override readonly name: string = 'oxdna3/coaxstk';
  protected override stackSite: StackRule = stackingSiteOxdna3;
  protected override etaCxst: number[][] = ETA_CXST_OXDNA3;
}
