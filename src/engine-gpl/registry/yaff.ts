/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/YAFF/bond_mm3.cpp, angle_cross.cpp and pair_lj_switch3_coulgauss_long.cpp (and their headers) at
 * commit b332f28 (the YAFF package). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * YAFF styles ported from LAMMPS; registered for the core through src/engine/lazyStyles.ts (group gpl_yaff).
 */
import type { Bonded, Pair } from '../../engine/force/types';
import { BondMM3 } from '../yaff/bond_mm3';
import { AngleCross } from '../yaff/angle_cross';
import { PairLJSwitch3CoulGaussLong } from '../yaff/pair_lj_switch3_coulgauss_long';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const BONDS: Record<string, () => Bonded> = {
  mm3: () => new BondMM3(),
};

export const ANGLES: Record<string, () => Bonded> = {
  cross: () => new AngleCross(),
};

export const PAIRS: Record<string, () => Pair> = {
  'lj/switch3/coulgauss/long': () => new PairLJSwitch3CoulGaussLong(),
};
