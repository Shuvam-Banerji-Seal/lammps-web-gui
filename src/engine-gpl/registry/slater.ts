/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the pair styles coul/slater/long
 * (src/EXTRA-PAIR/pair_coul_slater_long.cpp) and dpd/coul/slater/long (src/DPD-BASIC/pair_dpd_coul_slater_long.cpp).
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_slater; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import { PairCoulSlaterLong } from '../slater/pair_coul_slater_long';
import { PairDPDCoulSlaterLong } from '../slater/pair_dpd_coul_slater_long';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'coul/slater/long': () => new PairCoulSlaterLong(),
  'dpd/coul/slater/long': () => new PairDPDCoulSlaterLong(),
};
