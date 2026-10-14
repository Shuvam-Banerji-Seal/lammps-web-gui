/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the pair style e3b
 * (src/EXTRA-PAIR/pair_e3b.cpp, pair_e3b.h). Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_e3b; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import { PairE3B } from '../e3b/pair_e3b';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  e3b: () => new PairE3B(),
};
