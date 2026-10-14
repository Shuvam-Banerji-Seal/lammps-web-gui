/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-PAIR/pair_bondval.cpp and pair_bondval_vec.cpp at commit b332f28 (see src/engine-gpl/bondval/).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_bondval; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import { PairBondVal } from '../bondval/pair_bondval';
import { PairBondValVec } from '../bondval/pair_bondval_vec';

/** pair styles. */
export const PAIRS: Record<string, () => Pair> = {
  bondval: () => new PairBondVal(),
  'bondval/vec': () => new PairBondValVec(),
};
