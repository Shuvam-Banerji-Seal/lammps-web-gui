/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANYBODY/pair_bop.cpp and
 * pair_bop.h (pair_style bop). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_bop; the name listed there must match this key).
 */
import type { Pair } from '../../engine/force/types';
import { PairBOP } from '../bop/pair_bop';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  bop: () => new PairBOP(),
};
