/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_extep.cpp and src/MANYBODY/pair_extep.h at commit b332f28. Copyright Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * pair_style extep, registered for the core through src/engine/lazyStyles.ts (group gpl_extep).
 */
import type { Pair } from '../../engine/force/types';
import { PairExTeP } from '../extep/pair_extep';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const PAIRS: Record<string, () => Pair> = {
  extep: () => new PairExTeP(),
};
