/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the MANYBODY pair style sw
 * (src/MANYBODY/pair_sw.cpp and pair_sw.h). Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_sw; the names listed there must match these keys).
 * sw/mod stays in the core (src/engine/force/pair/sw.ts).
 */
import type { Pair } from '../../engine/force/types';
import { PairSW } from '../manybody3/pair_sw';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  sw: () => new PairSW(),
};
