/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/gran_sub_mod_normal.cpp (GranSubModNormalJKR), at commit b332f28; see
 * src/engine-gpl/granular/jkr.ts. Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'jkr' (lazyStyles.ts group gpl_granular_jkr): a pair_coeff or fix wall/gran with the jkr normal
 * model loads it, and the core granular contact takes the JKR contact radius and pull-off distance from it.
 */
import type { GranularJkrHook } from '../../engine/force/pair/granular';
import { jkrContactRadius, jkrPulloff } from '../granular/jkr';

export const HOOKS: Record<string, GranularJkrHook> = {
  jkr: { contactRadius: jkrContactRadius, pulloff: jkrPulloff },
};
