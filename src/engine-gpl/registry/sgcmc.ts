/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MC/fix_sgcmc.cpp (fix sgcmc, see src/engine-gpl/mc/fix_sgcmc.ts). LAMMPS is Copyright (2003) Sandia
 * Corporation, distributed under the GNU General Public License version 2; this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_sgcmc).
 */
import type { FixFactory } from '../../engine/styles';
import { FixSgcmc } from '../mc/fix_sgcmc';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  sgcmc: (sys, id, group, args) => new FixSgcmc(sys, id, group, args),
};
