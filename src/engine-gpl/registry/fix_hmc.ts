/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: fix hmc (src/MC/fix_hmc.cpp, fix_hmc.h).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE). Registered for the core through src/engine/lazyStyles.ts (group
 * gpl_fix_hmc; the names listed there must match these keys).
 */
import type { FixFactory } from '../../engine/styles';
import { FixHmc } from '../mc/fix_hmc';

/** fix style names. */
export const FIXES: Record<string, FixFactory> = {
  hmc: (sys, id, group, args) => new FixHmc(sys, id, group, args),
};
