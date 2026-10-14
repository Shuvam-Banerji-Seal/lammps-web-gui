/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EXTRA-FIX/fix_pafi.cpp, GPL-2.0.
 * Registered through src/engine/lazyStyles.ts.
 */
import type { FixFactory } from '../../engine/styles';
import { FixPAFI } from '../pafi/fix_pafi';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  pafi: (sys, id, group, args) => new FixPAFI(sys, id, group, args),
};
