/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SHOCK/fix_msst.cpp, fix_msst.h, fix_nphug.cpp, fix_nphug.h (fix msst, fix nphug) at commit b332f28
 * (Laurence Fried and Evan Reed for fix msst; the LAMMPS developers for fix nphug). Copyright Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * SHOCK package fixes; registered for the core through src/engine/lazyStyles.ts (group gpl_shock). A style that is not
 * ported yet throws a StyleError naming it (listed in src/engine/refusedStyles.ts).
 */
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import { FixMSST } from '../shock/fix_msst';
import { FixNPHug } from '../shock/fix_nphug';
import { FixWallPiston } from '../shock/fix_wall_piston';
import { FixAppendAtoms } from '../shock/fix_append_atoms';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  msst: (sys, id, group, args) => new FixMSST(sys, id, group, args),
  nphug: (sys, id, group, args) => new FixNPHug(sys, id, group, args),
  'wall/piston': (sys, id, group, args) => new FixWallPiston(sys, id, group, args),
  'append/atoms': (sys, id, group, args) => new FixAppendAtoms(sys, id, group, args),
};
