/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SHOCK/fix_msst.cpp, fix_msst.h, fix_nphug.cpp, fix_nphug.h (fix msst, fix nphug) at commit b332f28
 * (Laurence Fried and Evan Reed for fix msst; the LAMMPS developers for fix nphug). Copyright Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * SHOCK package fixes; registered for the core through src/engine/lazyStyles.ts (group gpl_shock). The two styles
 * that are not ported yet throw a StyleError naming them (listed in src/engine/refusedStyles.ts).
 */
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import { FixMSST } from '../shock/fix_msst';
import { FixNPHug } from '../shock/fix_nphug';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  msst: (sys, id, group, args) => new FixMSST(sys, id, group, args),
  nphug: (sys, id, group, args) => new FixNPHug(sys, id, group, args),
  // fix_wall_piston.cpp (SHOCK) is not ported: the piston is a moving wall with a stress-driven velocity
  'wall/piston': () => { throw new StyleError('fix wall/piston is not supported by the browser engine yet (src/SHOCK/fix_wall_piston.cpp is not ported)'); },
  // fix_append_atoms.cpp (SHOCK) is not ported: it adds atoms to the box at each step
  'append/atoms': () => { throw new StyleError('fix append/atoms is not supported by the browser engine yet (src/SHOCK/fix_append_atoms.cpp is not ported)'); },
};
