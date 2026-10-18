/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/ORIENT/fix_orient_fcc.cpp and
 * src/ORIENT/fix_orient_bcc.cpp at commit b332f28 (see src/engine-gpl/orient/). Copyright Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * ORIENT fix styles orient/fcc and orient/bcc, registered through src/engine/lazyStyles.ts (group gpl_orient).
 */
import type { FixFactory } from '../../engine/styles';
import { FixOrientFCC } from '../orient/fix_orient_fcc';
import { FixOrientBCC } from '../orient/fix_orient_bcc';

export const FIXES: Record<string, FixFactory> = {
  'orient/fcc': (sys, id, group, args) => new FixOrientFCC(sys, id, group, args),
  'orient/bcc': (sys, id, group, args) => new FixOrientBCC(sys, id, group, args),
};
