/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/QTB/fix_qtb.cpp, fix_qtb.h, fix_qbmsst.cpp, fix_qbmsst.h at commit b332f28 (Shen, Yuan, Qi, Tingting,
 * and Reed, Evan, and the LAMMPS developers). Copyright Sandia Corporation (GNU General Public License). This file
 * is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * QTB package fixes ported from LAMMPS; registered for the core through src/engine/lazyStyles.ts (group gpl_qtb).
 */
import type { FixFactory } from '../../engine/styles';
import { FixQTB } from '../qtb/fix_qtb';
import { FixQBMSST } from '../qtb/fix_qbmsst';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  qtb: (sys, id, group, args) => new FixQTB(sys, id, group, args),
  qbmsst: (sys, id, group, args) => new FixQBMSST(sys, id, group, args),
};
