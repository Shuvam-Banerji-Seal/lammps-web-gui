/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the style registration of
 * src/DPD-SMOOTH/fix_rigid_meso.cpp (fix rigid/meso). Its class is src/engine-gpl/rigid/fix_rigid_meso.ts.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_rigid_meso).
 */
import type { FixFactory } from '../../engine/styles';
import { FixRigidMeso } from '../rigid/fix_rigid_meso';

export const FIXES: Record<string, FixFactory> = {
  'rigid/meso': (sys, id, group, args) => new FixRigidMeso(sys, id, group, args, 'rigid/meso'),
};
