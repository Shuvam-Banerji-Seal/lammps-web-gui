/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the style registrations of
 * src/RIGID/fix_rigid_nvt.cpp, fix_rigid_npt.cpp and fix_rigid_nph.cpp (fix rigid/nvt, rigid/npt, rigid/nph). Its Nose-Hoover base is src/engine-gpl/rigid/fix_rigid_nh.ts, the
 * rigid-body base is src/engine-gpl/rigid/fix_rigid.ts (from src/RIGID/fix_rigid.cpp).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Contributing author of the source: Tony Sheh (U Michigan), Trung Dac Nguyen (U Michigan).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_rigid_nh); the core keeps the rigid/nvt refusal out of its
 * registries, so these names are the only ones.
 */
import type { FixFactory } from '../../engine/styles';
import { FixRigidNVT, FixRigidNPT, FixRigidNPH } from '../rigid/fix_rigid_nh';

export const FIXES: Record<string, FixFactory> = {
  'rigid/nvt': (sys, id, group, args) => new FixRigidNVT(sys, id, group, args, 'rigid/nvt'),
  'rigid/npt': (sys, id, group, args) => new FixRigidNPT(sys, id, group, args, 'rigid/npt'),
  'rigid/nph': (sys, id, group, args) => new FixRigidNPH(sys, id, group, args, 'rigid/nph'),
  'rigid/nvt/small': (sys, id, group, args) => new FixRigidNVT(sys, id, group, args, 'rigid/nvt/small'),
  'rigid/npt/small': (sys, id, group, args) => new FixRigidNPT(sys, id, group, args, 'rigid/npt/small'),
  'rigid/nph/small': (sys, id, group, args) => new FixRigidNPH(sys, id, group, args, 'rigid/nph/small'),
};
