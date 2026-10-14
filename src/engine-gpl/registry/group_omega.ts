/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/group.cpp (Group::omega and its point-mass sums), src/velocity.cpp, src/fix_momentum.cpp, src/variable.cpp;
 * the routine is in ../group/group_omega.ts. Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entries of lazyStyles.ts group gpl_group_omega: the command words velocity (zero angular), momentum
 * (fix momentum angular) and omega (the variable group function). The core reads each through gplHook(word).
 */
import type { GroupMotionHook } from '../../engine/groupMotion';
import { groupMotion } from '../group/group_omega';

const hook: GroupMotionHook = { motion: groupMotion };

export const HOOKS: Record<string, GroupMotionHook> = { velocity: hook, momentum: hook, omega: hook };
