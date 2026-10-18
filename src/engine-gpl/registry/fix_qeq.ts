/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/QEQ/fix_qeq.cpp, fix_qeq_point.cpp, fix_qeq_shielded.cpp, fix_qeq_dynamic.cpp, fix_qeq_fire.cpp and
 * fix_qeq_slater.cpp at commit b332f28. Copyright Sandia Corporation (LAMMPS development team, GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * fix qeq/point, qeq/shielded, qeq/dynamic, qeq/fire and qeq/slater (docs.lammps.org/fix_qeq.html). Merged through the
 * lazy group 'fix_qeq' of src/engine/lazyStyles.ts; the class is in ../qeq/fix_qeq.ts. qeq/ctip is the gpl_ctip group.
 */
import type { FixFactory } from '../../engine/styles';
import { FixQeq } from '../qeq/fix_qeq';

export const FIXES: Record<string, FixFactory> = {
  'qeq/point': (sys, id, group, args) => new FixQeq(sys, id, group, args, 'point'),
  'qeq/shielded': (sys, id, group, args) => new FixQeq(sys, id, group, args, 'shielded'),
  'qeq/dynamic': (sys, id, group, args) => new FixQeq(sys, id, group, args, 'dynamic'),
  'qeq/fire': (sys, id, group, args) => new FixQeq(sys, id, group, args, 'fire'),
  'qeq/slater': (sys, id, group, args) => new FixQeq(sys, id, group, args, 'slater'),
};
