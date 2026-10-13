/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/QEQ/fix_qeq.cpp, src/QEQ/fix_qeq_point.cpp, src/QEQ/fix_qeq_shielded.cpp and src/QEQ/fix_qeq_slater.cpp
 * at commit b332f28. Copyright Sandia Corporation (LAMMPS development team, GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entries 'qeq/point', 'qeq/shielded' and 'qeq/slater' (lazyStyles.ts group gpl_qeq): the command words
 * of fix qeq/point, qeq/shielded and qeq/slater. Each is a factory of the CG solver of its fix (../qeq/native_solver.ts);
 * the core FixQeq reads it through gplHook(word) (src/engine/gplHooks.ts).
 */
import type { QeqSolverHook } from '../../engine/fix/qeq';
import { NativeQeqSolver } from '../qeq/native_solver';

const solverHook: QeqSolverHook = { create: (o) => new NativeQeqSolver(o) };

export const HOOKS: Record<string, QeqSolverHook> = {
  'qeq/point': solverHook,
  'qeq/shielded': solverHook,
  'qeq/slater': solverHook,
};
