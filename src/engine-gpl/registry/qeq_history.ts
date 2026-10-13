/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/QEQ/fix_qeq.cpp, src/QEQ/fix_qeq_slater.cpp (see ../qeq/history.ts) at commit b332f28. Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'qeq/slater' (lazyStyles.ts group gpl_qeq_history): the command word of fix qeq/slater.
 */
import type { QeqHistoryHook } from '../../engine/fix/qeq';
import { QeqChargeHistory } from '../qeq/history';

export const HOOKS: Record<string, QeqHistoryHook> = {
  'qeq/slater': { create: (n, q) => new QeqChargeHistory(n, q) },
};
