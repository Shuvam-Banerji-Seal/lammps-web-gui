/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_born_matrix.cpp at commit b332f28 (see extra_compute/born_matrix.ts). Copyright
 * Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'born/matrix' (lazyStyles.ts group gpl_born_matrix): the core compute born/matrix calls it for
 * the numdiff keyword, through the slot in src/engine/compute/bornNumdiffHook.ts.
 */
import type { BornNumdiffHook } from '../../engine/compute/born_matrix';
import { bornMatrixNumdiff } from '../extra_compute/born_matrix';

export const HOOKS: Record<string, BornNumdiffHook> = { 'born/matrix': bornMatrixNumdiff };
