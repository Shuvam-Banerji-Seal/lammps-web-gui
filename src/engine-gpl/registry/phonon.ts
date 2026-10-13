/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/PHONON/fix_phonon.cpp and fix_phonon.h, dynamical_matrix.cpp and dynamical_matrix.h, third_order.cpp
 * and third_order.h at commit b332f28 (Ling-Ti Kong, and the LAMMPS developers). Copyright Sandia Corporation (GNU
 * General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * PHONON package fixes ported from LAMMPS; registered for the core through src/engine/lazyStyles.ts (group gpl_phonon).
 */
import type { FixFactory } from '../../engine/styles';
import type { Handler } from '../../engine/commands/args';
import { FixPhonon } from '../phonon/fix_phonon';
import { dynamicalMatrixCmd } from '../phonon/dynamical_matrix';

/** Command word -> handler (listed as COMMANDS in lazyStyles.ts). */
export const COMMANDS: Record<string, Handler> = {
  dynamical_matrix: dynamicalMatrixCmd,
};

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  phonon: (sys, id, group, args) => new FixPhonon(sys, id, group, args),
};
