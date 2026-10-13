/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/fix_deposit.cpp and src/math_extra.h at commit b332f28; the routine is in ../fix/deposit_rotation.ts.
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'deposit' (lazyStyles.ts group gpl_deposit): the command word of fix deposit. The core reads it
 * through gplHook('deposit') (src/engine/fix/deposit.ts).
 */
import type { DepositRotationHook } from '../../engine/fix/deposit';
import { nativeRotation } from '../fix/deposit_rotation';

export const HOOKS: Record<string, DepositRotationHook> = { deposit: { rotation: nativeRotation } };
