/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/fix_settorque_atom.cpp and fix_settorque_atom.h at commit b332f28 (see src/engine-gpl/granular/).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_settorque_atom; the names listed there must match).
 */
import type { Fix } from '../../engine/fix/fix';
import type { FixFactory } from '../../engine/styles';
import { FixSetTorqueAtom } from '../granular/fix_settorque_atom';

/** fix styles. */
export const FIXES: Record<string, FixFactory> = {
  'settorque/atom': (sys, id, group, args): Fix => new FixSetTorqueAtom(sys, id, group, args),
};
