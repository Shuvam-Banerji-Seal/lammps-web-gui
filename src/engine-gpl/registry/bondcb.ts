/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MC/fix_bond_create.cpp and src/MC/fix_bond_break.cpp at commit b332f28 (the LAMMPS developers; Copyright
 * Sandia Corporation). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Bond topology fixes registered for the core through src/engine/lazyStyles.ts (group gpl_bondcb).
 */
import type { FixFactory } from '../../engine/styles';
import { FixBondBreak } from '../mc/fix_bond_break';
import { FixBondCreate } from '../mc/fix_bond_create';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  'bond/create': (sys, id, group, args) => new FixBondCreate(sys, id, group, args),
  'bond/break': (sys, id, group, args) => new FixBondBreak(sys, id, group, args),
};
