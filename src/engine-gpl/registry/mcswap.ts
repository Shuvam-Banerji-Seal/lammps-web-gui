/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/MC/fix_atom_swap.cpp and src/MC/fix_atom_swap.h (Paul Crozier (SNL), Alexander Stukowski), with the other
 * MC-package fixes of this group (see the header of each file under src/engine-gpl/mcswap/). LAMMPS is Copyright (2003)
 * Sandia Corporation, distributed under the GNU General Public License version 2; this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_mcswap).
 */
import type { FixFactory } from '../../engine/styles';
import { FixAtomSwap } from '../mcswap/fix_atom_swap';
import { FixBondCreateAngle } from '../mcswap/fix_bond_create_angle';
import { FixBondSwap } from '../mcswap/fix_bond_swap';
import { FixTFMC } from '../mcswap/fix_tfmc';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  'atom/swap': (sys, id, group, args) => new FixAtomSwap(sys, id, group, args),
  tfmc: (sys, id, group, args) => new FixTFMC(sys, id, group, args),
  'bond/create/angle': (sys, id, group, args) => new FixBondCreateAngle(sys, id, group, args),
  'bond/swap': (sys, id, group, args) => new FixBondSwap(sys, id, group, args),
};
