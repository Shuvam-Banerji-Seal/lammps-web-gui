/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MOLECULE/bond_quartic.cpp, src/MISC/bond_special.cpp, src/DIPOLE/angle_dipole.cpp, src/MISC/pair_list.cpp
 * and src/MC/pair_dsmc.cpp at commit b332f28 (the LAMMPS developers; Copyright (2003) Sandia Corporation, GNU General
 * Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Bonded and pair styles of the MISC, MOLECULE, DIPOLE and MC packages, registered for the core through
 * src/engine/lazyStyles.ts (group gpl_miscx; the names listed there must match these keys).
 */
import type { Bonded, Pair } from '../../engine/force/types';
import { BondQuartic } from '../miscx/bond_quartic';
import { BondSpecial } from '../miscx/bond_special';
import { AngleDipole } from '../miscx/angle_dipole';
import { PairList } from '../miscx/pair_list';
import { PairDSMC } from '../miscx/pair_dsmc';

export const BONDS: Record<string, () => Bonded> = {
  quartic: () => new BondQuartic(),
  special: () => new BondSpecial(),
};
export const ANGLES: Record<string, () => Bonded> = {
  dipole: () => new AngleDipole(),
};
export const PAIRS: Record<string, () => Pair> = {
  list: () => new PairList(),
  dsmc: () => new PairDSMC(),
};
