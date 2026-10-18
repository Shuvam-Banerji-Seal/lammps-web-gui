/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_lcbop.cpp and pair_lcbop.h at commit b332f28. Authors credited in the C++ file
 * header: Dominik Wojt (Wroclaw University of Technology), based on pair_airebo by Ase Henry (MIT).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairLCBOP } from '../lcbop/pair_lcbop';

/** The long-range carbon bond-order style (pair_lcbop), merged into styles.ts through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  lcbop: () => new PairLCBOP(),
};
