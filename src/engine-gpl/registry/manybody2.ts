/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the MANYBODY pair styles gw, gw/zbl,
 * tersoff/table and polymorphic (src/MANYBODY/pair_gw.cpp, pair_gw_zbl.cpp, pair_tersoff_table.cpp,
 * pair_polymorphic.cpp and their headers). Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_manybody2; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import { PairGW, PairGWZBL } from '../manybody2/pair_gw';
import { PairTersoffTable } from '../manybody2/pair_tersoff_table';
import { PairPolymorphic } from '../manybody2/pair_polymorphic';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  gw: () => new PairGW(),
  'gw/zbl': () => new PairGWZBL(),
  'tersoff/table': () => new PairTersoffTable(),
  polymorphic: () => new PairPolymorphic(),
};
