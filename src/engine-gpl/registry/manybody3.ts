/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the MANYBODY pair styles eim
 * (src/MANYBODY/pair_eim.cpp, pair_eim.h), edip (src/MANYBODY/pair_edip.cpp, pair_edip.h) and edip/multi
 * (src/MANYBODY/pair_edip_multi.cpp, pair_edip_multi.h). Copyright (2003) Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_manybody3; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import { PairEIM } from '../manybody3/pair_eim';
import { PairEDIP, PairEDIPMulti } from '../manybody3/pair_edip';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  eim: () => new PairEIM(),
  edip: () => new PairEDIP(),
  'edip/multi': () => new PairEDIPMulti(),
};
