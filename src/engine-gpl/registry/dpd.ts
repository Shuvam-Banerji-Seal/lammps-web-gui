/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the DPD-BASIC package's pair styles
 * (src/DPD-BASIC/pair_dpd.cpp, pair_dpd_tstat.cpp, pair_dpd_ext.cpp, pair_dpd_ext_tstat.cpp). Copyright Sandia
 * Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (the names of this module are listed there).
 */
import type { Pair } from '../../engine/force/types';
import { PairDPD } from '../dpd/pair_dpd';
import { PairDPDExt } from '../dpd/pair_dpd_ext';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  dpd: () => new PairDPD(false),
  'dpd/tstat': () => new PairDPD(true),
  'dpd/ext': () => new PairDPDExt(false),
  'dpd/ext/tstat': () => new PairDPDExt(true),
};
