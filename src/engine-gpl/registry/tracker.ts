/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MISC/pair_tracker.cpp, pair_tracker.h and src/STORE/fix_store_local.cpp at commit b332f28 (see
 * src/engine-gpl/tracker/). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairTracker } from '../tracker/pair_tracker';

/** pair_style tracker (merged into styles.ts through lazyStyles.ts; the fix it creates is internal). */
export const PAIRS: Record<string, () => Pair> = {
  tracker: () => new PairTracker(),
};
