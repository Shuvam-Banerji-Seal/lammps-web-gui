/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DPD-REACT/pair_multi_lucy.cpp and pair_multi_lucy.h at commit b332f28 (see src/engine-gpl/multi_lucy/).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairMultiLucy } from '../multi_lucy/pair_multi_lucy';

/** pair_style multi/lucy (non-rx; merged into styles.ts through lazyStyles.ts). */
export const PAIRS: Record<string, () => Pair> = {
  'multi/lucy': () => new PairMultiLucy(),
};
