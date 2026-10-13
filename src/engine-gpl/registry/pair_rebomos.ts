/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_rebomos.cpp and pair_rebomos.h at commit b332f28 (see src/engine-gpl/rebomos/rebomos.ts).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairREBOMoS } from '../rebomos/rebomos';

/** Many-body MoS2 REBO style ported from LAMMPS (pair_rebomos); merged into styles.ts through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  rebomos: () => new PairREBOMoS(),
};
