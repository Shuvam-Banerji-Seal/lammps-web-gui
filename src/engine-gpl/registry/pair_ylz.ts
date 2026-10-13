/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ASPHERE/pair_ylz.cpp and pair_ylz.h at commit b332f28 (see src/engine-gpl/ylz/pair_ylz.ts).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairYLZ } from '../ylz/pair_ylz';

/** Membrane ellipsoid style ported from LAMMPS (pair_ylz); merged into styles.ts through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  ylz: () => new PairYLZ(),
};
