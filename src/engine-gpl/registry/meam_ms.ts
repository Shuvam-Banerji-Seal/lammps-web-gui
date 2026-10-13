/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MEAM (pair_meam_ms.cpp and the msmeam
 * terms of pair_meam.cpp and meam_*.cpp). GPL-2.0 (src/engine-gpl/LICENSE). Registered through src/engine/lazyStyles.ts
 * (the name list there is 'meam/ms').
 */
import type { Pair } from '../../engine/force/types';
import { PairMeamMs } from '../meam/pairMeamMs';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'meam/ms': () => new PairMeamMs(),
};
