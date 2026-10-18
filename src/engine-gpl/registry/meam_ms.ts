/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MEAM (pair_meam.cpp, pair_meam_ms.cpp and
 * meam_*.cpp). GPL-2.0 (src/engine-gpl/LICENSE). Registered through src/engine/lazyStyles.ts (the name list there is
 * 'meam/ms' and 'meam').
 */
import type { Pair } from '../../engine/force/types';
import { PairMeamMs } from '../meam/pairMeamMs';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'meam/ms': () => new PairMeamMs(),
  // pair_style meam is the same port with msmeamflag = 0 (pair_meam.cpp); it agrees with native at rounding level on
  // every meam case, where the engine's former own implementation differed by up to 2.3e-4 (w26meam_bi_sc)
  meam: () => new PairMeamMs(false),
};
