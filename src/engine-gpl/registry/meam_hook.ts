/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MEAM/pair_meam.cpp (PairMEAM, the
 * msmeamflag = 0 case) and the MEAM routines it calls (see src/engine-gpl/meam/meamModel.ts). GPL-2.0
 * (src/engine-gpl/LICENSE). Registered through src/engine/lazyStyles.ts (group gpl_meam_hook).
 *
 * The lazy HOOKS entry 'meam': the core pair_style meam (src/engine/force/pair/meam.ts) hands a parameter set it refuses
 * (the partly screened hcp second shell, nn2 = 1) to this standard MEAM pair style. The core keeps every other set.
 */
import type { Pair } from '../../engine/force/types';
import { PairMeamMs } from '../meam/pairMeamMs';

/** HOOKS: the factory of the standard MEAM pair (msmeamflag = 0), read by the core through gplHook('meam'). */
export const HOOKS: Record<string, () => Pair> = {
  meam: () => new PairMeamMs(false),
};
