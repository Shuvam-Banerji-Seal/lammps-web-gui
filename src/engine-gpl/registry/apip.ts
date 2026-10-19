/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the APIP package (src/APIP:
 * pair_eam_apip.cpp, pair_eam_fs_apip.cpp, pair_pace_apip.cpp, pair_pace_fast_apip.cpp, pair_pace_precise_apip.cpp)
 * at commit b332f28 (see ../apip/). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_apip; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import type { FixFactory } from '../../engine/styles';
import { FixAtomWeightAPIP } from '../apip/fix_atom_weight_apip';
import { PairEAMAPIP, PairEAMFSAPIP } from '../apip/pair_eam_apip';
import { PairPaceAPIP, PairPaceFastAPIP, PairPacePreciseAPIP } from '../apip/pair_pace_apip';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'eam/apip': () => new PairEAMAPIP(),
  'eam/fs/apip': () => new PairEAMFSAPIP(),
  'pace/apip': () => new PairPaceAPIP(),
  'pace/precise/apip': () => new PairPacePreciseAPIP(),
  'pace/fast/apip': () => new PairPaceFastAPIP(),
};

/** fix names. */
export const FIXES: Record<string, FixFactory> = {
  'atom_weight/apip': (sys, id, group, args) => new FixAtomWeightAPIP(sys, id, group, args),
};
