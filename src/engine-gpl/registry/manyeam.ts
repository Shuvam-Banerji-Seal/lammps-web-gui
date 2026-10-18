/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the pair styles adp (src/MANYBODY/pair_adp.cpp,
 * pair_adp.h), eam/cd and eam/cd/old (src/MANYBODY/pair_eam_cd.cpp, pair_eam_cd.h) and eam/he (src/MANYBODY/pair_eam_he.cpp,
 * pair_eam_he.h). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_manyeam; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import { PairADP } from '../manyeam/pair_adp';
import { PairEAMCD } from '../manyeam/pair_eam_cd';
import { PairEAMHE } from '../manyeam/pair_eam_he';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  adp: () => new PairADP(),
  'eam/cd': () => new PairEAMCD('eam/cd'),
  'eam/cd/old': () => new PairEAMCD('eam/cd/old'),
  'eam/he': () => new PairEAMHE(),
};
