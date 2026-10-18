/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the pair styles lj/cut/sphere
 * (src/EXTRA-PAIR/pair_lj_cut_sphere.cpp), lj/expand/sphere (pair_lj_expand_sphere.cpp), lj/expand/coul/long
 * (pair_lj_expand_coul_long.cpp), lj/gromacs/coul/gromacs (pair_lj_gromacs_coul_gromacs.cpp), lj/pirani (pair_lj_pirani.cpp),
 * morse/smooth/linear (pair_morse_smooth_linear.cpp) and nm/cut, nm/cut/split, nm/cut/coul/cut, nm/cut/coul/long
 * (pair_nm_cut*.cpp). Contributing authors are named in each source file. LAMMPS is Copyright (2003) Sandia Corporation
 * and distributed under the GNU General Public License, version 2; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_xpair2; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import { PairLJCutSphere } from '../xpair2/lj_cut_sphere';
import { PairLJExpandSphere } from '../xpair2/lj_expand_sphere';
import { PairLJExpandCoulLong } from '../xpair2/lj_expand_coul_long';
import { PairLJGromacsCoulGromacs } from '../xpair2/lj_gromacs_coul_gromacs';
import { PairLJPirani } from '../xpair2/lj_pirani';
import { PairMorseSmoothLinear } from '../xpair2/morse_smooth_linear';
import { PairNMCut, PairNMCutSplit, PairNMCutCoulCut, PairNMCutCoulLong } from '../xpair2/nm_cut';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'lj/cut/sphere': () => new PairLJCutSphere(),
  'lj/expand/sphere': () => new PairLJExpandSphere(),
  'lj/expand/coul/long': () => new PairLJExpandCoulLong(),
  'lj/gromacs/coul/gromacs': () => new PairLJGromacsCoulGromacs(),
  'lj/pirani': () => new PairLJPirani(),
  'morse/smooth/linear': () => new PairMorseSmoothLinear(),
  'nm/cut': () => new PairNMCut(),
  'nm/cut/split': () => new PairNMCutSplit(),
  'nm/cut/coul/cut': () => new PairNMCutCoulCut(),
  'nm/cut/coul/long': () => new PairNMCutCoulLong(),
};
