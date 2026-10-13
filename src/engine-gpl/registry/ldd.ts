/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the style registry of src/BOCS/pair_ldd.cpp
 * (pair ldd) at commit b332f28. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). BOCS package authors (from the
 * source): Michael R. DeLyser and Maria C. Lesniewski (Pennsylvania State University).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_ldd; the names listed there must match).
 */
import type { Pair } from '../../engine/force/types';
import { PairLdd } from '../ldd/pair_ldd';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  ldd: () => new PairLdd(),
};
