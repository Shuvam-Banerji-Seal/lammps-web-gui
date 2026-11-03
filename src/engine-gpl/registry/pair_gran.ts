/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/pair_gran_hooke_history.cpp, pair_gran_hertz_history.cpp and pair_gran_hooke.cpp at commit
 * b332f28 (Contributing authors: Leo Silbert (SNL), Gary Grest (SNL)). Copyright (2003) Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairGranNative } from '../granular/pair_gran_native';

/** pair_style gran/hooke, gran/hooke/history and gran/hertz/history; merged into styles.ts through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  'gran/hooke': () => new PairGranNative('gran/hooke'),
  'gran/hooke/history': () => new PairGranNative('gran/hooke/history'),
  'gran/hertz/history': () => new PairGranNative('gran/hertz/history'),
};
