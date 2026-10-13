/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/YAFF/pair_mm3_switch3_coulgauss_long.cpp and .h at commit b332f28 (see
 * src/engine-gpl/mm3/pair_mm3_switch3_coulgauss_long.ts). Copyright (2003) Sandia Corporation (GNU GPL).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairMM3Switch3CoulGaussLong } from '../mm3/pair_mm3_switch3_coulgauss_long';

/** MM3 / switch3 / Gaussian-Coulomb style ported from LAMMPS (YAFF package); merged through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  'mm3/switch3/coulgauss/long': () => new PairMM3Switch3CoulGaussLong(),
};
