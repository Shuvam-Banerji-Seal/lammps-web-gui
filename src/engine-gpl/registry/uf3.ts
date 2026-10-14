/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ML-UF3/pair_uf3.cpp, pair_uf3.h, uf3_bspline_basis2.cpp, uf3_bspline_basis3.cpp at commit b332f28.
 * Registered through src/engine/lazyStyles.ts (group gpl_uf3).
 */
import type { Pair } from '../../engine/force/types';
import { PairUF3 } from '../uf3/pair_uf3';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  uf3: () => new PairUF3(),
};
