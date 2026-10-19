/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the ML-PACE package (src/ML-PACE:
 * pair_pace.cpp, pair_pace_extrapolation.cpp, compute_pace.cpp) and the ML-PACE library v.2025.12.4.p1 (see ../pace/).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_pace; the names listed there must match these keys).
 * pace/extrapolation and compute pace are refused by name (their factories throw; listed in src/engine/refusedStyles.ts).
 */
import type { Pair } from '../../engine/force/types';
import { StyleError } from '../../engine/force/types';
import type { ComputeFactory } from '../../engine/styles';
import { PairPace } from '../pace/pair_pace';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  pace: () => new PairPace(),
  'pace/extrapolation': (): Pair => {
    throw new StyleError('pair_style pace/extrapolation (ACE extrapolation grades) is not supported yet');
  },
};

/** compute names: compute pace (per-atom ACE descriptors) is not supported yet. */
export const COMPUTES: Record<string, ComputeFactory> = {
  pace: () => {
    throw new StyleError('compute pace (ACE descriptors) is not supported yet');
  },
};

