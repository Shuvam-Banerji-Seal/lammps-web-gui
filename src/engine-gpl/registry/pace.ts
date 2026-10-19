/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the ML-PACE package (src/ML-PACE:
 * pair_pace.cpp, pair_pace_extrapolation.cpp, compute_pace.cpp) and the ML-PACE library v.2025.12.4.p1 (see ../pace/).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_pace; the names listed there must match these keys).
 * pace/extrapolation is refused by name (its factory throws; listed in src/engine/refusedStyles.ts).
 */
import type { Pair } from '../../engine/force/types';
import { StyleError } from '../../engine/force/types';
import type { ComputeFactory } from '../../engine/styles';
import { PairPace } from '../pace/pair_pace';
import { ComputePACE } from '../pace/compute_pace';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  pace: () => new PairPace(),
  'pace/extrapolation': (): Pair => {
    throw new StyleError('pair_style pace/extrapolation (ACE extrapolation grades) is not supported yet');
  },
};

/** compute names. */
export const COMPUTES: Record<string, ComputeFactory> = {
  pace: (sys, id, group, args) => new ComputePACE(sys, id, group, args),
};

