/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/ML-POD (pair_pod.cpp, eapod.cpp),
 * GPL-2.0. Registered through src/engine/lazyStyles.ts. Evaluation only; fitting is not ported.
 */
import type { Pair } from '../../engine/force/types';
import { PairPOD } from '../pod/pair_pod';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  pod: () => new PairPOD(),
};
