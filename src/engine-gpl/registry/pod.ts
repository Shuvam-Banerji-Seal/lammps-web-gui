/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/ML-POD (pair_pod.cpp, eapod.cpp,
 * compute_pod_atom.cpp), GPL-2.0. Registered through src/engine/lazyStyles.ts. Evaluation only; fitting is not ported.
 */
import type { Pair } from '../../engine/force/types';
import type { ComputeFactory } from '../../engine/styles';
import { PairPOD } from '../pod/pair_pod';
import { ComputePODAtom } from '../pod/compute_pod_atom';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  pod: () => new PairPOD(),
};

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'pod/atom': (sys, id, group, args) => new ComputePODAtom(sys, id, group, args),
};
