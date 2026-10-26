/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/ML-POD (pair_pod.cpp, eapod.cpp,
 * compute_pod_atom.cpp, compute_podd_atom.cpp), GPL-2.0. Registered through src/engine/lazyStyles.ts. The fitpod command is the lazy group gpl_fitpod (registry/fitpod.ts).
 */
import type { Pair } from '../../engine/force/types';
import type { ComputeFactory } from '../../engine/styles';
import { PairPOD } from '../pod/pair_pod';
import { ComputePODAtom } from '../pod/compute_pod_atom';
import { ComputePODDAtom } from '../pod/compute_podd_atom';
import { ComputePODGlobal } from '../pod/compute_pod_global';
import { ComputePODLocal } from '../pod/compute_pod_local';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  pod: () => new PairPOD(),
};

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'pod/atom': (sys, id, group, args) => new ComputePODAtom(sys, id, group, args),
  'podd/atom': (sys, id, group, args) => new ComputePODDAtom(sys, id, group, args),
  'pod/global': (sys, id, group, args) => new ComputePODGlobal(sys, id, group, args),
  'pod/local': (sys, id, group, args) => new ComputePODLocal(sys, id, group, args),
};
