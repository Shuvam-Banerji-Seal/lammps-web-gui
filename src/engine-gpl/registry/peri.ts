/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/PERI/pair_peri_ves.cpp, src/PERI/pair_peri_eps.cpp, src/PERI/fix_peri_neigh.cpp and
 * src/PERI/compute_plasticity_atom.cpp at commit b332f28 (see the headers of src/engine-gpl/peri/). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Peridynamic pair styles peri/ves and peri/eps and the compute plasticity/atom; registered for the core through
 * src/engine/lazyStyles.ts (group peri_ves_eps).
 */
import type { Pair } from '../../engine/force/types';
import type { ComputeFactory } from '../../engine/styles';
import { PairPeriVES } from '../peri/pair_peri_ves';
import { PairPeriEPS } from '../peri/pair_peri_eps';
import { ComputePlasticityAtom } from '../peri/compute_plasticity_atom';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const PAIRS: Record<string, () => Pair> = {
  'peri/ves': () => new PairPeriVES(),
  'peri/eps': () => new PairPeriEPS(),
};

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'plasticity/atom': (sys, id, group, args) => new ComputePlasticityAtom(sys, id, group, args),
};
