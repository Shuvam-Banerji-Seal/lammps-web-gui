/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the EXTRA-COMPUTE styles
 * compute_entropy_atom.cpp (entropy/atom) and compute_basal_atom.cpp (basal/atom), commit b332f28. Contributing
 * author of the entropy compute: Pablo Piaggi (EPFL Lausanne). Copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_entropy_basal; the names listed there must match
 * these keys).
 */
import type { ComputeFactory } from '../../engine/styles';
import { ComputeEntropyAtom } from '../extra_compute/entropy_atom';
import { ComputeBasalAtom } from '../extra_compute/basal_atom';

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'entropy/atom': (sys, id, group, args) => new ComputeEntropyAtom(sys, id, group, args),
  'basal/atom': (sys, id, group, args) => new ComputeBasalAtom(sys, id, group, args),
};
