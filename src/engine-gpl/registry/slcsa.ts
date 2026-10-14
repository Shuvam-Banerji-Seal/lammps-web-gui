/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-COMPUTE/compute_slcsa_atom.cpp
 * (compute slcsa/atom), commit b332f28. Contributing author: Paul Lafourcade (CEA-DAM-DIF). Copyright Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_slcsa; the name listed there must match this key).
 */
import type { ComputeFactory } from '../../engine/styles';
import { ComputeSLCSAAtom } from '../slcsa/compute_slcsa_atom';

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'slcsa/atom': (sys, id, group, args) => new ComputeSLCSAAtom(sys, id, group, args),
};
