/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the style registry of src/PTM/compute_ptm_atom.cpp
 * (compute ptm/atom, PTM library by PM Larsen, MIT). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_ptm; the names listed there must match these keys).
 */
import type { ComputeFactory } from '../../engine/styles';
import { ComputePtmAtom } from '../ptm/compute_ptm_atom';

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'ptm/atom': (sys, id, group, args) => new ComputePtmAtom(sys, id, group, args),
};
