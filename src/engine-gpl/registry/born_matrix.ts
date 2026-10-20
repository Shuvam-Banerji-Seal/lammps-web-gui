/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registration of src/EXTRA-COMPUTE/compute_born_matrix.cpp (compute born/matrix) at commit b332f28.
 * Copyright Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_born_matrix); the core does not register this name.
 */
import type { ComputeFactory } from '../../engine/styles';
import { ComputeBornMatrix } from '../extra_compute/compute_born_matrix';

export const COMPUTES: Record<string, ComputeFactory> = {
  'born/matrix': (sys, id, group, args) => new ComputeBornMatrix(sys, id, group, args),
};
