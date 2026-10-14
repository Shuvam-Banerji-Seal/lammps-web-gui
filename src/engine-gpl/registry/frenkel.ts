/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_frenkel.cpp and compute_frenkel.h at commit b332f28 (see src/engine-gpl/extra_compute/frenkel.ts).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_frenkel; the names listed there must match).
 */
import type { Compute } from '../../engine/compute/compute';
import type { ComputeFactory } from '../../engine/styles';
import { ComputeFrenkel } from '../extra_compute/frenkel';

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  frenkel: (sys, id, group, args): Compute => new ComputeFrenkel(sys, id, group, args),
};
