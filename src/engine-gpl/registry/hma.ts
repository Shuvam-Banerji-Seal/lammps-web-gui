/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_hma.cpp, src/EXTRA-PAIR/pair_lj_smooth_linear.cpp and src/pair.cpp (hessian_twobody)
 * at commit b332f28 (see extra_compute/hma.ts). Copyright the LAMMPS development team (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Compute hma, loaded lazily through src/engine/lazyStyles.ts (group gpl_hma).
 */
import type { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { ComputeHMA } from '../extra_compute/hma';

export const COMPUTES: Record<string, (sys: System, id: string, group: string, args: string[]) => Compute> = {
  hma: (sys, id, group, args) => new ComputeHMA(sys, id, group, args),
};
