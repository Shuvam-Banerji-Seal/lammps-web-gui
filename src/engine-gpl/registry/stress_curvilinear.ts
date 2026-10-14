/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_stress_cylinder.cpp and src/EXTRA-COMPUTE/compute_stress_spherical.cpp at commit
 * b332f28 (style registration). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 */
import type { ComputeFactory } from '../../engine/styles';
import { ComputeStressCylinder, ComputeStressSpherical } from '../compute/stress_curvilinear';

export const COMPUTES: Record<string, ComputeFactory> = {
  'stress/cylinder': (sys, id, group, args) => new ComputeStressCylinder(sys, id, group, args),
  'stress/spherical': (sys, id, group, args) => new ComputeStressSpherical(sys, id, group, args),
};
