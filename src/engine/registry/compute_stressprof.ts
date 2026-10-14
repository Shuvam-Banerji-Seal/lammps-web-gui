import type { ComputeFactory } from '../styles';
import { ComputeStressCartesian, ComputeStressMop, ComputeStressMopProfile } from '../compute/stressprof';

/*
 * Spatial stress profile computes (docs.lammps.org/compute_stress_cartesian.html, compute_stress_mop.html).
 * stress/cylinder and stress/spherical (compute_stress_curvilinear.html) are in engine-gpl/registry/stress_curvilinear.ts.
 */

export const COMPUTES: Record<string, ComputeFactory> = {
  'stress/cartesian': (sys, id, group, args) => new ComputeStressCartesian(sys, id, group, args),
  'stress/mop': (sys, id, group, args) => new ComputeStressMop(sys, id, group, args),
  'stress/mop/profile': (sys, id, group, args) => new ComputeStressMopProfile(sys, id, group, args),
};
