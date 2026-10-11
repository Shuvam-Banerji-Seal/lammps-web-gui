import type { ComputeFactory } from '../styles';
import { StyleError } from '../force/types';
import { ComputeStressCartesian, ComputeStressMop, ComputeStressMopProfile } from '../compute/stressprof';

/*
 * Spatial stress profile computes (docs.lammps.org/compute_stress_cartesian.html,
 * compute_stress_mop.html, compute_stress_curvilinear.html). stress/cylinder and stress/spherical are
 * registered so that they fail with a named EngineError rather than as unknown styles.
 */
const curvilinear = (style: string): ComputeFactory => (_sys, id) => {
  throw new StyleError(`compute ${id} ${style}: not implemented in the browser engine. The native virial split across the shells was not reproduced for non-radial pairs (see src/engine/compute/stressprof.ts); use stress/cartesian or stress/mop/profile`);
};

export const COMPUTES: Record<string, ComputeFactory> = {
  'stress/cartesian': (sys, id, group, args) => new ComputeStressCartesian(sys, id, group, args),
  'stress/mop': (sys, id, group, args) => new ComputeStressMop(sys, id, group, args),
  'stress/mop/profile': (sys, id, group, args) => new ComputeStressMopProfile(sys, id, group, args),
  'stress/cylinder': curvilinear('stress/cylinder'),
  'stress/spherical': curvilinear('stress/spherical'),
};
