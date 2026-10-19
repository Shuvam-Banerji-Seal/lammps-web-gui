/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANIFOLD/fix_nve_manifold_rattle.cpp, fix_nvt_manifold_rattle.cpp, fix_manifoldforce.cpp and the
 * manifold_*.cpp files at commit b332f28 (Stefan Paquay, Eindhoven University of Technology). GNU General Public
 * License. This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * MANIFOLD package fixes ported from LAMMPS; registered for the core through src/engine/lazyStyles.ts (group gpl_manifold).
 */
import type { FixFactory } from '../../engine/styles';
import { FixNVEManifoldRattle } from '../manifold/fix_nve_manifold_rattle';
import { FixNVTManifoldRattle } from '../manifold/fix_nvt_manifold_rattle';
import { FixManifoldForce } from '../manifold/fix_manifoldforce';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  'nve/manifold/rattle': (sys, id, group, args) => new FixNVEManifoldRattle(sys, id, group, args),
  'nvt/manifold/rattle': (sys, id, group, args) => new FixNVTManifoldRattle(sys, id, group, args),
  manifoldforce: (sys, id, group, args) => new FixManifoldForce(sys, id, group, args),
};
