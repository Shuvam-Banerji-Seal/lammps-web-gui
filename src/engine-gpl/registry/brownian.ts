/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BROWNIAN/fix_brownian_base.cpp, fix_brownian.cpp, fix_brownian_sphere.cpp, fix_brownian_asphere.cpp,
 * fix_propel_self.cpp, fix_align_self.cpp and their headers at commit b332f28 (Sam Cameron, University of Bristol,
 * Arthur Straube, Zuse Institute Berlin, and the LAMMPS developers). Copyright Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * BROWNIAN package fixes ported from LAMMPS; registered for the core through src/engine/lazyStyles.ts (group brownian).
 */
import type { FixFactory } from '../../engine/styles';
import { FixBrownian } from '../brownian/fix_brownian';
import { FixBrownianAsphere } from '../brownian/fix_brownian_asphere';
import { FixBrownianSphere } from '../brownian/fix_brownian_sphere';
import { FixAlignSelf } from '../brownian/fix_align_self';
import { FixPropelSelf } from '../brownian/fix_propel_self';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  brownian: (sys, id, group, args) => new FixBrownian(sys, id, group, args),
  'brownian/sphere': (sys, id, group, args) => new FixBrownianSphere(sys, id, group, args),
  'brownian/asphere': (sys, id, group, args) => new FixBrownianAsphere(sys, id, group, args),
  'propel/self': (sys, id, group, args) => new FixPropelSelf(sys, id, group, args),
  'align/self': (sys, id, group, args) => new FixAlignSelf(sys, id, group, args),
};
