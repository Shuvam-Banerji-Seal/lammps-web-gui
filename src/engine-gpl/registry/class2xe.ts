/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/CLASS2/angle_class2xe.cpp at commit b332f28. Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * ClassII-xe styles ported from LAMMPS; registered for the core through src/engine/lazyStyles.ts (group gpl_class2xe).
 */
import type { Bonded } from '../../engine/force/types';
import { AngleClass2xe } from '../class2xe/angle_class2xe';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const ANGLES: Record<string, () => Bonded> = {
  class2xe: () => new AngleClass2xe(),
};
