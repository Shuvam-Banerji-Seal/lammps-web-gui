/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/CLASS2/angle_class2.cpp and angle_class2.h at commit b332f28 (Contributing author: Eric Simon, Cray).
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Bonded } from '../../engine/force/types';
import { AngleClass2Native } from '../class2/angle_class2';

/** angle_style class2 (COMPASS); merged into styles.ts through lazyStyles.ts. */
export const ANGLES: Record<string, () => Bonded> = { class2: () => new AngleClass2Native() };
