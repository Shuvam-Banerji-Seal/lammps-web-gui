/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/CLASS2/dihedral_class2xe.cpp and dihedral_class2xe.h, at commit b332f28.
 * Authored by Josh Kemppainen (Michigan Technological University) per the C++ file header.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Bonded } from '../../engine/force/types';
import { DihedralClass2xe } from '../class2xe/dihedral_class2xe';

/** dihedral_style class2xe (ClassII-xe); merged into styles.ts through lazyStyles.ts. */
export const DIHEDRALS: Record<string, () => Bonded> = { class2xe: () => new DihedralClass2xe() };
