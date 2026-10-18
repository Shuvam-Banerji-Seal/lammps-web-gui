/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/CLASS2/dihedral_class2.cpp and dihedral_class2.h at commit b332f28 (Contributing author: Eric Simon,
 * Cray). Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Bonded } from '../../engine/force/types';
import { DihedralClass2 } from '../class2/dihedral_class2';

/** dihedral_style class2 (COMPASS); merged into styles.ts through lazyStyles.ts. */
export const DIHEDRALS: Record<string, () => Bonded> = { class2: () => new DihedralClass2() };
