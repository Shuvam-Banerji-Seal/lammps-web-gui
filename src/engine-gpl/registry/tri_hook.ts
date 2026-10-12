/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ASPHERE/atom_vec_tri.cpp (AtomVecTri::data_atom_bonus) with MathExtra::inertia_triangle (src/math_extra.cpp)
 * and MathEigen::jacobi3 (src/math_eigen.cpp), at commit b332f28. Copyright Sandia Corporation (LAMMPS development
 * team, GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'tri' (lazyStyles.ts group gpl_tri): the command word 'tri' of atom_style tri loads it before
 * read_data, and the Triangles section calls triBodyHook() in styles.ts. The derivation is tri/tri_bonus.ts.
 */
import type { TriBonusHook } from '../../engine/triBonus';
import { TRI_BODY_HOOK } from '../tri/tri_bonus';

export const HOOKS: Record<string, TriBonusHook> = { tri: TRI_BODY_HOOK };
