/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/reset_atoms_id.cpp at commit b332f28 (see reset_atoms/spatial_sort.ts). Copyright Sandia Corporation
 * (LAMMPS development team, GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'reset_atoms' (lazyStyles.ts group gpl_reset_atoms): the command word reset_atoms loads it, and
 * commands/reset_atoms.ts reads the spatial order through gplHook('reset_atoms') for reset_atoms id sort yes.
 */
import type { SpatialSortHook } from '../../engine/commands/reset_atoms';
import { spatialSortOrder } from '../reset_atoms/spatial_sort';

export const HOOKS: Record<string, SpatialSortHook> = { reset_atoms: spatialSortOrder };
