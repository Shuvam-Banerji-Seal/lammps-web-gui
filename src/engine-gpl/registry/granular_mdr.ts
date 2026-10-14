/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/gran_sub_mod_normal.cpp and src/GRANULAR/fix_granular_mdr.cpp (the mdr normal model of pair_style
 * granular), at commit b332f28; see src/engine-gpl/granular/mdr/ for the file-level sources. Copyright (LAMMPS
 * development team, GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'mdr' (lazyStyles.ts group gpl_granular_mdr): a pair_coeff or fix wall/gran/region with the mdr
 * normal model loads it, and the core granular pair creates its MDR model through this hook.
 */
import type { GranularMdrHook } from '../../engine/force/pair/granular';
import { MdrModel } from '../granular/mdr/model';

export const HOOKS: Record<string, GranularMdrHook> = {
  mdr: { create: (mat) => new MdrModel(mat) },
};
