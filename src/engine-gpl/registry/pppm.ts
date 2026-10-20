/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/pppm.cpp (set_grid_global, adjust_gewald, compute_gf_ad, poisson_ad, fieldforce_ad) at commit
 * b332f28; the grid and G-ewald choice is in ../pppm/grid.ts and the analytic-differentiation style in ../pppm/kspace_ad.ts.
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entries of group gpl_pppm (src/engine/lazyStyles.ts): the command words of the PPPM styles that the core
 * PPPM reaches through PppmGplHook (triclinic grid choice, kspace_modify diff ad). The names listed there must match
 * these keys. Every key is the same helper.
 */
import type { PppmGplHook } from '../../engine/force/kspace/pppm';
import { PPPM_HOOK } from '../pppm/kspace_ad';

export const HOOKS: Record<string, PppmGplHook> = {
  pppm: PPPM_HOOK,
  'pppm/cg': PPPM_HOOK,
  'pppm/dielectric': PPPM_HOOK,
  'pppm/electrode': PPPM_HOOK,
  'pppm/rk': PPPM_HOOK,
  'pppm/tip4p': PPPM_HOOK,
};
