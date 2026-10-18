/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/pppm_stagger.cpp and pppm_stagger.h, src/KSPACE/pppm_rk.cpp and pppm_rk.h at commit b332f28
 * (see the headers of ../pppmx/pppm_stagger.ts and ../pppmx/pppm_rk.ts for the contributing authors). Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_pppmx; the names listed there must match these keys).
 */
import type { KSpace } from '../../engine/force/types';
import { KSpacePPPMStagger } from '../pppmx/pppm_stagger';
import { KSpacePPPMRK } from '../pppmx/pppm_rk';

/** kspace_style names. */
export const KSPACES: Record<string, () => KSpace> = {
  'pppm/stagger': () => new KSpacePPPMStagger(),
  'pppm/rk': () => new KSpacePPPMRK(),
};
