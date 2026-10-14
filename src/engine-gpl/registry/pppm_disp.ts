/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the kspace style pppm/disp
 * (src/KSPACE/pppm_disp.cpp, scope in src/engine-gpl/pppm_disp/kspace.ts). Loaded lazily through src/engine/lazyStyles.ts
 * (group gpl_pppm_disp).
 */
import type { KSpace } from '../../engine/force/types';
import { KSpacePPPMDisp } from '../pppm_disp/kspace';

export const KSPACES: Record<string, () => KSpace> = {
  'pppm/disp': () => new KSpacePPPMDisp(),
};
