/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the kspace style pppm/disp
 * (src/KSPACE/pppm_disp.cpp, scope in src/engine-gpl/pppm_disp/kspace.ts) and pppm/disp/tip4p (src/KSPACE/pppm_disp_tip4p.cpp,
 * src/engine-gpl/tip4pdisp/pppm_disp_tip4p.ts). Loaded lazily through src/engine/lazyStyles.ts (group gpl_pppm_disp).
 */
import type { KSpace } from '../../engine/force/types';
import { KSpacePPPMDisp } from '../pppm_disp/kspace';
import { KSpacePPPMDispTIP4P } from '../tip4pdisp/pppm_disp_tip4p';

export const KSPACES: Record<string, () => KSpace> = {
  'pppm/disp': () => new KSpacePPPMDisp(),
  'pppm/disp/tip4p': () => new KSpacePPPMDispTIP4P(),
};
