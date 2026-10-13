/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the DIPOLE-package kspace style
 * pppm/dipole (src/KSPACE/pppm_dipole.cpp). Loaded lazily through src/engine/lazyStyles.ts (group gpl_dipole).
 */
import type { KSpace } from '../../engine/force/types';
import { KSpacePPPMDipole } from '../dipole/pppm_dipole';

export const KSPACES: Record<string, () => KSpace> = {
  'pppm/dipole': () => new KSpacePPPMDipole(),
};
