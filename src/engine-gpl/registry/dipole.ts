/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the DIPOLE-package kspace style
 * pppm/dipole (src/KSPACE/pppm_dipole.cpp) and the DIPOLE-package pair lj/cut/dipole/long (src/DIPOLE/pair_lj_cut_dipole_long.cpp).
 * Loaded lazily through src/engine/lazyStyles.ts (group gpl_dipole).
 */
import type { KSpace, Pair } from '../../engine/force/types';
import { KSpacePPPMDipole } from '../dipole/pppm_dipole';
import { PairLJCutDipoleLong } from '../dipole/pair_lj_cut_dipole_long';

export const PAIRS: Record<string, () => Pair> = {
  'lj/cut/dipole/long': () => new PairLJCutDipoleLong(),
};

export const KSPACES: Record<string, () => KSpace> = {
  'pppm/dipole': () => new KSpacePPPMDipole(),
};
