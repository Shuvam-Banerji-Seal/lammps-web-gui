/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/KSPACE/ewald_disp.cpp (kspace_style ewald/disp and its alias ewald/disp/dipole, ewald_disp.h:
 * "KSpaceStyle(ewald/disp/dipole,EwaldDisp)") and src/DIPOLE/pair_lj_long_dipole_long.cpp (see ../ewald_disp). LAMMPS
 * is Copyright Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_ewald_disp).
 */
import type { KSpace, Pair } from '../../engine/force/types';
import { KSpaceEwaldDisp } from '../ewald_disp/ewald_disp';
import { PairLJLongDipoleLong } from '../ewald_disp/pair_lj_long_dipole_long';

/** kspace_style names. */
export const KSPACES: Record<string, () => KSpace> = {
  'ewald/disp': () => new KSpaceEwaldDisp('ewald/disp'),
  'ewald/disp/dipole': () => new KSpaceEwaldDisp('ewald/disp/dipole'),
};

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'lj/long/dipole/long': () => new PairLJLongDipoleLong(),
};
