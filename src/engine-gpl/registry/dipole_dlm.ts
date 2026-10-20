/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the update dipole/dlm loop of src/fix_nve_sphere.cpp, src/fix_nh_sphere.cpp and src/BPM/fix_nve_bpm_sphere.cpp
 * at commit b332f28; the routine is in ../dipole/dlm.ts. Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'dipole/dlm' (lazyStyles.ts group gpl_dipole_dlm): the update value of fix nve/sphere,
 * nve/bpm/sphere and nvt/npt/nph/sphere. The core reads it through gplHook('dipole/dlm') (src/engine/gplHooks.ts).
 */
import type { DipoleDlmHook } from '../../engine/fix/nve_sphere';
import { dlmUpdate } from '../dipole/dlm';

export const HOOKS: Record<string, DipoleDlmHook> = { 'dipole/dlm': { dlmUpdate } };
