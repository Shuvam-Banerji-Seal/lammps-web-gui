/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the ELECTRODE package's style registry (src/ELECTRODE/ewald_electrode.h and fix_electrode_conp.h, the
 * KSpaceStyle and FixStyle macros) at commit b332f28. ELECTRODE contributing authors: Ludwig Ahrens-Iwers (TUHH),
 * Shern Tee (UQ), Robert Meissner (TUHH). Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_electrode): kspace_style ewald/electrode and
 * fix electrode/conp. The names listed there must match these keys.
 */
import type { FixFactory } from '../../engine/styles';
import type { KSpace } from '../../engine/force/types';
import { KSpaceEwaldElectrode } from '../electrode/ewald_electrode';
import { FixElectrodeConp } from '../electrode/fix_electrode_conp';

/** Fix style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'electrode/conp': (sys, id, group, args) => new FixElectrodeConp(sys, id, group, args),
};

/** KSpace style name -> factory. */
export const KSPACES: Record<string, () => KSpace> = {
  'ewald/electrode': () => new KSpaceEwaldElectrode(),
};
