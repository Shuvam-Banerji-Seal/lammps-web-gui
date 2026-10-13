/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registrations of src/UEF (fix nvt/uef, fix npt/uef, compute temp/uef, compute pressure/uef) at
 * commit b332f28. Copyright Sandia Corporation (GNU General Public License); contributing author David Nicholson
 * (MIT). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_uef); the core does not register these names.
 */
import type { ComputeFactory, FixFactory } from '../../engine/styles';
import { newFixNPTUef, newFixNVTUef } from '../uef/fix_nh_uef';
import { ComputePressureUef } from '../uef/compute_pressure_uef';
import { ComputeTempUef } from '../uef/compute_temp_uef';

export const FIXES: Record<string, FixFactory> = {
  'nvt/uef': (sys, id, group, args) => newFixNVTUef(sys, id, group, args),
  'npt/uef': (sys, id, group, args) => newFixNPTUef(sys, id, group, args),
};

export const COMPUTES: Record<string, ComputeFactory> = {
  'temp/uef': (sys, id, group, args) => new ComputeTempUef(sys, id, group, args),
  'pressure/uef': (sys, id, group, args) => new ComputePressureUef(sys, id, group, args),
};
