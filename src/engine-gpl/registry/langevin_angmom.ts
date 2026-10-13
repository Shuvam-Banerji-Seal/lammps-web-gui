/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/fix_langevin.cpp (FixLangevin::angmom_thermostat) and src/math_extra.cpp (MathExtra::mq_to_omega) at
 * commit b332f28; the routine is in ../thermostat/langevin_angmom.ts. Copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'angmom' (lazyStyles.ts group gpl_langevin_angmom): the command word of fix langevin keyword angmom.
 * The core reads it through gplHook('angmom') (src/engine/gplHooks.ts).
 */
import type { LangevinAngmomHook } from '../../engine/fix/thermostats';
import { angmomTorques } from '../thermostat/langevin_angmom';

export const HOOKS: Record<string, LangevinAngmomHook> = { angmom: { angmomTorques } };
