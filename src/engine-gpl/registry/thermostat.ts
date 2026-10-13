/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registrations of src/EXTRA-FIX/fix_temp_csvr.cpp and src/EXTRA-FIX/fix_temp_csld.cpp (fix temp/csvr,
 * fix temp/csld) at commit b332f28. Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_thermostat); the core does not register these two names.
 */
import type { FixFactory } from '../../engine/styles';
import { FixTempCSVR } from '../thermostat/fix_temp_csvr';
import { FixTempCSLD } from '../thermostat/fix_temp_csld';

export const FIXES: Record<string, FixFactory> = {
  'temp/csld': (sys, id, group, args) => new FixTempCSLD(sys, id, group, args),
  'temp/csvr': (sys, id, group, args) => new FixTempCSVR(sys, id, group, args),
};
