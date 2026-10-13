/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registrations (FixStyle and ComputeStyle macros) of src/EXTRA-FIX/fix_gle.h,
 * src/REPLICA/fix_grem.h, src/EXTRA-FIX/fix_electron_stopping_fit.h and src/REPLICA/compute_pressure_grem.h at commit b332f28; the classes are in src/engine-gpl/extrafix/. Copyright
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_extrafix); the core does not register these names.
 */
import type { FixFactory } from '../../engine/styles';
import { FixGLE } from '../extrafix/fix_gle';
import { FixGrem } from '../extrafix/fix_grem';
import { FixElectronStoppingFit } from '../extrafix/fix_electron_stopping_fit';

export const FIXES: Record<string, FixFactory> = {
  gle: (sys, id, group, args) => new FixGLE(sys, id, group, args),
  grem: (sys, id, group, args) => new FixGrem(sys, id, group, args),
  'electron/stopping/fit': (sys, id, group, args) => new FixElectronStoppingFit(sys, id, group, args),
};
