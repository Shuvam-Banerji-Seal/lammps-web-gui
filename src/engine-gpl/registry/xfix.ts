/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the FixStyle registrations of src/EXTRA-FIX/fix_nvk.h, fix_ffl.h, fix_gld.h, fix_smd.h,
 * fix_ave_correlate_long.h, fix_deform_pressure.h, fix_efield_tip4p.h, fix_nonaffine_displacement.h and src/fix_press_langevin.h,
 * src/fix_set.h, src/fix_pair.h at commit b332f28. Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_xfix; the names listed there must match these keys).
 */
import type { FixFactory } from '../../engine/styles';
import { FixNVK } from '../xfix/fix_nvk';
import { FixFFL } from '../xfix/fix_ffl';
import { FixGLD } from '../xfix/fix_gld';
import { FixSMD } from '../xfix/fix_smd';
import { FixAveCorrelateLong } from '../xfix/fix_ave_correlate_long';
import { FixSet } from '../xfix/fix_set';
import { FixNonaffineDisplacement } from '../xfix/fix_nonaffine_displacement';
import { FixPair } from '../xfix/fix_pair';
import { FixEfieldTIP4P } from '../xfix/fix_efield_tip4p';
import { FixPressLangevin } from '../xfix/fix_press_langevin';
import { FixDeformPressure } from '../xfix/fix_deform_pressure';

export const FIXES: Record<string, FixFactory> = {
  nvk: (sys, id, group, args) => new FixNVK(sys, id, group, args),
  ffl: (sys, id, group, args) => new FixFFL(sys, id, group, args),
  gld: (sys, id, group, args) => new FixGLD(sys, id, group, args),
  smd: (sys, id, group, args) => new FixSMD(sys, id, group, args),
  'ave/correlate/long': (sys, id, group, args) => new FixAveCorrelateLong(sys, id, group, args),
  set: (sys, id, group, args) => new FixSet(sys, id, group, args),
  'nonaffine/displacement': (sys, id, group, args) => new FixNonaffineDisplacement(sys, id, group, args),
  pair: (sys, id, group, args) => new FixPair(sys, id, group, args),
  'efield/tip4p': (sys, id, group, args) => new FixEfieldTIP4P(sys, id, group, args),
  'press/langevin': (sys, id, group, args) => new FixPressLangevin(sys, id, group, args),
  'deform/pressure': (sys, id, group, args) => new FixDeformPressure(sys, id, group, args),
};
