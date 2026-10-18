/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MC/fix_charge_regulation.cpp (and .h) at commit b332f28 (Tine Curk and Jiaxing Yuan; the LAMMPS
 * developers). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * Monte Carlo fixes ported from LAMMPS; registered for the core through src/engine/lazyStyles.ts (group gpl_mc).
 */
import type { FixFactory } from '../../engine/styles';
import { FixChargeRegulation } from '../mc/fix_charge_regulation';
import { FixGEMC } from '../mc/fix_gemc';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  'charge/regulation': (sys, id, group, args) => new FixChargeRegulation(sys, id, group, args),
  gemc: (sys, id, group, args) => new FixGEMC(sys, id, group, args),
};
