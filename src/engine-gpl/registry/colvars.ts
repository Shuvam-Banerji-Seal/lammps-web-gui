/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: fix colvars (src/COLVARS/fix_colvars.cpp,
 * colvarproxy_lammps.cpp) and the Colvars library it drives (lib/colvars). Copyright Sandia Corporation and the Colvars
 * developers (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_colvars; the names listed there must match these keys).
 */
import type { FixFactory } from '../../engine/styles';
import { FixColvars } from '../colvars/fix_colvars';

export const FIXES: Record<string, FixFactory> = {
  colvars: (sys, id, group, args) => new FixColvars(sys, id, group, args),
};
