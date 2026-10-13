/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the fix style registration of
 * src/fix_ave_histo_weight.cpp (FixStyle(ave/histo/weight)). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_ave_histo_weight; the names listed there must
 * match).
 */
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { FixAveHistoWeight } from '../avehisto/fix_ave_histo_weight';

/** fix styles. */
export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'ave/histo/weight': (sys, id, group, args) => new FixAveHistoWeight(sys, id, group, args),
};
