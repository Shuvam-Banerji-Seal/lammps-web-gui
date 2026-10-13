/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_filter_corotate.cpp (and .h) at commit b332f28 (Christoph Junghans, Jonathan Kaiser and
 * Arno Bohlen, the LAMMPS developers). Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * fix filter/corotate is not ported yet: this module only refuses it, with the reason. Registered for the core
 * through src/engine/lazyStyles.ts (group gpl_filtercorotate) and listed in src/engine/refusedStyles.ts.
 */
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';

const REASON = 'fix filter/corotate is not supported by the browser engine: the native fix needs the respa force hooks '
  + '(pre_force_respa, post_force_respa and the level force copies of the rRESPA integrator, fix_filter_corotate.cpp '
  + 'setmask and setup), which run/respa.ts does not call for fixes, and the cluster finder (find_clusters, '
  + 'general_cluster) is not ported';

/** Style name -> factory that refuses (the name is listed in refusedStyles.ts). */
export const FIXES: Record<string, FixFactory> = {
  'filter/corotate': () => { throw new StyleError(REASON); },
};
