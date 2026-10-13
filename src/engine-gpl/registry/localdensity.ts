/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the style registry of src/MANYBODY/
 * pair_local_density.cpp (pair local/density) at commit b332f28. LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_localdensity; the names listed there must match).
 */
import type { Pair } from '../../engine/force/types';
import { PairLocalDensity } from '../localdensity/pair_local_density';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'local/density': () => new PairLocalDensity(),
};
