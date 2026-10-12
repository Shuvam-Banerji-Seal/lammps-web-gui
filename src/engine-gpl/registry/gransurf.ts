/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registrations of src/GRANSURF/pair_surf_granular.cpp and src/GRANSURF/fix_surface_local.cpp at
 * commit b332f28 (pair_style surf/granular; fix surface/local). Copyright Sandia Corporation (LAMMPS development team,
 * GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gransurf); the names there must match these keys.
 */
import type { Pair } from '../../engine/force/types';
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { PairSurfGranular } from '../gransurf/pair_surf_granular';
import { FixSurfaceLocal } from '../gransurf/fix_surface_local';
import { StyleError } from '../../engine/force/types';

export const PAIRS: Record<string, () => Pair> = {
  'surf/granular': () => new PairSurfGranular(),
};

export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'surface/local': (sys, id, group, args) => new FixSurfaceLocal(sys, id, group, args),
  // fix surface/global (surfaces from molecule files or STL) is not ported: refused with its reason
  'surface/global': () => {
    throw new StyleError('fix surface/global is not supported yet (surfaces from molecule or STL files; use fix surface/local with Lines or Triangles in the data file)');
  },
};
