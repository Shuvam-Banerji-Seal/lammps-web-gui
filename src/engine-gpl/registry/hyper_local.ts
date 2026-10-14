/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the style registration of
 * src/REPLICA/fix_hyper_local.cpp (fix hyper/local). Its class is src/engine-gpl/hyper/fix_hyper_local.ts.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_hyper_local).
 */
import type { FixFactory } from '../../engine/styles';
import { FixHyperLocal } from '../hyper/fix_hyper_local';

export const FIXES: Record<string, FixFactory> = {
  'hyper/local': (sys, id, group, args) => new FixHyperLocal(sys, id, group, args),
};
