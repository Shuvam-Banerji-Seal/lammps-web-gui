/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the style registry of fix nvt/sllod
 * (src/fix_nvt_sllod.cpp). Contributing author (from the source): Pieter in 't Veld (SNL). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_sllod; the names are listed there).
 */
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { FixNVTSllod } from '../fix/fix_nvt_sllod';

/** fix styles. */
export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'nvt/sllod': (sys, id, group, args) => new FixNVTSllod(sys, id, group, args),
};
