/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the style registry of src/DPD-SMOOTH for
 * fix meso/move (src/DPD-SMOOTH/fix_meso_move.cpp). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing
 * author (from the source): Morteza Jalalvand (IASBS).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_dpdsmooth; the names listed there must match).
 */
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { FixMesoMove } from '../dpdsmooth/fix_meso_move';

/** fix styles. */
export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'meso/move': (sys, id, group, args) => new FixMesoMove(sys, id, group, args),
};
