/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registrations of src/ASPHERE/pair_line_lj.cpp, pair_tri_lj.cpp, fix_nve_line.cpp and
 * fix_nve_tri.cpp at commit b332f28 (pair_style line/lj, tri/lj; fix nve/line, nve/tri). Copyright Sandia
 * Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group line_tri); the names there must match these keys.
 */
import type { Pair } from '../../engine/force/types';
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { PairLineLJ } from '../line/pair_line_lj';
import { FixNVELine } from '../line/fix_nve_line';
import { PairTriLJ } from '../tri/pair_tri_lj';
import { FixNVETri } from '../tri/fix_nve_tri';

export const PAIRS: Record<string, () => Pair> = {
  'line/lj': () => new PairLineLJ(),
  'tri/lj': () => new PairTriLJ(),
};

export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'nve/line': (sys, id, group, args) => new FixNVELine(sys, id, group, args),
  'nve/tri': (sys, id, group, args) => new FixNVETri(sys, id, group, args),
};
