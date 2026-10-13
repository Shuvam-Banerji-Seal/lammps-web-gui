/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-PAIR/pair_coul_ctip.cpp and src/QEQ/fix_qeq_ctip.cpp at commit b332f28 (see the headers of
 * ../ctip/pair_coul_ctip.ts and ../qeq/fix_qeq_ctip.ts). Copyright (2003) Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The CTIP styles: pair coul/ctip (for pair_style hybrid/overlay eam/fs coul/ctip) and fix qeq/ctip. Merged through
 * the gpl_ctip group of src/engine/lazyStyles.ts.
 */
import type { Fix } from '../../engine/fix/fix';
import type { Pair } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { PairCoulCtip } from '../ctip/pair_coul_ctip';
import { FixQeqCtip } from '../qeq/fix_qeq_ctip';

export const PAIRS: Record<string, () => Pair> = {
  'coul/ctip': () => new PairCoulCtip(),
};

export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'qeq/ctip': (sys, id, group, args) => new FixQeqCtip(sys, id, group, args),
};
