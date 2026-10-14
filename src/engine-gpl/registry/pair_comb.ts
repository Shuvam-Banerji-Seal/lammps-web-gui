/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_comb.cpp and src/MANYBODY/pair_comb.h (style comb), and
 * src/MANYBODY/fix_qeq_comb.cpp and src/MANYBODY/fix_qeq_comb.h (style qeq/comb), at commit b332f28.
 * Authors credited in the C++ file headers: Tzu-Ray Shan (U Florida, then Sandia); Ray Shan (Sandia).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 */
import type { Fix } from '../../engine/fix/fix';
import type { Pair } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { PairComb } from '../comb/comb';
import { PairComb3 } from '../comb3/pair_comb3';
import { FixQeqComb } from '../comb/fix_qeq_comb';

/** COMB styles (comb, comb3 stage A) and the charge-equilibration fix, ported from LAMMPS (pair_comb, pair_comb3, fix_qeq_comb); merged through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  comb: () => new PairComb(),
  comb3: () => new PairComb3(),
};

export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'qeq/comb': (sys, id, group, args) => new FixQeqComb(sys, id, group, args),
};
