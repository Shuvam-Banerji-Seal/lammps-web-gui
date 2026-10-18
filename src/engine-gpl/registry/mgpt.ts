/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MGPT/pair_mgpt.h (PairStyle(mgpt,PairMGPT)) at commit b332f28 (the MGPT package). Contributing authors:
 * Tomas Oppelstrup and John Moriarty (LLNL). Copyright (2015) Lawrence Livermore National Security, LLC,
 * LLNL-CODE-674031 (GNU General Public License version 2; the LLNL notice is in ../mgpt/NOTICE.llnl).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairMGPT } from '../mgpt/pair_mgpt';

/** pair_style mgpt ported from LAMMPS (src/MGPT); merged into styles.ts through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  mgpt: () => new PairMGPT(),
};
