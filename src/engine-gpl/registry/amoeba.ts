/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/pair_amoeba.cpp, pair_hippo.cpp, angle_amoeba.cpp at commit b332f28. Copyright (2003) Sandia
 * Corporation, distributed under the GNU General Public License (the LAMMPS file headers). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * AMOEBA package styles: pair amoeba (stage 2a: hal; src/engine-gpl/amoeba), and refusals for the styles not ported yet,
 * so the core shows the reason instead of "unknown style". Registered through src/engine/lazyStyles.ts (group gpl_amoeba);
 * the refusal-only names are listed in src/engine/refusedStyles.ts.
 */
import { StyleError } from '../../engine/force/types';
import type { Bonded } from '../../engine/force/types';
import type { Pair } from '../../engine/force/types';
import { PairAmoeba } from '../amoeba/pair_amoeba';
import { AngleAmoeba } from '../amoeba/angle_amoeba';

const refuse = (what: string): never => {
  throw new StyleError(`${what} is not ported yet`);
};

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  amoeba: () => new PairAmoeba(),
  hippo: () => refuse('pair hippo: the HIPPO terms'),
};

/** angle_style names. */
export const ANGLES: Record<string, () => Bonded> = {
  amoeba: () => new AngleAmoeba(),
};
