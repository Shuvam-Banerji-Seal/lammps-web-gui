/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ASPHERE/pair_gayberne.cpp and src/math_extra.h/.cpp at commit b332f28 (Copyright (2003) Sandia
 * Corporation, GNU General Public License; contributing author Mike Brown, SNL). Registered through
 * src/engine/lazyStyles.ts (group gpl_asphere_gayberne).
 */
import type { Pair } from '../../engine/force/types';
import { PairGayBerne } from '../asphere/pair_gayberne';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  gayberne: () => new PairGayBerne(),
};
