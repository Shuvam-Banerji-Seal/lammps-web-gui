/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the ASPHERE package styles pair granular/superellipsoid
 * (src/ASPHERE/pair_granular_superellipsoid.cpp, commit b332f28), GPL-2.0 (src/engine-gpl/LICENSE). Authors are named in the
 * ported files (Jacopo Bilotto, Jibril B. Coulibaly).
 *
 * Superellipsoid styles registered for the core through src/engine/lazyStyles.ts (group superellipsoid).
 */
import type { Pair } from '../../engine/force/types';
import { PairGranularSuperellipsoid } from '../superellipsoid/pair_granular_superellipsoid';

export const PAIRS: Record<string, () => Pair> = {
  'granular/superellipsoid': () => new PairGranularSuperellipsoid(),
};
