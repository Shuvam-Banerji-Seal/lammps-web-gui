/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the kspace style esp and the pair
 * styles coul/esp and lj/cut/coul/esp (src/KSPACE/esp.cpp, pair_coul_esp.cpp, pair_lj_cut_coul_esp.cpp; scope in
 * src/engine-gpl/esp/kspace_esp.ts). Loaded lazily through src/engine/lazyStyles.ts (group gpl_esp).
 */
import type { KSpace, Pair } from '../../engine/force/types';
import { KSpaceESP } from '../esp/kspace_esp';
import { PairCoulEsp } from '../esp/pair_coul_esp';
import { PairLJCutCoulEsp } from '../esp/pair_lj_cut_coul_esp';

export const KSPACES: Record<string, () => KSpace> = {
  esp: () => new KSpaceESP(),
};

export const PAIRS: Record<string, () => Pair> = {
  'coul/esp': () => new PairCoulEsp(),
  'lj/cut/coul/esp': () => new PairLJCutCoulEsp(),
};
