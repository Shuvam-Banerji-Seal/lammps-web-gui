/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the CORESHELL, DIELECTRIC and KSPACE pair styles listed in the style sources (src/CORESHELL/pair_coul_long_cs.cpp, pair_born_coul_dsf_cs.cpp,
 * pair_born_coul_wolf_cs.cpp, pair_born_coul_long_cs.cpp, pair_buck_coul_long_cs.cpp; src/CORESHELL/pair_coul_wolf_cs.h,
 * pair_lj_cut_coul_long_cs.h, pair_lj_class2_coul_long_cs.h; src/DIELECTRIC/pair_coul_cut_dielectric.h, pair_coul_long_dielectric.h,
 * pair_lj_cut_coul_debye_dielectric.h, pair_lj_long_coul_long_dielectric.h; src/KSPACE/pair_buck_long_coul_long.h,
 * pair_lj_long_tip4p_long.h) at commit b332f28. Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_csdiel; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import { PairCoulLongCS, PairCoulWolfCS, PairLJCutCoulLongCS, PairLJClass2CoulLongCS } from '../csdiel/coreshell';
import { PairCoulCutDielectric, PairCoulLongDielectric, PairLJCutCoulDebyeDielectric } from '../csdiel/dielectric';
import { PairLJLongCoulLongDielectric } from '../csdiel/lj_long_dielectric';
import { PairBuckLongCoulLong } from '../csdiel/buck_long';
import { PairBornCoulDsfCS, PairBornCoulWolfCS } from '../csdiel/born_dsf_wolf_cs';
import { PairBornCoulLongCS, PairBuckCoulLongCS } from '../csdiel/born_buck_long_cs';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'coul/wolf/cs': () => new PairCoulWolfCS(),
  'coul/long/cs': () => new PairCoulLongCS(),
  'lj/cut/coul/long/cs': () => new PairLJCutCoulLongCS(),
  'lj/class2/coul/long/cs': () => new PairLJClass2CoulLongCS(),
  'coul/cut/dielectric': () => new PairCoulCutDielectric(),
  'coul/long/dielectric': () => new PairCoulLongDielectric(),
  'lj/cut/coul/debye/dielectric': () => new PairLJCutCoulDebyeDielectric(),
  'lj/long/coul/long/dielectric': () => new PairLJLongCoulLongDielectric(),
  'buck/long/coul/long': () => new PairBuckLongCoulLong(),
  'born/coul/dsf/cs': () => new PairBornCoulDsfCS(),
  'born/coul/wolf/cs': () => new PairBornCoulWolfCS(),
  'born/coul/long/cs': () => new PairBornCoulLongCS(),
  'buck/coul/long/cs': () => new PairBuckCoulLongCS(),
};
