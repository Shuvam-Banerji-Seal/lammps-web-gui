/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/KSPACE/msm.cpp, msm_cg.cpp, pair_coul_msm.cpp, pair_lj_cut_coul_msm.cpp, pair_born_coul_msm.cpp,
 * pair_buck_coul_msm.cpp, pair_lj_charmm_coul_msm.cpp and src/CG-SPICA/pair_lj_spica_coul_msm.cpp (kspace_style msm and
 * msm/cg and their pair styles, see ../msm). LAMMPS is Copyright Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_msm).
 */
import type { KSpace, Pair } from '../../engine/force/types';
import { KSpaceMSM, KSpaceMSMCG } from '../msm/msm';
import { PairBornCoulMSM, PairBuckCoulMSM, PairCoulMSM, PairLJCharmmCoulMSM, PairLJCutCoulMSM, PairLJSpicaCoulMSM } from '../msm/pair_coul_msm';

/** kspace_style names. */
export const KSPACES: Record<string, () => KSpace> = {
  msm: () => new KSpaceMSM(),
  'msm/cg': () => new KSpaceMSMCG(),
};

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'coul/msm': () => new PairCoulMSM(),
  'lj/cut/coul/msm': () => new PairLJCutCoulMSM(),
  'born/coul/msm': () => new PairBornCoulMSM(),
  'buck/coul/msm': () => new PairBuckCoulMSM(),
  'lj/charmm/coul/msm': () => new PairLJCharmmCoulMSM(),
  'lj/spica/coul/msm': () => new PairLJSpicaCoulMSM('lj/spica/coul/msm'),
  'lj/sdk/coul/msm': () => new PairLJSpicaCoulMSM('lj/sdk/coul/msm'),
};
