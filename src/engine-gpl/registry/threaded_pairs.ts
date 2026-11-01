/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the GPL-ported pair styles the pair
 * threads of the notebook engine run (src/MANYBODY/pair_sw.cpp as src/engine-gpl/manybody3/pair_sw.ts; the CORESHELL
 * styles of src/CORESHELL and buck/long/coul/long of src/KSPACE/pair_buck_long_coul_long.cpp, as ported in
 * src/engine-gpl/csdiel; pace of src/ML-PACE/pair_pace.cpp, as ported in src/engine-gpl/pace; reaxff of src/REAXFF, as
 * ported in src/engine-gpl/reaxff, whose bond-order loop the threads share, Pair.threadKernel). The pair worker
 * loads this module through src/engine/lazyStyles.ts loadGplThreadedPair, whose GPL_THREADED_PAIRS lists these names.
 */
import type { Pair } from '../../engine/force/types';
import { PairSW } from '../manybody3/pair_sw';
import { PairEffCut } from '../eff/pair_eff_cut';
import { PairBornCoulLongCS, PairBuckCoulLongCS } from '../csdiel/born_buck_long_cs';
import { PairBornCoulDsfCS, PairBornCoulWolfCS } from '../csdiel/born_dsf_wolf_cs';
import { PairBuckLongCoulLong } from '../csdiel/buck_long';
import { PairCoulLongCS, PairCoulWolfCS, PairLJClass2CoulLongCS, PairLJCutCoulLongCS } from '../csdiel/coreshell';
import { PairPace } from '../pace/pair_pace';
import { PairReaxFF } from '../reaxff/pair';

/** This module's URL: the pair worker imports it as it is (lazyStyles.ts loadGplThreadedPair). */
export const MODULE_URL: string = import.meta.url;

/** Factories of the worker's copies, by style name. */
export const THREADED: Record<string, () => Pair> = {
  sw: () => new PairSW(),
  'eff/cut': () => new PairEffCut(),
  'born/coul/long/cs': () => new PairBornCoulLongCS(),
  'buck/coul/long/cs': () => new PairBuckCoulLongCS(),
  'born/coul/dsf/cs': () => new PairBornCoulDsfCS(),
  'born/coul/wolf/cs': () => new PairBornCoulWolfCS(),
  'coul/wolf/cs': () => new PairCoulWolfCS(),
  'coul/long/cs': () => new PairCoulLongCS(),
  'lj/cut/coul/long/cs': () => new PairLJCutCoulLongCS(),
  'lj/class2/coul/long/cs': () => new PairLJClass2CoulLongCS(),
  'buck/long/coul/long': () => new PairBuckLongCoulLong(),
  pace: () => new PairPace(),
  reaxff: () => new PairReaxFF(),
};
