/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the GPL-ported pair styles the pair
 * threads of the notebook engine run (src/MANYBODY/pair_sw.cpp as src/engine-gpl/manybody3/pair_sw.ts). The pair worker
 * loads this module through src/engine/lazyStyles.ts loadGplThreadedPair, whose GPL_THREADED_PAIRS lists these names.
 */
import type { Pair } from '../../engine/force/types';
import { PairSW } from '../manybody3/pair_sw';

/** This module's URL: the pair worker imports it as it is (lazyStyles.ts loadGplThreadedPair). */
export const MODULE_URL: string = import.meta.url;

/** Factories of the worker's copies, by style name. */
export const THREADED: Record<string, () => Pair> = {
  sw: () => new PairSW(),
};
