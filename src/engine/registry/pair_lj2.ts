import type { Pair } from '../force/types';
import { PairLJClass2, PairLJCubic, PairLJGromacs, PairMieCut } from '../force/pair/lj_variants2';

/** Pair styles added by the wave-1 rerun (pair_lj2); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'lj/gromacs': () => new PairLJGromacs(),
  'lj/class2': () => new PairLJClass2(),
  'lj/cubic': () => new PairLJCubic(),
  'mie/cut': () => new PairMieCut(),
};
