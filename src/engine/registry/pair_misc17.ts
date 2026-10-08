import type { Pair } from '../force/types';
import { PairHarmonicCut } from '../force/pair/harmonic_cut';

/** misc pair styles (wave 17, GLM worker); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'harmonic/cut': () => new PairHarmonicCut(),
};
