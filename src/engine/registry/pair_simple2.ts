import type { Pair } from '../force/types';
import { PairSoft, PairYukawa, PairGauss, PairZero } from '../force/pair/simple2';

/** Pair styles added by the wave-1 rerun (pair_simple2); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  soft: () => new PairSoft(),
  yukawa: () => new PairYukawa(),
  gauss: () => new PairGauss(),
  zero: () => new PairZero(),
};
