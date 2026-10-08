import type { Pair } from '../force/types';
import { PairZBL } from '../force/pair/zbl';

/** Pair styles added by wave 3 (pair_zbl); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  zbl: () => new PairZBL(),
};
