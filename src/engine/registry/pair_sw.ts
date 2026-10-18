import type { Pair } from '../force/types';
import { PairSWMod } from '../force/pair/sw';

/** Many-body pair styles added by wave 2 (pair_sw); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'sw/mod': () => new PairSWMod(),
};
