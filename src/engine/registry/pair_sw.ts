import type { Pair } from '../force/types';
import { PairSW, PairSWMod } from '../force/pair/sw';

/** Many-body pair styles added by wave 2 (pair_sw); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  sw: () => new PairSW(),
  'sw/mod': () => new PairSWMod(),
};
