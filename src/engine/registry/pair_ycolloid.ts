import type { Pair } from '../force/types';
import { PairYukawaColloid } from '../force/pair/yukawa_colloid';

/** Pair styles added by wave 4 (pair_ycolloid); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'yukawa/colloid': () => new PairYukawaColloid(),
};
