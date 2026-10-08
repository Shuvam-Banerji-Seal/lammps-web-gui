import type { Pair } from '../force/types';
import { PairColloid } from '../force/pair/colloid';

/** Pair styles added by wave 4 (pair_colloid); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  colloid: () => new PairColloid(),
};
