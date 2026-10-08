import type { Pair } from '../force/types';
import { PairTersoff } from '../force/pair/tersoff';

/** Many-body pair styles added by wave 2 (pair_3body); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  tersoff: () => new PairTersoff(),
};
