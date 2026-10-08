import type { Pair } from '../force/types';
import { PairGranular } from '../force/pair/granular';

/** pair_style granular (Haiku wave); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  granular: () => new PairGranular(),
};
