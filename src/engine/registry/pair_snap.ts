import type { Pair } from '../force/types';
import { PairSnap } from '../force/pair/snap';

/** Many-body pair styles added by the SNAP wave (pair_snap); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  snap: () => new PairSnap(),
};
