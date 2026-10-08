import type { Pair } from '../force/types';
import { PairEAM, PairEAMAlloy, PairEAMFS } from '../force/pair/eam';

/** Many-body pair styles added by wave 2 (pair_eam); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  eam: () => new PairEAM(),
  'eam/alloy': () => new PairEAMAlloy(),
  'eam/fs': () => new PairEAMFS(),
};
