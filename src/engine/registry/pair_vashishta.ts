import type { Pair } from '../force/types';
import { PairVashishta } from '../force/pair/vashishta';

/** Many-body pair styles added by wave 4 (pair_vashishta); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  vashishta: () => new PairVashishta(),
};
