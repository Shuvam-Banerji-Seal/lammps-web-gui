import type { Pair } from '../force/types';
import { PairEIM } from '../force/pair/eim';

/** pair_style eim (Haiku wave 14); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  eim: () => new PairEIM(),
};
