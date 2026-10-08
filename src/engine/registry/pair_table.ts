import type { Pair } from '../force/types';
import { PairTable } from '../force/pair/table';

/** Pair styles added by wave 3 (pair_table); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  table: () => new PairTable(),
};
