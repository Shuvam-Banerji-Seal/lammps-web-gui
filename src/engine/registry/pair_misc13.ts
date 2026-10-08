import type { Pair } from '../force/types';
import { PairVashishtaTable } from '../force/pair/vashishta_table';

/** Pair styles of the Haiku wave 13; merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'vashishta/table': () => new PairVashishtaTable(),
};
