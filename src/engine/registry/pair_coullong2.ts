import type { Pair } from '../force/types';
import { PairBornCoulLong, PairBuckCoulLong } from '../force/pair/coul_long2';

/** Pair styles added by wave 3 (pair_coullong2); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'born/coul/long': () => new PairBornCoulLong(),
  'buck/coul/long': () => new PairBuckCoulLong(),
};
