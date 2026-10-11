import type { Bonded, Pair } from '../force/types';
import { BondBpmSpring } from '../force/bond/bpm';
import { PairBpmSpring } from '../force/pair/bpm';

/** BPM package styles (wave 51); loaded lazily (lazyStyles.ts). Style name -> factory. */
export const BONDS: Record<string, () => Bonded> = {
  'bpm/spring': () => new BondBpmSpring(),
};
export const PAIRS: Record<string, () => Pair> = {
  'bpm/spring': () => new PairBpmSpring(),
};
