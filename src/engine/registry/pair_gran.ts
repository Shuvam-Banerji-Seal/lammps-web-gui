import type { Pair } from '../force/types';
import { PairGran } from '../force/pair/gran';

/** Granular pair styles (atom_style sphere); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'gran/hooke': () => new PairGran('gran/hooke'),
  'gran/hooke/history': () => new PairGran('gran/hooke/history'),
  'gran/hertz/history': () => new PairGran('gran/hertz/history'),
};
