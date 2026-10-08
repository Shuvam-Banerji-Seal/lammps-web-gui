import type { Pair } from '../force/types';
import { PairBuck, PairBuckCoulCut, PairBorn, PairMorse } from '../force/pair/simple';

/** Pair styles added by the simple wave; merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  buck: () => new PairBuck(),
  'buck/coul/cut': () => new PairBuckCoulCut(),
  born: () => new PairBorn(),
  morse: () => new PairMorse(),
};
