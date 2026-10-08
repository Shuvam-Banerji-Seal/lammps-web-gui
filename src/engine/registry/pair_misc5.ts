import type { Pair } from '../force/types';
import { PairATM } from '../force/pair/atm';
import { PairNB3BHarmonic, PairNB3BScreened } from '../force/pair/nb3b';

/** Three-body pair styles of the Haiku wave 10 (nb3b/harmonic, atm); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  atm: () => new PairATM(),
  'nb3b/harmonic': () => new PairNB3BHarmonic(),
  'nb3b/screened': () => new PairNB3BScreened(),
};
