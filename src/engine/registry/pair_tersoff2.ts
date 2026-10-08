import type { Pair } from '../force/types';
import { PairTersoffMod, PairTersoffModC, PairTersoffZBL } from '../force/pair/tersoff_variants';

/** pair_style tersoff/mod, tersoff/mod/c, tersoff/zbl (Haiku wave 14); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'tersoff/mod': () => new PairTersoffMod(),
  'tersoff/mod/c': () => new PairTersoffModC(),
  'tersoff/zbl': () => new PairTersoffZBL(),
};
