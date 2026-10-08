import type { Pair } from '../force/types';
import { PairLJCutTIP4PCut, PairLJCutTIP4PLong, PairTIP4PCut, PairTIP4PLong } from '../force/pair/tip4p';

/** TIP4P pair styles (Haiku wave 12); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'lj/cut/tip4p/cut': () => new PairLJCutTIP4PCut(),
  'lj/cut/tip4p/long': () => new PairLJCutTIP4PLong(),
  'tip4p/cut': () => new PairTIP4PCut(),
  'tip4p/long': () => new PairTIP4PLong(),
};
