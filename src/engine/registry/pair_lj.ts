import type { Pair } from '../force/types';
import { PairLJ96Cut, PairLJExpand, PairLJSmooth, PairLJSmoothLinear } from '../force/pair/lj_variants';

/** Pair styles added by the lj wave; merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'lj96/cut': () => new PairLJ96Cut(),
  'lj/expand': () => new PairLJExpand(),
  'lj/smooth': () => new PairLJSmooth(),
  'lj/smooth/linear': () => new PairLJSmoothLinear(),
};
