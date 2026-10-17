import type { Pair } from '../force/types';
import { PairResquared } from '../force/pair/resquared';

/** aspherical (ellipsoid) pair styles (wave 19); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  resquared: () => new PairResquared(),
};
