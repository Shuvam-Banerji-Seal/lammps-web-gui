import type { Pair } from '../force/types';
import { PairGayBerne } from '../force/pair/gayberne';

/** aspherical (ellipsoid) pair styles (wave 19); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  gayberne: () => new PairGayBerne(),
};
