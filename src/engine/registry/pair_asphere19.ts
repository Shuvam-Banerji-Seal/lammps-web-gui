import type { Pair } from '../force/types';

/** aspherical (ellipsoid) pair styles (wave 19); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {};
