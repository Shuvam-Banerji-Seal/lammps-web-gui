import type { Pair } from '../force/types';

/** TIP4P pair styles (Haiku wave 12); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {};
