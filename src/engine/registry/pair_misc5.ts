import type { Pair } from '../force/types';

/** Three-body pair styles of the Haiku wave 10 (nb3b/harmonic, atm); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {};
