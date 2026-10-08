import type { Pair } from '../force/types';

/** pair_style lj/relres (Haiku wave); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {};
