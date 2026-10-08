import type { Pair } from '../force/types';

/** pair_style tersoff/mod, tersoff/mod/c, tersoff/zbl (Haiku wave 14); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {};
