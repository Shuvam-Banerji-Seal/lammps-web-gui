import type { Pair } from '../force/types';
import { PairLJRelRes } from '../force/pair/lj_relres';

/** pair_style lj/relres (Haiku wave); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'lj/relres': () => new PairLJRelRes(),
};
