import type { ComputeFactory } from '../styles';
import { ComputePair } from '../compute/pair';

/** compute pair (Haiku wave 15); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  pair: (sys, id, group, args) => new ComputePair(sys, id, group, args),
};
