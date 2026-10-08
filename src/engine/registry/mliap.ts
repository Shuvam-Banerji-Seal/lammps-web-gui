import type { ComputeFactory } from '../styles';
import type { Pair } from '../force/types';
import { PairMliap } from '../force/pair/mliap';
import { ComputeMliap } from '../compute/mliap';

/** pair_style mliap and compute mliap (wave 16); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  mliap: () => new PairMliap(),
};
export const COMPUTES: Record<string, ComputeFactory> = {
  mliap: (sys, id, group, args) => new ComputeMliap(sys, id, group, args),
};
