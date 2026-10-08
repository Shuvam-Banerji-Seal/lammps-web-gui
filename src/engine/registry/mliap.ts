import type { ComputeFactory } from '../styles';
import type { Pair } from '../force/types';

/** pair_style mliap and compute mliap (wave 16); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {};
export const COMPUTES: Record<string, ComputeFactory> = {};
