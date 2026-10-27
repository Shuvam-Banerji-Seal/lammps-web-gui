import { pairAdaptAccess, pairExtractDim, pairExtractHas, type AdaptPairsHook } from '../fix/adaptPairs';

/**
 * fix adapt's pair parameters (native extract() names and where the engine keeps each value), loaded when a command
 * names adapt (lazyStyles.ts), so the table stays out of the engine worker's first chunk.
 */
export const HOOKS: Record<string, AdaptPairsHook> = {
  adapt: { pairExtractHas, pairExtractDim, pairAdaptAccess },
};
