import { pairAdaptAccess, pairExtractDim, pairExtractHas, type AdaptPairsHook } from '../fix/adaptPairs';
import { bondedAdaptAccess, bondedExtractHas } from '../fix/adaptBonded';

/**
 * fix adapt's pair and bonded parameters (native extract() names and where the engine keeps each value), loaded when a
 * command names adapt (lazyStyles.ts), so the tables stay out of the engine worker's first chunk.
 */
export const HOOKS: Record<string, AdaptPairsHook> = {
  adapt: { pairExtractHas, pairExtractDim, pairAdaptAccess, bondedExtractHas, bondedAdaptAccess },
};
