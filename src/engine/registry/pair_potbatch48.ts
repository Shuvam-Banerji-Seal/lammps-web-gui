import { PairPedone } from '../force/pair/pedone';
import { PairSMATB, PairSMATBSingle } from '../force/pair/smatb';
import { PairEDIP, PairEDIPMulti } from '../force/pair/edip';

/** Pair styles of batch 48 (pedone, smatb, edip, edip/multi); merged through lazyStyles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => unknown> = {
  pedone: () => new PairPedone(),
  smatb: () => new PairSMATB(),
  'smatb/single': () => new PairSMATBSingle(),
  edip: () => new PairEDIP(),
  'edip/multi': () => new PairEDIPMulti(),
};
