import { PairPedone } from '../force/pair/pedone';
import { PairSMATB, PairSMATBSingle } from '../force/pair/smatb';

/** Pair styles of batch 48 (pedone, smatb, smatb/single); merged through lazyStyles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => unknown> = {
  pedone: () => new PairPedone(),
  smatb: () => new PairSMATB(),
  'smatb/single': () => new PairSMATBSingle(),
};
