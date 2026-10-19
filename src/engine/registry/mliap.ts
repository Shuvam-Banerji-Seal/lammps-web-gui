import type { ComputeFactory } from '../styles';
import type { Pair } from '../force/types';
import { PairMliap } from '../force/pair/mliap';
import { ComputeMliap } from '../compute/mliap';
import { gplHook } from '../gplHooks';
import type { MliapAceHook } from '../mliap/ace';

/** pair_style mliap and compute mliap (wave 16); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  mliap: () => new PairMliap(),
};
export const COMPUTES: Record<string, ComputeFactory> = {
  mliap: (sys, id, group, args) => {
    // descriptor ace: the ACE projections are ported from ML-PACE, so this compute is the lazily loaded HOOKS entry
    // 'ace' (src/engine/mliap/ace.ts)
    const k = args.indexOf('descriptor');
    if (k >= 0 && args[k + 1] === 'ace') return gplHook<MliapAceHook>('ace').compute(sys, id, group, args);
    return new ComputeMliap(sys, id, group, args);
  },
};
