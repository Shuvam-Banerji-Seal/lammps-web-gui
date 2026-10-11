import type { ComputeFactory } from '../styles';
import { ComputeGroupGroup, ComputePairTally } from '../compute/tally';

/** Pairwise group computes (compute_group_group, compute_tally); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'group/group': (sys, id, group, args) => new ComputeGroupGroup(sys, id, group, args),
  'pe/tally': (sys, id, group, args) => new ComputePairTally(sys, id, group, args, 'pe/tally'),
  'force/tally': (sys, id, group, args) => new ComputePairTally(sys, id, group, args, 'force/tally'),
  'stress/tally': (sys, id, group, args) => new ComputePairTally(sys, id, group, args, 'stress/tally'),
};
