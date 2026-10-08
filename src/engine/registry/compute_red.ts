import type { ComputeFactory } from '../styles';
import { ComputeReduce, ComputeReduceRegion, ComputeDisplaceAtom, ComputeCoordAtom } from '../compute/reduce';

/** Compute styles added by wave 2 (compute_red); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  reduce: (sys, id, group, args) => new ComputeReduce(sys, id, group, args),
  'reduce/region': (sys, id, group, args) => new ComputeReduceRegion(sys, id, group, args),
  'displace/atom': (sys, id, group, args) => new ComputeDisplaceAtom(sys, id, group, args),
  'coord/atom': (sys, id, group, args) => new ComputeCoordAtom(sys, id, group, args),
};
