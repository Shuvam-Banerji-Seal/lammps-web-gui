import type { ComputeFactory } from '../styles';
import { ComputeCentroAtom, ComputeCnaAtom, ComputeClusterAtom, ComputeFragmentAtom, ComputeAggregateAtom, ComputeHexorderAtom } from '../compute/struct';

/** Computes of the Haiku wave 11 (compute_struct); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'centro/atom': (sys, id, group, args) => new ComputeCentroAtom(sys, id, group, args),
  'cna/atom': (sys, id, group, args) => new ComputeCnaAtom(sys, id, group, args),
  'cluster/atom': (sys, id, group, args) => new ComputeClusterAtom(sys, id, group, args),
  'fragment/atom': (sys, id, group, args) => new ComputeFragmentAtom(sys, id, group, args),
  'aggregate/atom': (sys, id, group, args) => new ComputeAggregateAtom(sys, id, group, args),
  'hexorder/atom': (sys, id, group, args) => new ComputeHexorderAtom(sys, id, group, args),
};
