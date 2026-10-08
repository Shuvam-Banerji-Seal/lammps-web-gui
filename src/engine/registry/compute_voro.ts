import type { ComputeFactory } from '../styles';
import { ComputeVoronoiAtom } from '../compute/voronoi';

/** compute voronoi/atom (Haiku wave); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'voronoi/atom': (sys, id, group, args) => new ComputeVoronoiAtom(sys, id, group, args),
};
