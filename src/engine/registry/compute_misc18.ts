import type { ComputeFactory } from '../styles';
import { ComputeCountType, ComputeERotateSphereAtom, ComputeNBondAtom } from '../compute/misc18';
import { ComputeADF } from '../compute/adf';
import { ComputePropertyGrid } from '../compute/property_grid';
import { ComputeBornMatrix } from '../compute/born_matrix';

/** misc computes (wave 18, space-bunny swarm); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'nbond/atom': (sys, id, group, args) => new ComputeNBondAtom(sys, id, group, args),
  'count/type': (sys, id, group, args) => new ComputeCountType(sys, id, group, args),
  'erotate/sphere/atom': (sys, id, group, args) => new ComputeERotateSphereAtom(sys, id, group, args),
  adf: (sys, id, group, args) => new ComputeADF(sys, id, group, args),
  'property/grid': (sys, id, group, args) => new ComputePropertyGrid(sys, id, group, args),
  'born/matrix': (sys, id, group, args) => new ComputeBornMatrix(sys, id, group, args),
};