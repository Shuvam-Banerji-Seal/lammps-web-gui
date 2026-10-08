import type { ComputeFactory } from '../styles';
import { ComputeCountType, ComputeERotateSphereAtom, ComputeNBondAtom } from '../compute/misc18';

/** misc computes (wave 18, space-bunny swarm); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'nbond/atom': (sys, id, group, args) => new ComputeNBondAtom(sys, id, group, args),
  'count/type': (sys, id, group, args) => new ComputeCountType(sys, id, group, args),
  'erotate/sphere/atom': (sys, id, group, args) => new ComputeERotateSphereAtom(sys, id, group, args),
};