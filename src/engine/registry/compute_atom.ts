import type { ComputeFactory } from '../styles';
import { ComputeKEAtom, ComputePEAtom, ComputeStressAtom, ComputePropertyAtom } from '../compute/peratom';

/** Compute styles added by wave 2 (compute_atom); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'ke/atom': (sys, id, group, args) => new ComputeKEAtom(sys, id, group, args),
  'pe/atom': (sys, id, group, args) => new ComputePEAtom(sys, id, group, args),
  'stress/atom': (sys, id, group, args) => new ComputeStressAtom(sys, id, group, args),
  'property/atom': (sys, id, group, args) => new ComputePropertyAtom(sys, id, group, args),
};
