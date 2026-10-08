import type { ComputeFactory } from '../styles';
import { ComputeCom, ComputeGyration, ComputeMSD, ComputeRDF, ComputeVACF } from '../compute/global';

/** Compute styles added by wave 2 (compute_global); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  com: (sys, id, group, args) => new ComputeCom(sys, id, group, args),
  gyration: (sys, id, group, args) => new ComputeGyration(sys, id, group, args),
  msd: (sys, id, group, args) => new ComputeMSD(sys, id, group, args),
  vacf: (sys, id, group, args) => new ComputeVACF(sys, id, group, args),
  rdf: (sys, id, group, args) => new ComputeRDF(sys, id, group, args),
};
