import type { ComputeFactory } from '../styles';
import { ComputeHeatFlux } from '../compute/heat_flux';

/** Computes of the Haiku wave 11 (compute_gk); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'heat/flux': (sys, id, group, args) => new ComputeHeatFlux(sys, id, group, args),
};
