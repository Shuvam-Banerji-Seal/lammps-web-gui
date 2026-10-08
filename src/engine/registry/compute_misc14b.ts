import type { ComputeFactory } from '../styles';
import { ComputeViscosityCos } from '../compute/viscosity_cos';

/** compute viscosity/cos (Haiku wave 14, with fix accelerate/cos); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'viscosity/cos': (sys, id, group, args) => new ComputeViscosityCos(sys, id, group, args),
};
