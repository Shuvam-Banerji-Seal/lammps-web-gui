import type { ComputeFactory } from '../styles';
import { ComputeTempDeform } from '../compute/temp_deform';

/** Compute styles added by wave 3 (compute_deform); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'temp/deform': (sys, id, group, args) => new ComputeTempDeform(sys, id, group, args),
};
