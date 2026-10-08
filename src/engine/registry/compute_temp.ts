import type { ComputeFactory } from '../styles';
import { ComputeTempPartial, ComputeTempCom, ComputeTempRegion } from '../compute/temp_bias';

/** Compute styles added by wave 2 (compute_temp); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'temp/partial': (s, i, g, a) => new ComputeTempPartial(s, i, g, a),
  'temp/com': (s, i, g, a) => new ComputeTempCom(s, i, g, a),
  'temp/region': (s, i, g, a) => new ComputeTempRegion(s, i, g, a),
};
