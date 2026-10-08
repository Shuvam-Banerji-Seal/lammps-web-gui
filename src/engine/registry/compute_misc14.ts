import type { ComputeFactory } from '../styles';
import { ComputeTempRamp } from '../compute/temp_ramp';
import { ComputeTempChunk } from '../compute/temp_chunk';
import { ComputeTempCs } from '../compute/temp_cs';

/** Computes of the Haiku wave 14; merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'temp/ramp': (s, i, g, a) => new ComputeTempRamp(s, i, g, a),
  'temp/chunk': (s, i, g, a) => new ComputeTempChunk(s, i, g, a),
  'temp/cs': (s, i, g, a) => new ComputeTempCs(s, i, g, a),
};
