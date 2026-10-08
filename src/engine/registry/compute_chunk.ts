import type { ComputeFactory } from '../styles';
import { ComputeChunkAtom, ComputeComChunk, ComputeMsdChunk } from '../compute/chunk';

/** compute chunk/atom and per-chunk computes (Haiku wave); merged into styles.ts. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'chunk/atom': (s, i, g, a) => new ComputeChunkAtom(s, i, g, a),
  'com/chunk': (s, i, g, a) => new ComputeComChunk(s, i, g, a),
  'msd/chunk': (s, i, g, a) => new ComputeMsdChunk(s, i, g, a),
};
