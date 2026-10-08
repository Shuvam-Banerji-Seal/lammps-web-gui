import type { FixFactory } from '../styles';
import { FixAveChunk } from '../fix/ave_chunk';

/** fix ave/chunk (Haiku wave); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'ave/chunk': (s, i, g, a) => new FixAveChunk(s, i, g, a),
};
