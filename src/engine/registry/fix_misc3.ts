import type { FixFactory } from '../styles';
import { FixVector, FixWallReflectStochastic, FixWallRegion } from '../fix/misc3';

/** fix vector, wall/region, wall/reflect/stochastic (Haiku wave); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  vector: (sys, id, group, args) => new FixVector(sys, id, group, args),
  'wall/region': (sys, id, group, args) => new FixWallRegion(sys, id, group, args),
  'wall/reflect/stochastic': (sys, id, group, args) => new FixWallReflectStochastic(sys, id, group, args),
};
