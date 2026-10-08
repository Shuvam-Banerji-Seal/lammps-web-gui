import type { FixFactory } from '../styles';
import { FixAccelerateCos } from '../fix/accelerate_cos';

/** Fixes of the Haiku wave 14; merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'accelerate/cos': (sys, id, group, args) => new FixAccelerateCos(sys, id, group, args),
};
