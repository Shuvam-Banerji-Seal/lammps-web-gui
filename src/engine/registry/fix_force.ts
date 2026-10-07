import { FixAddForce, FixAveForce, FixSetForce } from '../fix/force_mods';
import type { FixFactory } from '../styles';

/** Fix styles added by wave 2 (fix_force); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  setforce: (sys, id, group, args) => new FixSetForce(sys, id, group, args),
  addforce: (sys, id, group, args) => new FixAddForce(sys, id, group, args),
  aveforce: (sys, id, group, args) => new FixAveForce(sys, id, group, args),
};
