import type { FixFactory } from '../styles';
import { FixViscous, FixGravity, FixLineforce, FixPlaneforce, FixEfield } from '../fix/force_ext';

/** Fix styles added by wave 2 (fix_ext); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  viscous: (sys, id, group, args) => new FixViscous(sys, id, group, args),
  gravity: (sys, id, group, args) => new FixGravity(sys, id, group, args),
  lineforce: (sys, id, group, args) => new FixLineforce(sys, id, group, args),
  planeforce: (sys, id, group, args) => new FixPlaneforce(sys, id, group, args),
  efield: (sys, id, group, args) => new FixEfield(sys, id, group, args),
};
