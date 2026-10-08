import type { FixFactory } from '../styles';
import { FixMomentum, FixRecenter, FixNVELimit, FixNVENoforce } from '../fix/motion';

/** Fix styles added by wave 2 (fix_mom); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  momentum: (sys, id, group, args) => new FixMomentum(sys, id, group, args),
  recenter: (sys, id, group, args) => new FixRecenter(sys, id, group, args),
  'nve/limit': (sys, id, group, args) => new FixNVELimit(sys, id, group, args),
  'nve/noforce': (sys, id, group, args) => new FixNVENoforce(sys, id, group, args),
};
