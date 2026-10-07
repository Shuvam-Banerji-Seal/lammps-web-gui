import type { FixFactory } from '../styles';
import { FixSpring, FixSpringSelf } from '../fix/springs';

/** Fix styles added by wave 2 (fix_motion); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  spring: (sys, id, group, args) => new FixSpring(sys, id, group, args),
  'spring/self': (sys, id, group, args) => new FixSpringSelf(sys, id, group, args),
};
