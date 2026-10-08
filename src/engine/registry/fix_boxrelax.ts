import type { FixFactory } from '../styles';
import { FixBoxRelax } from '../fix/box_relax';

/** Fix styles added by wave 10 (fix box/relax); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'box/relax': (sys, id, group, args) => new FixBoxRelax(sys, id, group, args),
};
