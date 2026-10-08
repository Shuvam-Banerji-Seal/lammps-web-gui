import type { FixFactory } from '../styles';
import { FixDeform } from '../fix/deform';

/** Fix styles added by wave 3 (fix_deform); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  deform: (sys, id, group, args) => new FixDeform(sys, id, group, args),
};
