import type { FixFactory } from '../styles';
import { FixIndent, FixWallReflect } from '../fix/indent';

/** Fix styles added by wave 2 (fix_wref); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  indent: (sys, id, group, args) => new FixIndent(sys, id, group, args),
  'wall/reflect': (sys, id, group, args) => new FixWallReflect(sys, id, group, args),
};
