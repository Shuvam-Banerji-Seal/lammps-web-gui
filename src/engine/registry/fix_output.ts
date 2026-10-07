import type { FixFactory } from '../styles';
import { FixAveTime, FixPrint } from '../fix/ave_time';

/** Fix styles added by wave 2 (fix_output); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'ave/time': (sys, id, group, args) => new FixAveTime(sys, id, group, args),
  print: (sys, id, group, args) => new FixPrint(sys, id, group, args),
};
