import type { FixFactory } from '../styles';
import { FixNumdiff, FixNumdiffVirial } from '../fix/numdiff';

/** fix numdiff, fix numdiff/virial (wave 16); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  numdiff: (sys, id, group, args) => new FixNumdiff(sys, id, group, args),
  'numdiff/virial': (sys, id, group, args) => new FixNumdiffVirial(sys, id, group, args),
};
