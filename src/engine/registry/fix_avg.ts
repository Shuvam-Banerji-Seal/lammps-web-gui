import type { FixFactory } from '../styles';
import { FixAveAtom, FixAveHisto } from '../fix/ave_atom';

/** Fix styles added by wave 2 (fix_avg); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'ave/atom': (sys, id, group, args) => new FixAveAtom(sys, id, group, args),
  'ave/histo': (sys, id, group, args) => new FixAveHisto(sys, id, group, args),
};
