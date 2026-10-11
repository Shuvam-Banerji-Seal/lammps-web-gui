import type { FixFactory } from '../styles';
import { FixIlves } from '../fix/ilves';

/** fix ilves (ILVES bond and angle constraints); merged into styles.ts when loaded. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  ilves: (sys, id, group, args) => new FixIlves(sys, id, group, args),
};
