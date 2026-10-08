import type { FixFactory } from '../styles';
import { FixHeat, FixEhex } from '../fix/ehex';

/** fix heat (HEX) and fix ehex (eHEX); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  heat: (sys, id, group, args) => new FixHeat(sys, id, group, args),
  ehex: (sys, id, group, args) => new FixEhex(sys, id, group, args),
};
