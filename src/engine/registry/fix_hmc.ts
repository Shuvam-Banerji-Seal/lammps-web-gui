import type { FixFactory } from '../styles';
import { FixHmc } from '../fix/hmc';

/** fix hmc (wave 32); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  hmc: (sys, id, group, args) => new FixHmc(sys, id, group, args),
};
