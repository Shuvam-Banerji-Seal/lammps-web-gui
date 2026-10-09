import type { FixFactory } from '../styles';
import { FixGcmc } from '../fix/gcmc';

/** fix gcmc (wave 32); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  gcmc: (sys, id, group, args) => new FixGcmc(sys, id, group, args),
};
