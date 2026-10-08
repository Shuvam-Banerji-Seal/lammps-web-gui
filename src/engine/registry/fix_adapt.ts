import type { FixFactory } from '../styles';
import { FixAdapt } from '../fix/adapt';

/** fix adapt (Haiku wave 9); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  adapt: (sys, id, group, args) => new FixAdapt(sys, id, group, args),
};
