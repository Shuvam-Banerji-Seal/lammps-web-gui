import type { FixFactory } from '../styles';
import { FixPour } from '../fix/pour';
import { FixDeposit } from '../fix/deposit';

/** fix pour, deposit (Haiku wave); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  pour: (sys, id, group, args) => new FixPour(sys, id, group, args),
  deposit: (sys, id, group, args) => new FixDeposit(sys, id, group, args),
};
