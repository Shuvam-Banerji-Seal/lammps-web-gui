import type { FixFactory } from '../styles';
import { FixGJF } from '../fix/gjf';

/** fix gjf (Haiku wave 9); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  gjf: (sys, id, group, args) => new FixGJF(sys, id, group, args),
};
