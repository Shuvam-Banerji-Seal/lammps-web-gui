import type { FixFactory } from '../styles';
import { FixMolSwap } from '../fix/mol_swap';

/** fix mol/swap (wave 30); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'mol/swap': (s, i, g, a) => new FixMolSwap(s, i, g, a),
};
