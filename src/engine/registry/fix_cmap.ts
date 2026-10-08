import type { FixFactory } from '../styles';
import { FixCmap } from '../fix/cmap';

/** fix cmap (Haiku wave 15); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  cmap: (sys, id, group, args) => new FixCmap(sys, id, group, args),
};
