import type { FixFactory } from '../styles';
import { FixTTM } from '../fix/ttm';

/** fix ttm, ttm/grid (Haiku wave 14; ttm/mod is not registered, see fix/ttm_mod.ts); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  ttm: (sys, id, group, args) => new FixTTM(sys, id, group, args, 'ttm'),
  'ttm/grid': (sys, id, group, args) => new FixTTM(sys, id, group, args, 'ttm/grid'),
};
