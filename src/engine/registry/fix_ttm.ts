import type { FixFactory } from '../styles';
import { FixTTM } from '../fix/ttm';
import { FixTTMThermal } from '../fix/ttm_thermal';
import { FixTTMCascade } from '../fix/ttm_cascade';
import { FixTTMMod } from '../fix/ttm_mod';

/** fix ttm, ttm/grid, ttm/mod (fix/ttm_mod.ts), ttm/thermal and ttm/cascade; merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  ttm: (sys, id, group, args) => new FixTTM(sys, id, group, args, 'ttm'),
  'ttm/grid': (sys, id, group, args) => new FixTTM(sys, id, group, args, 'ttm/grid'),
  'ttm/mod': (sys, id, group, args) => new FixTTMMod(sys, id, group, args),
  'ttm/thermal': (sys, id, group, args) => new FixTTMThermal(sys, id, group, args),
  'ttm/cascade': (sys, id, group, args) => new FixTTMCascade(sys, id, group, args),
};
