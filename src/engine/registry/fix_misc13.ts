import type { FixFactory } from '../styles';
import { FixController } from '../fix/controller';
import { FixAveGrid } from '../fix/ave_grid';

/** Fixes of the Haiku wave 13; merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  controller: (sys, id, group, args) => new FixController(sys, id, group, args),
  'ave/grid': (sys, id, group, args) => new FixAveGrid(sys, id, group, args),
};
