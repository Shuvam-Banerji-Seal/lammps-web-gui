import type { FixFactory } from '../styles';
import { FixWallTable } from '../fix/wall_table';

/** fix wall/table (wave 16, GLM worker); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'wall/table': (sys, id, group, args) => new FixWallTable(sys, id, group, args),
};
