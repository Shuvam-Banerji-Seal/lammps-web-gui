import type { FixFactory } from '../styles';
import { FixWallTable } from '../fix/wall_table';
import { FixWallFlow } from '../fix/wall_flow';

/** fix wall/table (wave 16, GLM worker) and fix wall/flow (wall/flow wave); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'wall/table': (sys, id, group, args) => new FixWallTable(sys, id, group, args),
  'wall/flow': (sys, id, group, args) => new FixWallFlow(sys, id, group, args),
};
