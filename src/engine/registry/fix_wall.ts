import type { FixFactory } from '../styles';
import { FixWallLJ93, FixWallLJ126, FixWallLJ1043, FixWallHarmonic, FixWallMorse } from '../fix/walls';

/** Fix styles added by wave 2 (fix_wall); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'wall/lj93': (sys, id, group, args) => new FixWallLJ93(sys, id, group, args),
  'wall/lj126': (sys, id, group, args) => new FixWallLJ126(sys, id, group, args),
  'wall/lj1043': (sys, id, group, args) => new FixWallLJ1043(sys, id, group, args),
  'wall/harmonic': (sys, id, group, args) => new FixWallHarmonic(sys, id, group, args),
  'wall/morse': (sys, id, group, args) => new FixWallMorse(sys, id, group, args),
};
