import type { FixFactory } from '../styles';
import { FixWallGran } from '../fix/wall_gran';
import { FixFreeze } from '../fix/freeze';
import { StyleError } from '../force/types';

/** fix wall/gran, wall/gran/region, freeze (Haiku wave); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'wall/gran': (sys, id, group, args) => new FixWallGran(sys, id, group, args),
  // docs.lammps.org/fix_wall_gran_region.html: not implemented yet
  'wall/gran/region': () => { throw new StyleError('fix wall/gran/region is not supported yet'); },
  freeze: (sys, id, group, args) => new FixFreeze(sys, id, group, args),
};
