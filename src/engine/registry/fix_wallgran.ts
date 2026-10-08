import type { FixFactory } from '../styles';
import { FixWallGran, FixWallGranGranular } from '../fix/wall_gran';
import { FixFreeze } from '../fix/freeze';
import { StyleError } from '../force/types';

/** fix wall/gran, wall/gran/region, freeze (Haiku wave); merged into styles.ts. Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  // fstyle granular: the contact models of pair_style granular against a flat wall (fix_wall_gran.rst)
  'wall/gran': (sys, id, group, args) => (args[0] === 'granular'
    ? new FixWallGranGranular(sys, id, group, args, false)
    : new FixWallGran(sys, id, group, args)),
  // docs.lammps.org/fix_wall_gran_region.html: fstyle granular with a block, sphere or cylinder region
  'wall/gran/region': (sys, id, group, args) => {
    if (args[0] !== 'granular') throw new StyleError('fix wall/gran/region: only fstyle granular is supported yet');
    return new FixWallGranGranular(sys, id, group, args, true);
  },
  freeze: (sys, id, group, args) => new FixFreeze(sys, id, group, args),
};
