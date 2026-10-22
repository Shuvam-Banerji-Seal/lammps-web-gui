import type { FixFactory } from '../styles';
import { FixWallGran, FixWallGranGranular } from '../fix/wall_gran';
import { FixFreeze } from '../fix/freeze';
import { FixHeatFlow } from '../fix/heat_flow';
import { StyleError } from '../force/types';
import { missingArgs } from '../force/util';
import type { System } from '../system';

/** fix wall/gran, wall/gran/region, freeze (Haiku wave), heat/flow; merged into styles.ts. Style name -> factory. */
/**
 * As native LAMMPS does (src/GRANULAR/fix_wall_gran.cpp at b332f28, the base of both styles): at least one word, then
 * the atom attributes omega and radius, before the contact model is read.
 */
const wallGranHead = (sys: System, args: string[]): void => {
  if (args.length < 1) throw missingArgs('fix wall/gran');
  if (!sys.state.omega) throw new StyleError('Fix wall/gran requires atom attribute omega');
  if (!sys.state.radius) throw new StyleError('Fix wall/gran requires atom attribute radius');
};

export const FIXES: Record<string, FixFactory> = {
  // fstyle granular: the contact models of pair_style granular against a flat wall (fix_wall_gran.rst)
  'wall/gran': (sys, id, group, args) => {
    wallGranHead(sys, args);
    return args[0] === 'granular' ? new FixWallGranGranular(sys, id, group, args, false) : new FixWallGran(sys, id, group, args);
  },
  // docs.lammps.org/fix_wall_gran_region.html: fstyle granular with a block, sphere or cylinder region
  'wall/gran/region': (sys, id, group, args) => {
    wallGranHead(sys, args);
    if (!['granular', 'hooke', 'hooke/history', 'hertz/history'].includes(args[0])) {
      throw new StyleError(`fix ${id} wall/gran/region: unknown fstyle '${args[0] ?? ''}' (granular, hooke, hooke/history, hertz/history)`);
    }
    return new FixWallGranGranular(sys, id, group, args, true);
  },
  freeze: (sys, id, group, args) => new FixFreeze(sys, id, group, args),
  // docs.lammps.org/fix_heat_flow.html: integrates the temperature of atoms from their heat flow (GRANULAR heat conduction)
  'heat/flow': (sys, id, group, args) => new FixHeatFlow(sys, id, group, args),
};
