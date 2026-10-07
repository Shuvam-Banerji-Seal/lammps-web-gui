import type { FixSpec } from './types';
import { FixEnforce2d, FixNve, type Fix } from './integrate';
import { FixLangevin, FixNvt, FixTempBerendsen, FixTempRescale } from './thermostats';

/*
 * Fix construction from parsed specs. The interpreter parses every
 * supported fix's arguments (docs.lammps.org/fix_<style>.html) into a
 * FixSpec; this module turns a spec into a Fix object.
 */

export class FixNotImplementedError extends Error {}

export const makeFix = (spec: FixSpec): Fix => {
  switch (spec.style) {
    case 'nve':
      return new FixNve(spec.id);
    case 'enforce2d':
      return new FixEnforce2d(spec.id);
    case 'langevin':
      return new FixLangevin(spec.id, spec.tStart, spec.tStop, spec.damp, spec.seed);
    case 'temp/berendsen':
      return new FixTempBerendsen(spec.id, spec.tStart, spec.tStop, spec.damp);
    case 'temp/rescale':
      return new FixTempRescale(spec.id, spec.every, spec.tStart, spec.tStop, spec.window, spec.fraction);
    case 'nvt':
      return new FixNvt(spec.id, spec.tStart, spec.tStop, spec.damp);
    default: {
      // All FixSpec styles are handled above; kept for future spec variants.
      const style = (spec as { style: string }).style;
      throw new FixNotImplementedError(`fix ${style} is not available in this build of the notebook engine yet`);
    }
  }
};
