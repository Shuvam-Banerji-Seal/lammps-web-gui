import type { FixFactory } from '../styles';
import { StyleError } from '../force/types';

/*
 * fix rigid/nvt and rigid/nvt/small: not supported by the browser engine. The Nose-Hoover chain on the
 * body momenta (docs.lammps.org/fix_rigid.html: keywords temp and tparam) was tried and measured against
 * native LAMMPS (black box): the temperatures after 10 and 60 steps differed from every variant tried, so
 * the styles fail with a named error instead of running a different thermostat.
 * Style name -> factory, merged into FIX_STYLES in styles.ts.
 */
const unsupported = (style: string): FixFactory => () => {
  throw new StyleError(`fix ${style} is not supported by the browser engine: its Nose-Hoover chain for rigid bodies (keywords temp, tparam) does not reproduce native LAMMPS yet`);
};

export const FIXES: Record<string, FixFactory> = {
  'rigid/nvt': unsupported('rigid/nvt'),
  'rigid/nvt/small': unsupported('rigid/nvt/small'),
};
