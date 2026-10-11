import type { FixFactory } from '../styles';
import { FixDtReset } from '../fix/dt_reset';
import { FixElectronStopping } from '../fix/electron_stopping';
import { FixAveMoments } from '../fix/ave_moments';

/*
 * Fix styles from the EXTRA/PACKAGES examples (dt/reset, electron/stopping, ave/moments). Style name -> factory.
 * fix electron/stopping/fit is deliberately not registered: docs.lammps.org/fix_electron_stopping.html
 * describes its coefficients but gives no equation for the stopping power, so it is refused as an unknown style.
 */
export const FIXES: Record<string, FixFactory> = {
  'dt/reset': (sys, id, group, args) => new FixDtReset(sys, id, group, args),
  'electron/stopping': (sys, id, group, args) => new FixElectronStopping(sys, id, group, args),
  'ave/moments': (sys, id, group, args) => new FixAveMoments(sys, id, group, args),
};
