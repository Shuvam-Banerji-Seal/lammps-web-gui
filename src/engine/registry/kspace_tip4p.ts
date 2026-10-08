import type { KSpace } from '../force/types';
import { KSpacePPPMTIP4P } from '../force/kspace/pppm_tip4p';

/** TIP4P kspace styles (Haiku wave 12); merged into styles.ts. Style name -> factory. */
export const KSPACES: Record<string, () => KSpace> = {
  'pppm/tip4p': () => new KSpacePPPMTIP4P(),
};
