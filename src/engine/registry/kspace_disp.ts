import type { KSpace } from '../force/types';
import { KSpaceEwaldDisp } from '../force/kspace/ewald_disp';

/** kspace_style ewald/disp (wave 16); merged into styles.ts. Style name -> factory. */
export const KSPACES: Record<string, () => KSpace> = {
  'ewald/disp': () => new KSpaceEwaldDisp(),
};
