import type { KSpace } from '../force/types';
import { KSpaceEwaldDipole } from '../force/kspace/ewald_dipole';

/** kspace_style ewald/dipole (wave 44); lazily loaded (lazyStyles.ts). Style name -> factory. */
export const KSPACES: Record<string, () => KSpace> = {
  'ewald/dipole': () => new KSpaceEwaldDipole(),
};
