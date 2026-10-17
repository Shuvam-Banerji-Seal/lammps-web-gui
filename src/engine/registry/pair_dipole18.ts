import type { KSpace, Pair } from '../force/types';
import { PairLJCutDipoleCut, PairLJSFDipoleSF } from '../force/pair/dipole';

/**
 * dipole pair styles (wave 18); merged into styles.ts. Style name -> factory. lj/long/dipole/long and its kspace
 * ewald/disp are in the GPL group gpl_ewald_disp (src/engine-gpl/ewald_disp).
 */
export const PAIRS: Record<string, () => Pair> = {
  'lj/cut/dipole/cut': () => new PairLJCutDipoleCut(),
  'lj/sf/dipole/sf': () => new PairLJSFDipoleSF(),
};
export const KSPACES: Record<string, () => KSpace> = {};
