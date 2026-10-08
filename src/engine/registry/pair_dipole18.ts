import type { KSpace, Pair } from '../force/types';
import { StyleError } from '../force/types';
import { PairLJCutDipoleCut, PairLJSFDipoleSF } from '../force/pair/dipole';

/** dipole pair styles and kspace (wave 18); merged into styles.ts. Style name -> factory. */
const longUnsupported = (name: string) => (): Pair => {
  throw new StyleError(`pair_style ${name} is not supported: the long-range dipole part needs kspace_style ewald/dipole (not implemented)`);
};

export const PAIRS: Record<string, () => Pair> = {
  'lj/cut/dipole/cut': () => new PairLJCutDipoleCut(),
  'lj/sf/dipole/sf': () => new PairLJSFDipoleSF(),
  'lj/cut/dipole/long': longUnsupported('lj/cut/dipole/long'),
  'lj/long/dipole/long': longUnsupported('lj/long/dipole/long'),
};
export const KSPACES: Record<string, () => KSpace> = {};
