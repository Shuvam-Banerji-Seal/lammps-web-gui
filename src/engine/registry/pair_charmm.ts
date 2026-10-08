import type { Pair } from '../force/types';
import { PairLJCharmmCoulCharmm, PairLJCharmmCoulCharmmImplicit, PairLJCharmmCoulLong } from '../force/pair/charmm';

/** Pair styles added by wave 3 (pair_charmm); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'lj/charmm/coul/charmm': () => new PairLJCharmmCoulCharmm(),
  'lj/charmm/coul/charmm/implicit': () => new PairLJCharmmCoulCharmmImplicit(),
  'lj/charmm/coul/long': () => new PairLJCharmmCoulLong(),
};
