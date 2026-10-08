import type { Pair } from '../force/types';
import { PairLJLongCoulLong } from '../force/pair/lj_long';
import { PairLJCharmmCoulCharmm, PairLJCharmmCoulCharmmImplicit, PairLJCharmmCoulLong, PairLJCharmmfswCoulCharmmfsh, PairLJCharmmfswCoulLong } from '../force/pair/charmm';

/** Pair styles added by wave 3 (pair_charmm); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'lj/charmm/coul/charmm': () => new PairLJCharmmCoulCharmm(),
  'lj/charmm/coul/charmm/implicit': () => new PairLJCharmmCoulCharmmImplicit(),
  'lj/charmm/coul/long': () => new PairLJCharmmCoulLong(),
  'lj/charmmfsw/coul/charmmfsh': () => new PairLJCharmmfswCoulCharmmfsh(),
  'lj/charmmfsw/coul/long': () => new PairLJCharmmfswCoulLong(),
  // lj/long/coul/long (pair_lj_long.html) is registered here with the CHARMM pair styles.
  'lj/long/coul/long': () => new PairLJLongCoulLong(),
};
