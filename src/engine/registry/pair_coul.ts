import type { Pair } from '../force/types';
import { PairCoulCut, PairCoulDebye, PairCoulDsf, PairCoulWolf } from '../force/pair/coul';

/** Pair styles added by the coul wave; merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'coul/cut': () => new PairCoulCut(),
  'coul/debye': () => new PairCoulDebye(),
  'coul/dsf': () => new PairCoulDsf(),
  'coul/wolf': () => new PairCoulWolf(),
};
