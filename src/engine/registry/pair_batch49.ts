import { PairBuck6dCoulGaussDsf, PairBuck6dCoulGaussLong } from '../force/pair/buck6d_gauss';
import { PairCoulDiel, PairGaussCut } from '../force/pair/gauss_diel';
import { PairMomb } from '../force/pair/momb';

/** Pair styles of batch 49 (buck6d/coul/gauss/dsf and /long, gauss/cut, coul/diel, ...); merged through lazyStyles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => unknown> = {
  'buck6d/coul/gauss/dsf': () => new PairBuck6dCoulGaussDsf(),
  'buck6d/coul/gauss/long': () => new PairBuck6dCoulGaussLong(),
  'gauss/cut': () => new PairGaussCut(),
  'coul/diel': () => new PairCoulDiel(),
  momb: () => new PairMomb(),
};
