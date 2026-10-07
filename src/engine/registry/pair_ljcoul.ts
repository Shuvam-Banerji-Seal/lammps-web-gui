import type { Pair } from '../force/types';
import { PairLJCutCoulCut, PairLJCutCoulDebye, PairLJCutCoulDsf, PairLJCutCoulWolf } from '../force/pair/lj_coul';

/** Pair styles added by the wave-1 rerun (pair_ljcoul); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'lj/cut/coul/cut': () => new PairLJCutCoulCut(),
  'lj/cut/coul/debye': () => new PairLJCutCoulDebye(),
  'lj/cut/coul/dsf': () => new PairLJCutCoulDsf(),
  'lj/cut/coul/wolf': () => new PairLJCutCoulWolf(),
};
