import type { Pair } from '../force/types';
import { PairBornCoulDsf, PairBornCoulWolf, PairBornCoulDsfCS, PairBornCoulWolfCS } from '../force/pair/born_coul18';

/** misc pair styles (wave 18, space-bunny swarm); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  'born/coul/wolf': () => new PairBornCoulWolf(),
  'born/coul/dsf': () => new PairBornCoulDsf(),
  'born/coul/wolf/cs': () => new PairBornCoulWolfCS(),
  'born/coul/dsf/cs': () => new PairBornCoulDsfCS(),
};
