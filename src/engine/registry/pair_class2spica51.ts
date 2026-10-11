import type { Bonded, Pair } from '../force/types';
import { StyleError } from '../force/types';
import { PairLJClass2Coul, PairLJSpica, SPICA_MSM_REFUSAL } from '../force/pair/class2_spica51';
import { AngleSpica } from '../force/angle/spica51';

/*
 * class 2 Coulomb styles, SPICA (lj/spica, lj/sdk) pair styles and the spica / sdk angle styles (batch 51);
 * merged through lazyStyles.ts. Style name -> factory. The msm variants are refused (refusedStyles.ts).
 */
export const PAIRS: Record<string, () => Pair> = {
  'lj/class2/coul/cut': () => new PairLJClass2Coul('lj/class2/coul/cut'),
  'lj/class2/coul/long': () => new PairLJClass2Coul('lj/class2/coul/long'),
  'lj/spica': () => new PairLJSpica('lj/spica'),
  'lj/spica/coul/long': () => new PairLJSpica('lj/spica/coul/long'),
  'lj/sdk': () => new PairLJSpica('lj/sdk'),
  'lj/sdk/coul/long': () => new PairLJSpica('lj/sdk/coul/long'),
  'lj/spica/coul/msm': () => { throw new StyleError(SPICA_MSM_REFUSAL('lj/spica/coul/msm')); },
  'lj/sdk/coul/msm': () => { throw new StyleError(SPICA_MSM_REFUSAL('lj/sdk/coul/msm')); },
};

export const ANGLES: Record<string, () => Bonded> = {
  spica: () => new AngleSpica('spica'),
  sdk: () => new AngleSpica('sdk'),
};
