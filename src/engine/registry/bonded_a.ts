import type { Bonded } from '../force/types';
import {
  BondFene, BondFeneExpand, BondMorse, BondNonlinear, BondClass2,
  BondGromos, BondHarmonicShift, BondHarmonicShiftCut, BondZero,
} from '../force/bond/styles';
import {
  AngleCosine, AngleCosineSquared, AngleCosinePeriodic, AngleCosineShift,
  AngleCosineDelta, AngleCharmm, AngleQuartic, AngleFourier,
  AngleFourierSimple, AngleZero,
} from '../force/angle/styles';

/** Bond and angle styles added by wave 1; merged into styles.ts. Style name -> factory. */
export const BONDS: Record<string, () => Bonded> = {
  fene: () => new BondFene(),
  'fene/expand': () => new BondFeneExpand(),
  morse: () => new BondMorse(),
  nonlinear: () => new BondNonlinear(),
  class2: () => new BondClass2(),
  gromos: () => new BondGromos(),
  'harmonic/shift': () => new BondHarmonicShift(),
  'harmonic/shift/cut': () => new BondHarmonicShiftCut(),
  zero: () => new BondZero(),
};

export const ANGLES: Record<string, () => Bonded> = {
  cosine: () => new AngleCosine(),
  'cosine/squared': () => new AngleCosineSquared(),
  'cosine/periodic': () => new AngleCosinePeriodic(),
  'cosine/shift': () => new AngleCosineShift(),
  'cosine/delta': () => new AngleCosineDelta(),
  charmm: () => new AngleCharmm(),
  quartic: () => new AngleQuartic(),
  fourier: () => new AngleFourier(),
  'fourier/simple': () => new AngleFourierSimple(),
  zero: () => new AngleZero(),
};
