import type { Bonded } from '../force/types';
import { AngleClass2P6 } from '../force/angle/class2';
import { ImproperInversionHarmonic } from '../force/improper/inversion';
import { AngleCosineBuck6d } from '../force/angle/buck6d';

/** Bonded styles of wave 50 (class2/p6, inversion/harmonic, cosine/buck6d); merged lazily via lazyStyles.ts. */
export const ANGLES: Record<string, () => Bonded> = {
  'class2/p6': () => new AngleClass2P6(),
  'cosine/buck6d': () => new AngleCosineBuck6d(),
};

export const IMPROPERS: Record<string, () => Bonded> = {
  'inversion/harmonic': () => new ImproperInversionHarmonic(),
};
