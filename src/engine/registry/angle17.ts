import type { Bonded } from '../force/types';
import { AngleGaussian } from '../force/angle/gaussian';

/** angle styles (wave 17, GLM worker); merged into styles.ts. Style name -> factory. */
export const ANGLES: Record<string, () => Bonded> = {
  gaussian: () => new AngleGaussian(),
};
