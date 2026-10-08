import type { Bonded } from '../force/types';
import { AngleCosineSquaredRestricted, AngleMm3 } from '../force/angle/misc18';

/** angle styles (wave 18, space-bunny swarm); merged into styles.ts. Style name -> factory. */
export const ANGLES: Record<string, () => Bonded> = {
  'cosine/squared/restricted': () => new AngleCosineSquaredRestricted(),
  mm3: () => new AngleMm3(),
};
