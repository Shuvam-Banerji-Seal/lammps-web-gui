import type { Bonded } from '../force/types';
import { ImproperClass2 } from '../force/improper/class2';

/** The class2 (COMPASS) improper style (wave 35); merged into styles.ts. The angle and dihedral styles are in engine-gpl/registry/angle_class2 and dihedral_class2. */
export const IMPROPERS: Record<string, () => Bonded> = { class2: () => new ImproperClass2() };
