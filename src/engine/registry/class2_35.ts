import type { Bonded } from '../force/types';
import { AngleClass2 } from '../force/angle/class2';
import { DihedralClass2 } from '../force/dihedral/class2';
import { ImproperClass2 } from '../force/improper/class2';

/** class2 (COMPASS) angle, dihedral and improper styles (wave 35); merged into styles.ts. */
export const ANGLES: Record<string, () => Bonded> = { class2: () => new AngleClass2() };
export const DIHEDRALS: Record<string, () => Bonded> = { class2: () => new DihedralClass2() };
export const IMPROPERS: Record<string, () => Bonded> = { class2: () => new ImproperClass2() };
