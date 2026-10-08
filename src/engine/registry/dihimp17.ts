import type { Bonded } from '../force/types';
import { DihedralCosineSquaredRestricted } from '../force/dihedral/misc18';
import { ImproperDistharm, ImproperSqdistharm } from '../force/improper/misc18';

/** dihedral and improper styles (wave 17, GLM worker); merged into styles.ts. Style name -> factory. */
export const DIHEDRALS: Record<string, () => Bonded> = {
  'cosine/squared/restricted': () => new DihedralCosineSquaredRestricted(),
};

export const IMPROPERS: Record<string, () => Bonded> = {
  distharm: () => new ImproperDistharm(),
  sqdistharm: () => new ImproperSqdistharm(),
};