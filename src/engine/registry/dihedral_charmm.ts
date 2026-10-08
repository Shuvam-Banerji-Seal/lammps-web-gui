import type { Bonded } from '../force/types';
import { DihedralCharmm, DihedralCharmmfsw } from '../force/dihedral/charmm';

/** CHARMM dihedral styles (Haiku wave 13); merged into styles.ts. Style name -> factory. */
export const DIHEDRALS: Record<string, () => Bonded> = {
  charmm: () => new DihedralCharmm(),
  charmmfsw: () => new DihedralCharmmfsw(),
};
