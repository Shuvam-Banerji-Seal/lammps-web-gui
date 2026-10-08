import type { Bonded } from '../force/types';

/** CHARMM dihedral styles (Haiku wave 13); merged into styles.ts. Style name -> factory. */
export const DIHEDRALS: Record<string, () => Bonded> = {};
