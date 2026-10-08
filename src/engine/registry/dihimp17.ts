import type { Bonded } from '../force/types';

/** dihedral and improper styles (wave 17, GLM worker); merged into styles.ts. Style name -> factory. */
export const DIHEDRALS: Record<string, () => Bonded> = {};
export const IMPROPERS: Record<string, () => Bonded> = {};
