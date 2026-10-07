import type { Bonded } from '../force/types';

/** Dihedral and improper styles added by wave 1; merged into styles.ts. Style name -> factory. */
export const DIHEDRALS: Record<string, () => Bonded> = {};
export const IMPROPERS: Record<string, () => Bonded> = {};
