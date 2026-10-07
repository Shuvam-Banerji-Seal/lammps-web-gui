import type { Bonded } from '../force/types';

/** Bond and angle styles added by wave 1; merged into styles.ts. Style name -> factory. */
export const BONDS: Record<string, () => Bonded> = {};
export const ANGLES: Record<string, () => Bonded> = {};
