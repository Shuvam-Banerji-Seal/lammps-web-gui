import type { Bonded } from '../force/types';

/** bond styles (wave 17, GLM worker); merged into styles.ts. Style name -> factory. */
export const BONDS: Record<string, () => Bonded> = {};
