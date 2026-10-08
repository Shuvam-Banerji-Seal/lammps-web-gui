import type { KSpace, Pair } from '../force/types';

/** dipole pair styles and kspace (wave 18); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {};
export const KSPACES: Record<string, () => KSpace> = {};
