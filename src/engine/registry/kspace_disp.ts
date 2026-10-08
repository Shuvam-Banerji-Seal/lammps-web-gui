import type { KSpace } from '../force/types';

/** kspace_style ewald/disp (wave 16); merged into styles.ts. Style name -> factory. */
export const KSPACES: Record<string, () => KSpace> = {};
