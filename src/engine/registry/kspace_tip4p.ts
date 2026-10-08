import type { KSpace } from '../force/types';

/** TIP4P kspace styles (Haiku wave 12); merged into styles.ts. Style name -> factory. */
export const KSPACES: Record<string, () => KSpace> = {};
