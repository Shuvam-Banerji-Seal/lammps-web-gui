import type { FixFactory } from '../styles';

/*
 * The Nose-Hoover rigid styles (rigid/nvt, rigid/npt, rigid/nph and their /small variants) are native-parity ports in
 * src/engine-gpl/registry/fix_rigid_nh.ts (lazy group gpl_rigid_nh). Nothing is refused here any more.
 * Style name -> factory, merged into FIXES in styles.ts.
 */
export const FIXES: Record<string, FixFactory> = {};
