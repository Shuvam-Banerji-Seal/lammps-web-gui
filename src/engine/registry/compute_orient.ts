import type { ComputeFactory } from '../styles';
import { ComputeOrientorderAtom } from '../compute/orientorder';

/** Compute styles added by wave 4 (compute_orient); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'orientorder/atom': (sys, id, group, args) => new ComputeOrientorderAtom(sys, id, group, args),
};
