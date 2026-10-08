import type { ComputeFactory } from '../styles';
import { ComputeSnaAtom, ComputeSnaDeriv } from '../compute/sna';

/** compute sna/atom, snad/atom and snav/atom (Haiku wave); merged into styles.ts. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'sna/atom': (sys, id, group, args) => new ComputeSnaAtom(sys, id, group, args),
  'snad/atom': (sys, id, group, args) => new ComputeSnaDeriv(sys, id, group, args, 'snad'),
  'snav/atom': (sys, id, group, args) => new ComputeSnaDeriv(sys, id, group, args, 'snav'),
};
