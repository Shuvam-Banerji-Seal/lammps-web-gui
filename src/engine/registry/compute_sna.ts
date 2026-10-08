import type { ComputeFactory } from '../styles';
import { ComputeSnaAtom } from '../compute/sna';

/** compute sna/atom family (Haiku wave); merged into styles.ts. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'sna/atom': (sys, id, group, args) => new ComputeSnaAtom(sys, id, group, args),
};
