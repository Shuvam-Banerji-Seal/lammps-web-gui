import type { ComputeFactory } from '../styles';
import { ComputeSnap } from '../compute/snap_global';
import { ComputeSnaGrid, ComputeSnaGridLocal } from '../compute/sna_grid';
import { ComputeGaussianGridLocal } from '../compute/gaussian_grid';

/** compute snap, sna/grid, sna/grid/local (Haiku wave 15), gaussian/grid/local (wave 38); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  snap: (sys, id, group, args) => new ComputeSnap(sys, id, group, args),
  'sna/grid': (sys, id, group, args) => new ComputeSnaGrid(sys, id, group, args),
  'sna/grid/local': (sys, id, group, args) => new ComputeSnaGridLocal(sys, id, group, args),
  'gaussian/grid/local': (sys, id, group, args) => new ComputeGaussianGridLocal(sys, id, group, args),
};
