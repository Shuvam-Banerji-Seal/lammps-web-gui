import type { ComputeFactory } from '../styles';
import { ComputeTermLocal, ComputePropertyLocal } from '../compute/local';
import { StyleError } from '../force/types';

/** Computes of the Haiku wave 11 (compute_local); merged into styles.ts. Style name -> factory. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'bond/local': (sys, id, group, args) => new ComputeTermLocal(sys, id, group, args, 'bond'),
  'angle/local': (sys, id, group, args) => new ComputeTermLocal(sys, id, group, args, 'angle'),
  'dihedral/local': (sys, id, group, args) => new ComputeTermLocal(sys, id, group, args, 'dihedral'),
  'improper/local': (sys, id, group, args) => new ComputeTermLocal(sys, id, group, args, 'improper'),
  'property/local': (sys, id, group, args) => new ComputePropertyLocal(sys, id, group, args),
  'pair/local': () => {
    throw new StyleError('compute pair/local is not supported by the browser engine yet (the pair neighbour-list order is not reproduced)');
  },
};
