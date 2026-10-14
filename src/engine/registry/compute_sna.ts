import type { ComputeFactory } from '../styles';
import { ComputeSnaAtom, ComputeSnaDeriv } from '../compute/sna';
import { StyleError } from '../force/types';

/** compute sna/atom, snad/atom and snav/atom (Haiku wave); merged into styles.ts. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'sna/atom': (sys, id, group, args) => new ComputeSnaAtom(sys, id, group, args),
  'snad/atom': (sys, id, group, args) => new ComputeSnaDeriv(sys, id, group, args, 'snad'),
  'snav/atom': (sys, id, group, args) => new ComputeSnaDeriv(sys, id, group, args, 'snav'),
  // refused by name (refusedStyles.ts): native EXTRA-COMPUTE/compute_slcsa_atom.cpp (412 lines) reads five trained model files
  'slcsa/atom': () => {
    throw new StyleError("compute slcsa/atom is not implemented in the browser engine (native EXTRA-COMPUTE/compute_slcsa_atom.cpp reads five trained model files: mean descriptor, LDA scalings, logistic regression decision and bias, Mahalanobis)");
  },
};
