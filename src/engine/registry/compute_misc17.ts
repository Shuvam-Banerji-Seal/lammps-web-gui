import type { ComputeFactory } from '../styles';
import { ComputeBondedEnergy } from '../compute/bonded_energy';

/**
 * compute bond, angle, dihedral and improper: per-sub-style energies, valid only when the matching
 * *_style is hybrid (compute/bonded_energy.ts). Merged into styles.ts. Style name -> factory.
 */
export const COMPUTES: Record<string, ComputeFactory> = {
  bond: (sys, id, group, args) => new ComputeBondedEnergy(sys, id, group, args, 'bond'),
  angle: (sys, id, group, args) => new ComputeBondedEnergy(sys, id, group, args, 'angle'),
  dihedral: (sys, id, group, args) => new ComputeBondedEnergy(sys, id, group, args, 'dihedral'),
  improper: (sys, id, group, args) => new ComputeBondedEnergy(sys, id, group, args, 'improper'),
};
