import type { Bonded } from '../force/types';
import {
  DihedralOpls, DihedralMultiHarmonic, DihedralFourier, DihedralQuadratic,
  DihedralNHarmonic, DihedralCosineShiftExp, DihedralHelix, DihedralZero,
} from '../force/dihedral/styles';

/** Dihedral and improper styles added by wave 1; merged into styles.ts. Style name -> factory. */
export const DIHEDRALS: Record<string, () => Bonded> = {
  opls: () => new DihedralOpls(),
  'multi/harmonic': () => new DihedralMultiHarmonic(),
  fourier: () => new DihedralFourier(),
  quadratic: () => new DihedralQuadratic(),
  nharmonic: () => new DihedralNHarmonic(),
  'cosine/shift/exp': () => new DihedralCosineShiftExp(),
  helix: () => new DihedralHelix(),
  zero: () => new DihedralZero(),
};
export const IMPROPERS: Record<string, () => Bonded> = {};
