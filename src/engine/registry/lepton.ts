import type { FixFactory } from '../styles';
import type { Bonded, Pair } from '../force/types';
import { PairLepton } from '../force/pair/lepton';
import { BondLepton } from '../force/bond/lepton';
import { AngleLepton } from '../force/angle/lepton';
import { DihedralLepton } from '../force/dihedral/lepton';
import { FixWallLepton, FixEfieldLepton } from '../fix/lepton';

/** Lepton-expression styles (wave 16); merged into styles.ts. Style name -> factory. */
export const PAIRS: Record<string, () => Pair> = {
  lepton: () => new PairLepton('lepton'),
  'lepton/coul': () => new PairLepton('lepton/coul'),
  'lepton/sphere': () => new PairLepton('lepton/sphere'),
};
export const BONDS: Record<string, () => Bonded> = {
  lepton: () => new BondLepton(),
};
export const ANGLES: Record<string, () => Bonded> = {
  lepton: () => new AngleLepton(),
};
export const DIHEDRALS: Record<string, () => Bonded> = {
  lepton: () => new DihedralLepton(),
};
export const FIXES: Record<string, FixFactory> = {
  'wall/lepton': (sys, id, group, args) => new FixWallLepton(sys, id, group, args),
  'efield/lepton': (sys, id, group, args) => new FixEfieldLepton(sys, id, group, args),
};
