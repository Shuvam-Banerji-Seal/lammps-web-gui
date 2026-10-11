import type { ComputeFactory } from '../styles';
import { ComputeCnpAtom } from '../compute/cnp';
import { ComputeTempProfile } from '../compute/temp_profile';
import { ComputeHma } from '../compute/hma';

/** Computes of batch 48 (cnp/atom, temp/profile, hma); merged through lazyStyles.ts. entropy/atom is not registered (refused as an unsupported style). */
export const COMPUTES: Record<string, ComputeFactory> = {
  'cnp/atom': (s, i, g, a) => new ComputeCnpAtom(s, i, g, a),
  'temp/profile': (s, i, g, a) => new ComputeTempProfile(s, i, g, a),
  hma: (s, i, g, a) => new ComputeHma(s, i, g, a),
};
