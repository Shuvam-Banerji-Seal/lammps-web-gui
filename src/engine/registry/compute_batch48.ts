import type { ComputeFactory } from '../styles';
import { ComputeCnpAtom } from '../compute/cnp';
import { ComputeTempProfile } from '../compute/temp_profile';

/** Computes of batch 48 (cnp/atom, temp/profile); merged through lazyStyles.ts. entropy/atom is not registered (refused as an unsupported style). compute hma is the GPL port (lazy group gpl_hma). */
export const COMPUTES: Record<string, ComputeFactory> = {
  'cnp/atom': (s, i, g, a) => new ComputeCnpAtom(s, i, g, a),
  'temp/profile': (s, i, g, a) => new ComputeTempProfile(s, i, g, a),
};
