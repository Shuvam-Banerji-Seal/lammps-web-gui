import type { FixFactory } from '../styles';
import type { Pair } from '../force/types';
import { PairBodyNParticle } from '../force/pair/body_nparticle';
import { FixNVEBody } from '../fix/nve_body';

/** body particles (atom_style body nparticle; wave 48); merged into styles.ts through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  'body/nparticle': () => new PairBodyNParticle(),
};
export const FIXES: Record<string, FixFactory> = {
  'nve/body': (s, i, g, a) => new FixNVEBody(s, i, g, a),
};
