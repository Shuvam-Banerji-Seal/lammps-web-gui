import type { FixFactory } from '../styles';
import type { Pair } from '../force/types';
import { PairBodyNParticle } from '../force/pair/body_nparticle';
import { FixNVEBody } from '../fix/nve_body';

/*
 * Body particles (atom_style body nparticle; wave 48). merged into styles.ts through lazyStyles.ts.
 * Not registered, so refused by name by the style lookup ("pair_style 'body/rounded/polygon' is not supported by
 * the browser engine"): pair body/rounded/polygon, fix wall/body/polygon, fix nvt/body, npt/body and nph/body. The
 * rounded/polygon bodies are read by read_data (wave 49, tests/engineBodyPoly49.test.ts) but no style computes
 * their contacts yet.
 */
export const PAIRS: Record<string, () => Pair> = {
  'body/nparticle': () => new PairBodyNParticle(),
};
export const FIXES: Record<string, FixFactory> = {
  'nve/body': (s, i, g, a) => new FixNVEBody(s, i, g, a),
};
