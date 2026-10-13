import type { FixFactory } from '../styles';
import type { Pair } from '../force/types';
import { PairBodyNParticle } from '../force/pair/body_nparticle';
import { FixNVEBody } from '../fix/nve_body';

/*
 * Body particles (atom_style body nparticle; wave 48). merged into styles.ts through lazyStyles.ts.
 * Registered: pair body/nparticle and fix nve/body. pair body/rounded/polygon is in the GPL module
 * (src/engine-gpl/registry/body.ts). Refused by name (the style lookup gives "not supported by the browser
 * engine"): fix wall/body/polygon, fix nvt/body, npt/body and nph/body.
 */
export const PAIRS: Record<string, () => Pair> = {
  'body/nparticle': () => new PairBodyNParticle(),
};
export const FIXES: Record<string, FixFactory> = {
  'nve/body': (s, i, g, a) => new FixNVEBody(s, i, g, a),
};
