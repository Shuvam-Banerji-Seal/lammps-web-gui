import type { FixFactory } from '../styles';
import type { Pair } from '../force/types';
import { PairBodyNParticle } from '../force/pair/body_nparticle';
import { PairBodyRoundedPolygon } from '../force/pair/body_rounded_polygon';
import { FixNVEBody } from '../fix/nve_body';

/*
 * Body particles (atom_style body nparticle; wave 48). merged into styles.ts through lazyStyles.ts.
 * Registered: pair body/nparticle, pair body/rounded/polygon (wave 50: the normal elastic and cohesive laws, the
 * delta_ua scaling, the torques of the normal forces and the c_n and c_t damping; tests/engineBodyContact50.test.ts,
 * oracle cases w50bc_*) and fix nve/body. Refused by name (the style lookup gives "not supported by the browser
 * engine"): pair body/rounded/polygon with mu != 0 (friction) or the history keyword, fix wall/body/polygon, fix
 * nvt/body, npt/body and nph/body.
 */
export const PAIRS: Record<string, () => Pair> = {
  'body/nparticle': () => new PairBodyNParticle(),
  'body/rounded/polygon': () => new PairBodyRoundedPolygon(),
};
export const FIXES: Record<string, FixFactory> = {
  'nve/body': (s, i, g, a) => new FixNVEBody(s, i, g, a),
};
