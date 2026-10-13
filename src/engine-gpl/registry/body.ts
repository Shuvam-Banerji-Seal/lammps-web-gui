/*
 * Body particles ported from LAMMPS (commit b332f28; src/BODY/, GPL-2.0, see src/engine-gpl/body/). merged into
 * styles.ts through the lazy registry entry gpl_body in src/engine/lazyStyles.ts.
 * Registered: pair body/rounded/polygon (friction and damping; no contact history) and pair body/rounded/polyhedron
 * (friction and damping; no contact history; see engine-gpl/body/rounded_polyhedron_law.ts).
 */
import { type Pair } from '../../engine/force/types';
import { PairBodyRoundedPolygon } from '../body/pair_body_rounded_polygon';
import { PairBodyRoundedPolyhedron } from '../body/pair_body_rounded_polyhedron';

export const PAIRS: Record<string, () => Pair> = {
  'body/rounded/polygon': () => new PairBodyRoundedPolygon(),
  'body/rounded/polyhedron': () => new PairBodyRoundedPolyhedron(),
};
