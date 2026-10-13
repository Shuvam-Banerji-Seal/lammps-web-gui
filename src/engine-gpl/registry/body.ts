/*
 * Body particles ported from LAMMPS (commit b332f28; src/BODY/, GPL-2.0, see src/engine-gpl/body/). merged into
 * styles.ts through the lazy registry entry gpl_body in src/engine/lazyStyles.ts.
 * Registered: pair body/rounded/polygon (friction and damping; no contact history) and pair body/rounded/polyhedron
 * (friction and damping; no contact history; see engine-gpl/body/rounded_polyhedron_law.ts); fix nvt/body, npt/body
 * and nph/body (engine-gpl/body/fix_nh_body.ts), fix wall/body/polygon and wall/body/polyhedron (engine-gpl/body/fix_wall_body.ts)
 * and compute temp/body (engine-gpl/body/compute_temp_body.ts).
 */
import { type Pair } from '../../engine/force/types';
import type { ComputeFactory, FixFactory } from '../../engine/styles';
import { PairBodyRoundedPolygon } from '../body/pair_body_rounded_polygon';
import { PairBodyRoundedPolyhedron } from '../body/pair_body_rounded_polyhedron';
import { FixNHBody } from '../body/fix_nh_body';
import { ComputeTempBody } from '../body/compute_temp_body';
import { FixWallBody } from '../body/fix_wall_body';

export const PAIRS: Record<string, () => Pair> = {
  'body/rounded/polygon': () => new PairBodyRoundedPolygon(),
  'body/rounded/polyhedron': () => new PairBodyRoundedPolyhedron(),
};
export const FIXES: Record<string, FixFactory> = {
  'nvt/body': (s, i, g, a) => new FixNHBody(s, i, g, a, 'nvt'),
  'npt/body': (s, i, g, a) => new FixNHBody(s, i, g, a, 'npt'),
  'nph/body': (s, i, g, a) => new FixNHBody(s, i, g, a, 'nph'),
  'wall/body/polygon': (s, i, g, a) => new FixWallBody(s, i, g, a, 'polygon'),
  'wall/body/polyhedron': (s, i, g, a) => new FixWallBody(s, i, g, a, 'polyhedron'),
};
export const COMPUTES: Record<string, ComputeFactory> = {
  'temp/body': (s, i, g, a) => new ComputeTempBody(s, i, g, a),
};
