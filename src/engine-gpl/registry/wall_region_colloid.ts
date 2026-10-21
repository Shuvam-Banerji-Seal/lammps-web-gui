/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the colloid style of fix wall/region
 * (src/fix_wall_region.cpp at b332f28, ../wall/fix_wall_region_colloid.ts), as the hook the core fix calls (lazy group
 * gpl_wall_region_colloid, HOOKS word colloid, loaded by Session.execute).
 */
import { colloidWall, colloidWallForce } from '../wall/fix_wall_region_colloid';

export const HOOKS = { colloid: { colloidWall, colloidWallForce } };
