/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the region side of fix
 * wall/gran/region (src/GRANULAR/fix_wall_gran_region.cpp and the region styles src/region.cpp, region_block.cpp,
 * region_cylinder.cpp, region_cone.cpp, region_sphere.cpp, region_plane.cpp, region_intersect.cpp, region_union.cpp at
 * b332f28; ../granular/wall_gran_region.ts), as the hook the core fix calls (lazy group gpl_wall_gran_region, HOOKS word
 * wall/gran/region, loaded by Session.execute).
 */
import { createRegionWall } from '../granular/wall_gran_region';

export const HOOKS = { 'wall/gran/region': { createRegionWall } };
