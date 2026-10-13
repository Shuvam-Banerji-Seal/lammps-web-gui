/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_viscous_sphere.cpp at commit b332f28 (see src/engine-gpl/extrafix/fix_viscous_sphere.ts for the
 * per-file sources). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * EXTRA-FIX and ORIENT fix styles, registered through src/engine/lazyStyles.ts (group gpl_extrafix_misc).
 */
import type { FixFactory } from '../../engine/styles';
import { FixViscousSphere } from '../extrafix/fix_viscous_sphere';
import { FixFlowGauss } from '../extrafix/fix_flow_gauss';
import { FixWallEES } from '../extrafix/fix_wall_ees';
import { FixBAOAB } from '../extrafix/fix_baoab';
import { FixOrientECO } from '../extrafix/fix_orient_eco';
import { FixWallRegionEES } from '../extrafix/fix_wall_region_ees';

export const FIXES: Record<string, FixFactory> = {
  'viscous/sphere': (sys, id, group, args) => new FixViscousSphere(sys, id, group, args),
  'flow/gauss': (sys, id, group, args) => new FixFlowGauss(sys, id, group, args),
  'wall/ees': (sys, id, group, args) => new FixWallEES(sys, id, group, args),
  baoab: (sys, id, group, args) => new FixBAOAB(sys, id, group, args),
  'orient/eco': (sys, id, group, args) => new FixOrientECO(sys, id, group, args),
  'wall/region/ees': (sys, id, group, args) => new FixWallRegionEES(sys, id, group, args),
};
