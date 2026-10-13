/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registration of src/LATBOLTZ/fix_lb_fluid.h (FixStyle(lb/fluid,FixLbFluid)), fix_lb_viscous.h
 * (FixStyle(lb/viscous,FixLbViscous)) and fix_lb_momentum.h (FixStyle(lb/momentum,FixLbMomentum)) at commit b332f28.
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_latboltz); the names there must match these keys.
 */
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { FixLbFluid } from '../latboltz/fix_lb_fluid';
import { FixLbMomentum } from '../latboltz/fix_lb_momentum';
import { FixLbViscous } from '../latboltz/fix_lb_viscous';

export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'lb/fluid': (sys, id, group, args) => new FixLbFluid(sys, id, group, args),
  'lb/viscous': (sys, id, group, args) => new FixLbViscous(sys, id, group, args),
  'lb/momentum': (sys, id, group, args) => new FixLbMomentum(sys, id, group, args),
};
