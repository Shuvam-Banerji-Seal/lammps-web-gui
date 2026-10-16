/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/min_linesearch.cpp at commit b332f28 (see src/engine-gpl/min/linemin.ts).
 *
 * The lazy HOOKS entries 'minimize', 'hyper', 'prd' and 'tad' (lazyStyles.ts group gpl_linemin): the core minimizer
 * (run/min.ts) reads the default line search through the slot of gplHook('minimize'). The commands hyper, prd and tad run
 * the same minimizer (their quench), so their words load the group too.
 */
import { lineminQuadratic } from '../min/linemin';

const helper = { quadratic: lineminQuadratic };
export const HOOKS: Record<string, { quadratic: typeof lineminQuadratic }> = { minimize: helper, hyper: helper, prd: helper, tad: helper };
