/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_filter_corotate.cpp (and .h) at commit b332f28 (Lukas Fath, KIT). Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * fix filter/corotate, registered for the core through src/engine/lazyStyles.ts (group gpl_filtercorotate). It works with
 * run_style respa only (src/engine/run/respa.ts calls its RespaOuterFilter hooks).
 */
import type { FixFactory } from '../../engine/styles';
import { makeFixFilterCorotate } from '../filtercorotate/fix_filter_corotate';

/** Style name -> factory. */
export const FIXES: Record<string, FixFactory> = {
  'filter/corotate': makeFixFilterCorotate,
};
