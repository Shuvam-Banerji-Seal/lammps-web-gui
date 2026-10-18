/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/fix_graphics_isosurface.cpp at commit b332f28 (FixStyle(graphics/isosurface,FixGraphicsIsosurface)).
 * Copyright (2003) Sandia Corporation, GNU GPL. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix graphics/isosurface, registered through src/engine/lazyStyles.ts (group gpl_graphics_isosurface). It is a group of
 * its own because the gpl_graphics group is also loaded by the page for dump image's hooks, and this fix reads per-atom
 * compute, fix and variable values that only the engine needs.
 */
import type { FixFactory } from '../../engine/styles';
import { makeFixGraphicsIsosurface } from '../graphics/fix_graphics_isosurface';

/** fix names. */
export const FIXES: Record<string, FixFactory> = {
  'graphics/isosurface': makeFixGraphicsIsosurface,
};
