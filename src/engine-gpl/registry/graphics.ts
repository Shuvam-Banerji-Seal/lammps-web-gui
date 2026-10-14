/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/fix_graphics_objects.cpp, fix_graphics_lines.cpp, fix_graphics_arrows.cpp (ported) and
 * fix_graphics_labels.cpp, fix_graphics_periodic.cpp, fix_graphics_chunk.cpp, fix_graphics_replica.cpp,
 * fix_graphics_isosurface.cpp (refused) at commit b332f28. Copyright (2003) Sandia Corporation, GNU GPL. This file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * GRAPHICS package fixes: they provide the objects that dump image draws through its "fix" keyword. Registered
 * through src/engine/lazyStyles.ts (group gpl_graphics). The fixes that are not ported throw a StyleError naming the
 * reason; they are listed in src/engine/refusedStyles.ts.
 */
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import { makeFixGraphicsArrows } from '../graphics/fix_graphics_arrows';
import { makeFixGraphicsLines } from '../graphics/fix_graphics_lines';
import { makeFixGraphicsObjects } from '../graphics/fix_graphics_objects';

const refuse = (style: string, reason: string): FixFactory => () => {
  throw new StyleError(`fix graphics/${style} is not supported by the browser engine: ${reason}`);
};

/** fix names. */
export const FIXES: Record<string, FixFactory> = {
  'graphics/objects': makeFixGraphicsObjects,
  'graphics/lines': makeFixGraphicsLines,
  'graphics/arrows': makeFixGraphicsArrows,
  'graphics/labels': refuse('labels', 'text labels need the scalable font renderer (fix_graphics_labels.cpp uses scalable_font.cpp), which is not ported'),
  'graphics/periodic': refuse('periodic', 'the periodic-image objects (fix_graphics_periodic.cpp) are not ported'),
  'graphics/chunk': refuse('chunk', 'it needs compute chunk/atom and the convex-hull objects of image_objects.cpp, which are not ported'),
  'graphics/replica': refuse('replica', 'the replica objects (fix_graphics_replica.cpp) are not ported'),
  'graphics/isosurface': refuse('isosurface', 'the marching-cubes isosurface objects (fix_graphics_isosurface.cpp) are not ported'),
};
