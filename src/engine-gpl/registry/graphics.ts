/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/fix_graphics_objects.cpp, fix_graphics_lines.cpp, fix_graphics_arrows.cpp,
 * fix_graphics_labels.cpp, fix_graphics_periodic.cpp, fix_graphics_chunk.cpp and fix_graphics_replica.cpp at commit
 * b332f28 (fix graphics/isosurface is in ./graphics_isosurface.ts). Copyright (2003) Sandia Corporation, GNU GPL. This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * GRAPHICS package fixes: they provide the objects that dump image draws through its "fix" keyword. Registered
 * through src/engine/lazyStyles.ts (group gpl_graphics). Drawing of the objects: src/engine-gpl/graphics/draw.ts.
 */
import type { FixFactory } from '../../engine/styles';
import { makeFixGraphicsArrows } from '../graphics/fix_graphics_arrows';
import { makeFixGraphicsLines } from '../graphics/fix_graphics_lines';
import { makeFixGraphicsObjects } from '../graphics/fix_graphics_objects';
import { makeFixGraphicsLabels } from '../graphics/fix_graphics_labels';
import { makeFixGraphicsPeriodic } from '../graphics/fix_graphics_periodic';
import { makeFixGraphicsReplica } from '../graphics/fix_graphics_replica';
import { makeFixGraphicsChunk } from '../graphics/fix_graphics_chunk';
import { imageShadingHook } from '../graphics/image_shading';
import { atomObjectsHook } from '../graphics/atom_objects';
import { regionHook } from '../graphics/region_box';

/** fix names. */
export const FIXES: Record<string, FixFactory> = {
  'graphics/objects': makeFixGraphicsObjects,
  'graphics/lines': makeFixGraphicsLines,
  'graphics/arrows': makeFixGraphicsArrows,
  'graphics/labels': makeFixGraphicsLabels,
  'graphics/periodic': makeFixGraphicsPeriodic,
  'graphics/chunk': makeFixGraphicsChunk,
  'graphics/replica': makeFixGraphicsReplica,
};

/**
 * HOOKS: the shading and ssao helpers dump image reads through gplHook('image') (image_shading.ts): the draw_pixel
 * light sum with dump image's shiny, and the compute_SSAO occlusion pass.
 */
export const HOOKS = { image: imageShadingHook, atoms: atomObjectsHook, region: regionHook };
