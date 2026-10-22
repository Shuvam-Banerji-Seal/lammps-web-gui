/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/fix_graphics_objects.cpp, fix_graphics_arrows.cpp and fix_graphics_lines.cpp (the argument
 * parsing and PARSE_VARIABLE rule shared by the graphics fixes). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * The GRAPHICS source headers name no contributing author; Axel Kohlmeyer (Temple U) is the contributing author of
 * image.cpp and dump_image.cpp, the renderer these fixes feed.
 *
 * A number argument may be "v_name": the value is an equal-style variable evaluated when the image is made
 * (PARSE_VARIABLE in fix_graphics_objects.cpp: "if the value begins with v_ it is an equal-style variable").
 */
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { inumeric, numeric } from '../../engine/force/util';

/** A number or an equal-style variable name. */
export type NumOrVar = { num: number } | { variable: string };

export const parseNumOrVar = (w: string | undefined, what: string): NumOrVar => {
  if (w === undefined) throw new StyleError(`${what}: missing value`);
  if (w.startsWith('v_')) return { variable: w.slice(2) };
  return { num: numeric(w) };
};

export const evalNumOrVar = (sys: System, x: NumOrVar): number => ('num' in x ? x.num : sys.equalVariable(x.variable));

/** The graphics fixes read their numbers with utils::inumeric and utils::numeric (`what` documents the call site). */
export const parseInteger = (w: string | undefined, _what: string): number => inumeric(w);

export const parseNumber = (w: string | undefined, _what: string): number => numeric(w);
