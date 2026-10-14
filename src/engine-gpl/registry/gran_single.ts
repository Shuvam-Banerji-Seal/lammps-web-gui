/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the single() functions of
 * src/GRANULAR/pair_gran_hooke.cpp, pair_gran_hooke_history.cpp and pair_gran_hertz_history.cpp (see granular/gran_single.ts).
 * Contributing authors (from the sources): Leo Silbert (SNL), Gary Grest (SNL). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (kind HOOKS, one word per gran style): the core pair gran
 * reads its helper with gplHook(<style name>) (src/engine/force/pair/gran.ts).
 */
import type { GranSingleHook } from '../../engine/force/pair/gran';
import { granSingle } from '../granular/gran_single';

const HELPER: GranSingleHook = {
  single: (kind, p, inp, svector) => granSingle(kind, p, inp, svector),
};

/** The HOOKS entries: the command words that load this group (the three gran pair_style names). */
export const HOOKS: Record<string, GranSingleHook> = {
  'gran/hooke': HELPER,
  'gran/hooke/history': HELPER,
  'gran/hertz/history': HELPER,
};
