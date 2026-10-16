/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/MC/fix_neighbor_swap.cpp (fix neighbor/swap) and src/VORONOI/compute_voronoi_atom.cpp (the neighbors
 * rows that the fix reads), on the voro++ 0.4.6 port in src/engine-gpl/voro/ (BSD-3-Clause, LICENSE.voro++).
 * LAMMPS is Copyright (2003) Sandia Corporation, distributed under the GNU General Public License version 2; this file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_neighbor_swap): the fix by its style name, and
 * the native neighbour-order helper of compute voronoi/atom under the word 'voronoi/atom' (contract NativeVoroHook in
 * src/engine/compute/voronoi.ts).
 */
import type { FixFactory } from '../../engine/styles';
import { FixNeighborSwap } from '../mc/fix_neighbor_swap';
import { nativeNeighborRows, type NativeNeighborInput, type NativeNeighborRow } from '../voro/nativeNeighbors';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  'neighbor/swap': (sys, id, group, args) => new FixNeighborSwap(sys, id, group, args),
};

export const HOOKS = {
  'voronoi/atom': { rows: (inp: NativeNeighborInput): NativeNeighborRow[] | null => nativeNeighborRows(inp) },
};
