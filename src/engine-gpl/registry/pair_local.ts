/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the pair order of src/compute_pair_local.cpp
 * (compute_pairs() walks the half neighbor list of the occasional list in its stored order) and the enumeration of
 * src/npair_bin.cpp (the half/bin pair order, newton on or off) that builds that list (see neighbor/nativeOrder.ts).
 * LAMMPS is Copyright Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (kind HOOKS, word 'pair/local'): compute pair/local
 * (src/engine/compute/local.ts) reads the native row order through gplHook('pair/local').
 */
import { NativePairOrder } from '../neighbor/nativeOrder';
import type { PairLocalHook } from '../../engine/compute/local';

/** The HOOKS entry 'pair/local' (the contract is PairLocalHook, src/engine/compute/local.ts). */
export const HOOKS: Record<string, PairLocalHook> = {
  'pair/local': { pairOrder: (newton: boolean) => new NativePairOrder(newton) },
};
