/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the pair order of src/compute_pair_local.cpp
 * (compute_pairs() walks the half neighbor list of the occasional list in its stored order) and the enumeration of
 * src/npair_bin.cpp (the half/bin pair order, newton on or off) that builds that list (see neighbor/nativeOrder.ts).
 * LAMMPS is Copyright Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (kind HOOKS, words 'pair/local' and 'property/local'):
 * compute pair/local and the pair attributes of compute property/local (src/engine/compute/local.ts) read the native row
 * order through gplHook('pair/local') and gplHook('property/local').
 */
import { NativePairOrder } from '../neighbor/nativeOrder';
import type { PairLocalHook } from '../../engine/compute/local';

const pairLocal: PairLocalHook = { pairOrder: (newton: boolean) => new NativePairOrder(newton) };

/** The HOOKS entries 'pair/local' and 'property/local' (the contract is PairLocalHook, src/engine/compute/local.ts). */
export const HOOKS: Record<string, PairLocalHook> = {
  'pair/local': pairLocal,
  'property/local': pairLocal,
};
