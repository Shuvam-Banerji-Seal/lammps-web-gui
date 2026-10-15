/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): Force::pair_match (src/force.cpp), as the
 * DPD-REACT styles use it to find their pair style: pair_match(name, exact = 1) returns the pair style with that name, or
 * the sub-style of pair_style hybrid with that name. LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairHybrid } from '../../engine/force/pair/hybrid';

/** The pair style named name (exactly), or the hybrid sub-style of that name; null when the system has none. */
export const pairMatch = (p: Pair | null | undefined, name: string): Pair | null => {
  if (!p) return null;
  if (p.name === name) return p;
  if (p instanceof PairHybrid) return p.subs.find((x) => x.name === name)?.style ?? null;
  return null;
};
