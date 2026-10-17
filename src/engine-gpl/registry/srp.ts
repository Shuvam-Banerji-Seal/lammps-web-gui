/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/MISC/pair_srp.cpp, fix_srp.cpp, pair_srp_react.cpp and fix_srp_react.cpp (pair_style srp and srp/react
 * and their fixes SRP and SRPREACT, see ../srp). LAMMPS is Copyright Sandia Corporation and distributed
 * under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_srp).
 */
import type { Pair } from '../../engine/force/types';
import type { FixFactory } from '../../engine/styles';
import { PairSRP } from '../srp/pair_srp';
import { FixSRP } from '../srp/fix_srp';
import { PairSRPREACT } from '../srp/pair_srp_react';
import { FixSRPREACT } from '../srp/fix_srp_react';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  srp: () => new PairSRP(),
  'srp/react': () => new PairSRPREACT(),
};

/** fix styles: the fixes pair srp and srp/react make (native FixStyle(SRP,FixSRP), FixStyle(SRPREACT,FixSRPREACT)). */
export const FIXES: Record<string, FixFactory> = {
  SRP: (sys, id, group, args) => new FixSRP(sys, id, group, args),
  SRPREACT: (sys, id, group, args) => new FixSRPREACT(sys, id, group, args),
};
