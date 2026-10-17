/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Sources: src/MISC/pair_srp.cpp and fix_srp.cpp (pair_style srp and its fix SRP, see ../srp), and the names of
 * pair_srp_react.cpp and fix_srp_react.cpp, which stay refused. LAMMPS is Copyright Sandia Corporation and distributed
 * under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_srp); srp/react is listed in src/engine/refusedStyles.ts.
 */
import { StyleError } from '../../engine/force/types';
import type { Pair } from '../../engine/force/types';
import type { FixFactory } from '../../engine/styles';
import { PairSRP } from '../srp/pair_srp';
import { FixSRP } from '../srp/fix_srp';

const REACT_REASON = 'srp/react creates and deletes bond particles while fix bond/break and bond/create change the bonds (fix_srp_react.cpp), which is not ported yet';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  srp: () => new PairSRP(),
  'srp/react': () => { throw new StyleError(`pair_style srp/react is not supported by the browser engine yet: ${REACT_REASON}`); },
};

/** fix styles: SRP is the fix pair srp makes (native FixStyle(SRP,FixSRP)); srp/react stays refused. */
export const FIXES: Record<string, FixFactory> = {
  SRP: (sys, id, group, args) => new FixSRP(sys, id, group, args),
  'srp/react': () => { throw new StyleError(`fix srp/react is not supported by the browser engine yet: ${REACT_REASON}`); },
};
