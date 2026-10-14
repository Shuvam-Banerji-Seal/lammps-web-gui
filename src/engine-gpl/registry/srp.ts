/*
 * Refusals for styles named after LAMMPS sources. Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28):
 * the names of the SRP package (src/MISC/pair_srp.cpp, pair_srp_react.cpp, fix_srp.cpp, fix_srp_react.cpp). Nothing
 * here is ported: they are registered only so that using them names the reason. LAMMPS is Copyright Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Listed in src/engine/refusedStyles.ts.
 */
import { StyleError } from '../../engine/force/types';
import type { Pair } from '../../engine/force/types';

const SRP_REASON = 'fix srp inserts one bond-midpoint particle per bond as a real owned atom of an extra type, creates and deletes them at every neighbor build (fix_srp.cpp setup_pre_force, pre_exchange), and renumbers the atom arrays; the browser engine keeps a fixed atom set during a run, so this needs a core change that is not made yet';
const REACT_REASON = 'srp/react creates and deletes the same bond-midpoint particles as pair srp (fix_srp_react.cpp), which the browser engine does not support during a run yet';

/** pair_style names: refused. */
export const PAIRS: Record<string, () => Pair> = {
  srp: () => { throw new StyleError(`pair_style srp is not supported by the browser engine yet: ${SRP_REASON}`); },
  'srp/react': () => { throw new StyleError(`pair_style srp/react is not supported by the browser engine yet: ${REACT_REASON}`); },
};

/** fix styles: refused. */
export const FIXES: Record<string, () => never> = {
  srp: () => { throw new StyleError(`fix srp is not supported by the browser engine yet: ${SRP_REASON}`); },
  'srp/react': () => { throw new StyleError(`fix srp/react is not supported by the browser engine yet: ${REACT_REASON}`); },
};
