/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the reaction (rx) styles of the DPD-REACT package
 * (src/DPD-REACT: fix rx, fix eos/table/rx, pair exp6/rx). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). The registry lists
 * them for the core (src/engine/lazyStyles.ts).
 * pair table/rx and pair multi/lucy/rx are registered only to refuse (their factories throw; src/engine/refusedStyles.ts).
 */
import { StyleError, type Pair } from '../../engine/force/types';
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { PairExp6rx } from '../dpdrx/pair_exp6_rx';
import { FixRX } from '../dpdrx/fix_rx';
import { FixEOStableRX } from '../dpdrx/fix_eos_table_rx';

export const PAIRS: Record<string, () => Pair> = {
  'exp6/rx': () => new PairExp6rx(),
  'table/rx': () => { throw new StyleError('pair table/rx is not ported to the browser engine (pair_table_rx.cpp)'); },
  'multi/lucy/rx': () => { throw new StyleError('pair multi/lucy/rx is not ported to the browser engine (pair_multi_lucy_rx.cpp)'); },
};

export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  rx: (sys, id, group, args) => new FixRX(sys, id, group, args),
  'eos/table/rx': (sys, id, group, args) => new FixEOStableRX(sys, id, group, args),
};
