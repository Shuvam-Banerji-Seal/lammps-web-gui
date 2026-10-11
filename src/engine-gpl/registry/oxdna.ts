/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the CG-DNA package's style registry
 * (src/CG-DNA/bond_oxdna_fene.cpp, pair_oxdna_excv.cpp, pair_oxdna_stk.cpp, pair_oxdna_hbond.cpp,
 * pair_oxdna_xstk.cpp, pair_oxdna_coaxstk.cpp; the oxDNA2 and oxRNA2 styles are not ported yet). Original authors (from the headers): Oliver Henrich (University of Strathclyde, Glasgow) and
 * others. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (the names of this module are listed there).
 */
import type { Bonded, Pair } from '../../engine/force/types';
import { BondOxdnaFene } from '../oxdna/bond_fene';
import { PairOxdnaExcv } from '../oxdna/pair_excv';
import { PairOxdnaStk } from '../oxdna/pair_stk';
import { PairOxdnaHbond } from '../oxdna/pair_hbond';
import { PairOxdnaXstk } from '../oxdna/pair_xstk';
import { PairOxdnaCoaxstk } from '../oxdna/pair_coaxstk';

/** bond_style names. */
export const BONDS: Record<string, () => Bonded> = {
  'oxdna/fene': () => new BondOxdnaFene(),
};

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'oxdna/excv': () => new PairOxdnaExcv(),
  'oxdna/stk': () => new PairOxdnaStk(),
  'oxdna/hbond': () => new PairOxdnaHbond(),
  'oxdna/xstk': () => new PairOxdnaXstk(),
  'oxdna/coaxstk': () => new PairOxdnaCoaxstk(),
};
