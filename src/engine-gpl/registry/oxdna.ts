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
import type { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { FixNVEDot } from '../oxdna/fix_nve_dot';
import { FixNVEDotcLangevin } from '../oxdna/fix_nve_dotc_langevin';
import { BondOxdnaFene } from '../oxdna/bond_fene';
import { PairOxdnaExcv } from '../oxdna/pair_excv';
import { PairOxdnaStk } from '../oxdna/pair_stk';
import { PairOxdnaHbond } from '../oxdna/pair_hbond';
import { PairOxdnaXstk } from '../oxdna/pair_xstk';
import { PairOxdnaCoaxstk } from '../oxdna/pair_coaxstk';
import { BondOxdna2Fene, BondOxrna2Fene, PairOxdna2Excv, PairOxrna2Excv } from '../oxdna/variants';
import { PairOxdna2Dh, PairOxrna2Dh } from '../oxdna/pair_dh';
import { PairOxdna2Coaxstk } from '../oxdna/pair_oxdna2_coaxstk';
import { PairOxrna2Stk } from '../oxdna/pair_oxrna2_stk';
import { PairOxrna2Hbond } from '../oxdna/pair_hbond';
import { PairOxrna2Xstk } from '../oxdna/pair_xstk';

/** bond_style names. */
export const BONDS: Record<string, () => Bonded> = {
  'oxdna/fene': () => new BondOxdnaFene(),
  'oxdna2/fene': () => new BondOxdna2Fene(),
  'oxrna2/fene': () => new BondOxrna2Fene(),
};

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'oxdna/excv': () => new PairOxdnaExcv(),
  'oxdna2/excv': () => new PairOxdna2Excv(),
  'oxrna2/excv': () => new PairOxrna2Excv(),
  'oxdna2/dh': () => new PairOxdna2Dh(),
  'oxdna2/coaxstk': () => new PairOxdna2Coaxstk(),
  'oxrna2/hbond': () => new PairOxrna2Hbond(),
  'oxrna2/stk': () => new PairOxrna2Stk(),
  'oxrna2/xstk': () => new PairOxrna2Xstk(),
  'oxrna2/dh': () => new PairOxrna2Dh(),
  // the oxDNA2 stacking, hydrogen-bond and cross-stacking styles and the oxRNA2 coaxial style are the oxDNA1 classes (pair_oxdna_stk.h etc. register them so)
  'oxdna2/stk': () => new PairOxdnaStk(),
  'oxdna2/hbond': () => new PairOxdnaHbond(),
  'oxdna2/xstk': () => new PairOxdnaXstk(),
  'oxrna2/coaxstk': () => new PairOxdnaCoaxstk(),
  'oxdna/stk': () => new PairOxdnaStk(),
  'oxdna/hbond': () => new PairOxdnaHbond(),
  'oxdna/xstk': () => new PairOxdnaXstk(),
  'oxdna/coaxstk': () => new PairOxdnaCoaxstk(),
};

/** fix styles (fix nve/dot, fix_nve_dot.cpp). */
export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'nve/dot': (sys, id, group, args) => new FixNVEDot(sys, id, group, args),
  'nve/dotc/langevin': (sys, id, group, args) => new FixNVEDotcLangevin(sys, id, group, args),
};
