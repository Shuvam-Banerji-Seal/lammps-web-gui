/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the pair styles beck, born/gauss,
 * buck/mdf, lennard/mdf, lj/mdf, cosine/squared, coul/cut/global, coul/exclude, coul/slater/cut, ufm and wf/cut of
 * src/EXTRA-PAIR (see the header of each file in src/engine-gpl/xpair1). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_xpair1; the names listed there must match these keys).
 */
import type { Pair } from '../../engine/force/types';
import { PairBeck } from '../xpair1/pair_beck';
import { PairBornGauss } from '../xpair1/pair_born_gauss';
import { PairBuckMDF, PairLennardMDF, PairLJMDF } from '../xpair1/pair_mdf';
import { PairCosineSquared } from '../xpair1/pair_cosine_squared';
import { PairCoulCutGlobal, PairCoulExclude, PairCoulSlaterCut } from '../xpair1/pair_coul_family';
import { PairUFM } from '../xpair1/pair_ufm';
import { PairWFCut } from '../xpair1/pair_wf_cut';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  beck: () => new PairBeck(),
  'born/gauss': () => new PairBornGauss(),
  'buck/mdf': () => new PairBuckMDF(),
  'lennard/mdf': () => new PairLennardMDF(),
  'lj/mdf': () => new PairLJMDF(),
  'cosine/squared': () => new PairCosineSquared(),
  'coul/cut/global': () => new PairCoulCutGlobal(),
  'coul/exclude': () => new PairCoulExclude(),
  'coul/slater/cut': () => new PairCoulSlaterCut(),
  ufm: () => new PairUFM(),
  'wf/cut': () => new PairWFCut(),
};
