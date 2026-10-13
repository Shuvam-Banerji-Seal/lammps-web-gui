/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/ (pair styles kolmogorov/crespi/z and lebedeva/z), at commit b332f28; see the headers of
 * the files in src/engine-gpl/interlayer/. Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairCoulShield } from '../interlayer/coul_shield';
import { PairILPGrapheneHBN } from '../interlayer/ilp_graphene_hbn';
import { PairKolmogorovCrespiFull } from '../interlayer/kolmogorov_crespi_full';
import { PairKolmogorovCrespiZ } from '../interlayer/kolmogorov_crespi_z';
import { PairLebedevaZ } from '../interlayer/lebedeva_z';

/** INTERLAYER pair styles ported from LAMMPS; merged into the registries through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  'kolmogorov/crespi/z': () => new PairKolmogorovCrespiZ(),
  'coul/shield': () => new PairCoulShield(),
  'ilp/graphene/hbn': () => new PairILPGrapheneHBN(),
  'kolmogorov/crespi/full': () => new PairKolmogorovCrespiFull(),
  'lebedeva/z': () => new PairLebedevaZ(),
};
