/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SMTBQ/pair_smtbq.cpp and src/SMTBQ/pair_smtbq.h (style smtbq, PairSMTBQ), at commit b332f28.
 * Authors credited in the C++ file headers: N. Salles, O. Politano, E. Amzallag, R. Tetot (SMTBQ) and E. Maras
 * (Aalto University); Julian Gale (gammas). Copyright (2015) Universite de Bourgogne and Sandia Corporation
 * (LAMMPS). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairSMTBQ } from '../smtbq/pair_smtbq';

/** pair_style smtbq (the SMTBQ package): second-moment tight-binding potential with QEq charges inside the pair style. */
export const PAIRS: Record<string, () => Pair> = {
  smtbq: () => new PairSMTBQ(),
};
