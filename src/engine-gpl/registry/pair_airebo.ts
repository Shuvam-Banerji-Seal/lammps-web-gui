/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_airebo.cpp, pair_airebo.h, pair_rebo.cpp, pair_rebo.h,
 * pair_airebo_morse.cpp and pair_airebo_morse.h, at commit b332f28.
 * Authors credited in the C++ file headers: Ase Henry (MIT); bugfixes and optimizations:
 * Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer (Temple U), Markus Hoehnerbach
 * (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M: Thomas C. O'Connor (JHU) 2014.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { Pair } from '../../engine/force/types';
import { PairAIREBO } from '../airebo/airebo';

/** Many-body REBO styles ported from LAMMPS (pair_airebo); merged into styles.ts through lazyStyles.ts. */
export const PAIRS: Record<string, () => Pair> = {
  rebo: () => new PairAIREBO('rebo'),
  airebo: () => new PairAIREBO('airebo'),
  'airebo/morse': () => new PairAIREBO('airebo/morse'),
  'airebo/bc': () => new PairAIREBO('airebo/bc'),
};
