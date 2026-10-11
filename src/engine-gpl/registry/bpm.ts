/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BPM/bond_bpm.cpp, bond_bpm_rotational.cpp and their headers at commit b332f28 (Joel Clemmer, SNL,
 * and the BPM package authors). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * BPM bond styles ported from LAMMPS; registered for the core through src/engine/lazyStyles.ts (group bpm59).
 */
import type { Bonded, Pair } from '../../engine/force/types';
import { BondBpmPeri } from '../bpm/bond_bpm_peri';
import { BondBpmRotational } from '../bpm/bond_bpm_rotational';
import { BondBpmSpringPlastic } from '../bpm/bond_bpm_spring_plastic';
import { BondBpmZero } from '../bpm/bond_bpm_zero';
import { PairBpmPeri } from '../bpm/pair_bpm_peri';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const BONDS: Record<string, () => Bonded> = {
  'bpm/rotational': () => new BondBpmRotational(),
  'bpm/spring/plastic': () => new BondBpmSpringPlastic(),
  'bpm/zero': () => new BondBpmZero(),
  'bpm/peri': () => new BondBpmPeri(),
};

export const PAIRS: Record<string, () => Pair> = {
  'bpm/peri': () => new PairBpmPeri(),
};
