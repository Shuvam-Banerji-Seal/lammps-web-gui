/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MESONT/bond_mesocnt.cpp, angle_mesocnt.cpp and pair_mesocnt.cpp at commit b332f28 (contributing author
 * Philipp Kloza, University of Cambridge). Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * MESONT package styles (mesoscopic carbon nanotubes) ported from LAMMPS; registered for the core through
 * src/engine/lazyStyles.ts (group gpl_mesont).
 */
import type { Bonded, Pair } from '../../engine/force/types';
import { AngleMesoCNT } from '../mesont/angle_mesocnt';
import { BondMesoCNT } from '../mesont/bond_mesocnt';
import { PairMesoCNT, PairMesoCNTViscous } from '../mesont/pair_mesocnt';

/** bond_style mesocnt (docs.lammps.org/bond_mesocnt.html). */
export const BONDS: Record<string, () => Bonded> = { mesocnt: () => new BondMesoCNT() };
/** angle_style mesocnt (docs.lammps.org/angle_mesocnt.html). */
export const ANGLES: Record<string, () => Bonded> = { mesocnt: () => new AngleMesoCNT() };
/** pair_style mesocnt (docs.lammps.org/pair_mesocnt.html). */
export const PAIRS: Record<string, () => Pair> = {
  mesocnt: () => new PairMesoCNT(),
  'mesocnt/viscous': () => new PairMesoCNTViscous(),
};
