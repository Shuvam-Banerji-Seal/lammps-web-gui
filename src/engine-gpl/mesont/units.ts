/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MESONT/bond_mesocnt.cpp, angle_mesocnt.cpp and pair_mesocnt.cpp (the unit tables in their coeff()
 * functions), at commit b332f28. Contributing author Philipp Kloza (University of Cambridge). Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * eV -> energy-unit factor per unit style, as the three mesocnt coeff() functions read it (eunit), and the carbon
 * bond constant A_CC of the chirality presets.
 */
import { StyleError } from '../../engine/force/types';

/** LAMMPS: the eunit chain of bond_mesocnt.cpp / angle_mesocnt.cpp / pair_mesocnt.cpp coeff(). */
const EUNIT: Record<string, number> = {
  real: 23.06054966, metal: 1.0, si: 1.6021765e-19, cgs: 1.6021765e-12, electron: 3.674932248e-2,
  micro: 1.6021765e-4, nano: 1.6021765e2,
};

/** LAMMPS: static constexpr double A_CC = 1.421 (bond_mesocnt.cpp, angle_mesocnt.cpp). */
export const A_CC = 1.421;

/** Energy unit per eV for the unit style; names the style when it is not one the mesocnt styles support. */
export const eunitOf = (style: string, what: string): number => {
  const e = EUNIT[style];
  if (e === undefined) throw new StyleError(`${what} does not support ${style} units`);
  return e;
};

/** Radius (in Angstrom) of an (n,m) tube: sqrt(3 (n^2 + nm + m^2)) A_CC / 2 pi (the preset r_ang). */
export const chiralRadius = (n: number, m: number): number => Math.sqrt(3.0 * (n * n + n * m + m * m)) * A_CC / (2 * Math.PI);
