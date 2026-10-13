/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the style registry of src/DPD-SMOOTH/
 * pair_sdpd_taitwater_isothermal.cpp (pair sdpd/taitwater/isothermal) at commit b332f28. LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Contributing author (from the source): Morteza Jalalvand (IASBS).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_sdpd; the names listed there must match).
 */
import type { Pair } from '../../engine/force/types';
import { PairSDPDTaitwaterIsothermal } from '../sdpd/pair_sdpd_taitwater_isothermal';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'sdpd/taitwater/isothermal': () => new PairSDPDTaitwaterIsothermal(),
};
