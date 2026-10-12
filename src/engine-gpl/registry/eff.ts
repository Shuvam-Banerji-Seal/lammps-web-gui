/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the EFF package's style registry
 * (src/EFF/pair_eff_cut.cpp, fix_nve_eff.cpp, compute_temp_eff.cpp, compute_ke_eff.cpp). Stage 1: pair eff/cut,
 * fix nve/eff, compute temp/eff and ke/eff. Original author (from the sources): Andres Jaramillo-Botero (Caltech).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (the names of this module are listed there).
 */
import type { Pair } from '../../engine/force/types';
import type { Fix } from '../../engine/fix/fix';
import type { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { PairEffCut } from '../eff/pair_eff_cut';
import { FixNVEEff } from '../eff/fix_nve_eff';
import { ComputeTempEff } from '../eff/compute_temp_eff';
import { ComputeKEEff } from '../eff/compute_ke_eff';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'eff/cut': () => new PairEffCut(),
};

/** fix styles. */
export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'nve/eff': (sys, id, group, args) => new FixNVEEff(sys, id, group, args),
};

/** compute styles. */
export const COMPUTES: Record<string, (sys: System, id: string, group: string, args: string[]) => Compute> = {
  'temp/eff': (sys, id, group, args) => new ComputeTempEff(sys, id, group, args),
  'ke/eff': (sys, id, group, args) => new ComputeKEEff(sys, id, group, args),
};
