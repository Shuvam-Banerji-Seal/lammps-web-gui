/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BOCS/fix_bocs.cpp, fix_bocs.h, compute_pressure_bocs.cpp and compute_pressure_bocs.h at commit b332f28
 * (fix bocs and compute pressure/bocs; Nicholas J. H. Dunn and Michael R. DeLyser). Copyright Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * BOCS package styles; registered for the core through src/engine/lazyStyles.ts (group gpl_bocs). The classes are in
 * src/engine-gpl/bocs/.
 */
import type { ComputeFactory, FixFactory } from '../../engine/styles';
import { FixBocs } from '../bocs/fix_bocs';
import { ComputePressureBocs } from '../bocs/compute_pressure_bocs';

export const FIXES: Record<string, FixFactory> = {
  bocs: (sys, id, group, args) => new FixBocs(sys, id, group, args),
};

export const COMPUTES: Record<string, ComputeFactory> = {
  'pressure/bocs': (sys, id, group, args) => new ComputePressureBocs(sys, id, group, args),
};
