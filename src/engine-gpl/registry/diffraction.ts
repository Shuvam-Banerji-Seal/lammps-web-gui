/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the DIFFRACTION package's compute
 * styles compute_xrd.cpp and compute_saed.cpp (with the atomic scattering factor tables compute_xrd_consts.h and
 * compute_saed_consts.h) and the fix saed/vtk (fix_saed_vtk.cpp), commit b332f28. Contributing authors: Shawn Coleman and
 * Douglas Spearot (Arkansas); fix saed/vtk: Pieter in 't Veld (SNL). Copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_diffraction; the names listed there must match
 * these keys).
 */
import type { ComputeFactory, FixFactory } from '../../engine/styles';
import { ComputeXRD } from '../diffraction/compute_xrd';
import { ComputeSAED } from '../diffraction/compute_saed';
import { FixSAEDVTK } from '../diffraction/fix_saed_vtk';

/** fix styles. */
export const FIXES: Record<string, FixFactory> = {
  'saed/vtk': (sys, id, group, args) => new FixSAEDVTK(sys, id, group, args),
};

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  xrd: (sys, id, group, args) => new ComputeXRD(sys, id, group, args),
  saed: (sys, id, group, args) => new ComputeSAED(sys, id, group, args),
};
