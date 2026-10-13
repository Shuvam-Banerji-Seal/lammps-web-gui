/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the DIFFRACTION package's compute
 * styles compute_xrd.cpp and compute_saed.cpp (with the atomic scattering factor tables compute_xrd_consts.h and
 * compute_saed_consts.h), commit b332f28; the fix saed/vtk (fix_saed_vtk.cpp) is only registered to refuse. Contributing
 * authors: Shawn Coleman and Douglas Spearot (Arkansas). Copyright Sandia Corporation (GNU General Public License). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_diffraction; the names listed there must match
 * these keys). fix saed/vtk is listed in src/engine/refusedStyles.ts.
 */
import type { ComputeFactory, FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import { ComputeXRD } from '../diffraction/compute_xrd';
import { ComputeSAED } from '../diffraction/compute_saed';

/** fix styles: refused. */
export const FIXES: Record<string, FixFactory> = {
  'saed/vtk': () => {
    throw new StyleError('fix saed/vtk is not supported: it writes VTK files of the compute saed intensities (fix_saed_vtk.cpp is not ported)');
  },
};

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  xrd: (sys, id, group, args) => new ComputeXRD(sys, id, group, args),
  saed: (sys, id, group, args) => new ComputeSAED(sys, id, group, args),
};
