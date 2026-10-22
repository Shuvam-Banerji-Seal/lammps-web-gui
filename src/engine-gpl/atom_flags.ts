/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): which atom styles set the per-atom attribute
 * flags atom->esph_flag, rho_flag, cv_flag and vest_flag (the AtomVec constructors of src/SPH/atom_vec_sph.cpp,
 * src/DPD-REACT/atom_vec_dpd.cpp, src/DPD-MESO/atom_vec_edpd.cpp, atom_vec_mdpd.cpp, atom_vec_tdpd.cpp,
 * src/RHEO/atom_vec_rheo.cpp, atom_vec_rheo_thermal.cpp, src/MACHDYN/atom_vec_smd.cpp). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * Style constructors test these flags (pair sph/*, pair multi/lucy); a hybrid atom style has the flags of every
 * sub-style, since each sub-style's AtomVec constructor sets them on the same Atom.
 */
import { atomSubStyles } from '../engine/atoms';
import type { AtomStyle } from '../engine/types';

export type AtomFlag = 'esph' | 'rho' | 'cv' | 'vest';

const FLAG_STYLES: Record<AtomFlag, readonly string[]> = {
  esph: ['sph', 'smd', 'rheo/thermal'],
  rho: ['sph', 'dpd', 'mdpd', 'rheo', 'rheo/thermal'],
  cv: ['sph'],
  vest: ['sph', 'edpd', 'mdpd', 'tdpd', 'smd'],
};

/** True when the atom style sets atom->{flag}_flag. */
export const atomHasFlag = (st: AtomStyle, flag: AtomFlag): boolean => atomSubStyles(st).some((x) => FLAG_STYLES[flag].includes(x));
