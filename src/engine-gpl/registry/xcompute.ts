/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the EXTRA-COMPUTE styles
 * compute_ackland_atom.cpp, compute_ave_sphere_atom.cpp, compute_composition_atom.cpp, compute_efield_wolf_atom.cpp,
 * compute_rattlers_atom.cpp, compute_temp_rotate.cpp and compute_ti.cpp, and the pressure/cylinder style, which LAMMPS
 * registers as a second name of the stress/cylinder class (compute_stress_cylinder.h: ComputeStyle(pressure/cylinder,
 * ComputeStressCylinder)). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_xcompute; the names listed there must match these
 * keys).
 */
import type { ComputeFactory } from '../../engine/styles';
import { ComputeAcklandAtom } from '../xcompute/ackland_atom';
import { ComputeAveSphereAtom } from '../xcompute/ave_sphere_atom';
import { ComputeCompositionAtom } from '../xcompute/composition_atom';
import { ComputeEfieldWolfAtom } from '../xcompute/efield_wolf_atom';
import { ComputeRattlersAtom } from '../xcompute/rattlers_atom';
import { ComputeTempRotate } from '../xcompute/temp_rotate';
import { ComputeTI } from '../xcompute/ti';
import { ComputeStressCylinder } from '../compute/stress_curvilinear';

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'ackland/atom': (sys, id, group, args) => new ComputeAcklandAtom(sys, id, group, args),
  'ave/sphere/atom': (sys, id, group, args) => new ComputeAveSphereAtom(sys, id, group, args),
  'composition/atom': (sys, id, group, args) => new ComputeCompositionAtom(sys, id, group, args),
  'efield/wolf/atom': (sys, id, group, args) => new ComputeEfieldWolfAtom(sys, id, group, args),
  'pressure/cylinder': (sys, id, group, args) => new ComputeStressCylinder(sys, id, group, args),
  'rattlers/atom': (sys, id, group, args) => new ComputeRattlersAtom(sys, id, group, args),
  'temp/rotate': (sys, id, group, args) => new ComputeTempRotate(sys, id, group, args),
  ti: (sys, id, group, args) => new ComputeTI(sys, id, group, args),
};
