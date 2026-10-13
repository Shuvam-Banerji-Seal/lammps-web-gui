/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registrations of the RHEO package (src/RHEO/fix_rheo.cpp, fix_rheo_viscosity.cpp,
 * fix_rheo_pressure.cpp, pair_rheo.cpp, compute_rheo_property_atom.cpp) at commit b332f28. Copyright Sandia Corporation
 * (LAMMPS development team, GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_rheo; the names listed there must match these keys).
 */
import type { FixFactory, ComputeFactory } from '../../engine/styles';
import type { Pair } from '../../engine/force/types';
import { FixRHEO } from '../rheo/fix_rheo';
import { FixRHEOThermal } from '../rheo/fix_rheo_thermal';
import { FixRHEOViscosity, FixRHEOPressure } from '../rheo/accessories';
import { PairRHEO } from '../rheo/pair_rheo';
import { PairRHEOSolid } from '../rheo/pair_rheo_solid';
import { ComputeRHEOPropertyAtom } from '../rheo/compute_rheo_property_atom';

export const FIXES: Record<string, FixFactory> = {
  rheo: (sys, id, group, args) => new FixRHEO(sys, id, group, args),
  'rheo/pressure': (sys, id, group, args) => new FixRHEOPressure(sys, id, group, args),
  'rheo/viscosity': (sys, id, group, args) => new FixRHEOViscosity(sys, id, group, args),
  'rheo/thermal': (sys, id, group, args) => new FixRHEOThermal(sys, id, group, args),
};

export const PAIRS: Record<string, () => Pair> = {
  rheo: () => new PairRHEO(),
  'rheo/solid': () => new PairRHEOSolid(),
};

export const COMPUTES: Record<string, ComputeFactory> = {
  'rheo/property/atom': (sys, id, group, args) => new ComputeRHEOPropertyAtom(sys, id, group, args),
};
