/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the SPH package's shared definitions, from
 * src/SPH/pair_sph_taitwater.cpp, pair_sph_rhosum.cpp, fix_sph.cpp and the kernel constants of those files. LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE). Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer
 * EMI, Ernst Mach Institute).
 *
 * Model (docs.lammps.org/pair_sph_rhosum.html, pair_sph_taitwater.html): each particle carries a density rho, an
 * internal energy esph, a heat capacity cv and a velocity estimate vest; the rates drho and desph are summed by the
 * pair styles and integrated by fix sph / fix sph/stationary. Neighbour values of ghost atoms are read from their
 * owners, which is what LAMMPS' forward communication of rho, esph and vest does.
 */
import { StyleError } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import { isSphStyle, nativeOrder, SPH_CV, SPH_DESPH, SPH_DRHO, SPH_ESPH, SPH_RHO, SPH_VEST } from '../../engine/atoms';

/** Per-atom SPH arrays of a state (sph props of atom_style sph). */
export interface SphArrays {
  rho: Float64Array;
  drho: Float64Array;
  esph: Float64Array;
  desph: Float64Array;
  cv: Float64Array;
  vest: Float64Array;
}

/** The SPH arrays of the state; a StyleError naming `what` when the atom style has none. */
export const sphArrays = (s: SimState, what: string): SphArrays => {
  const get = (name: string): Float64Array => {
    const c = s.custom.get(name);
    if (!c || !isSphStyle(s.atomStyle)) throw new StyleError(`${what} requires atom style sph`);
    return c.data;
  };
  return {
    rho: get(SPH_RHO), drho: get(SPH_DRHO), esph: get(SPH_ESPH), desph: get(SPH_DESPH), cv: get(SPH_CV), vest: get(SPH_VEST),
  };
};

/**
 * Zeroes the rates of change of the owned atoms (AtomVecSPH::force_clear). The SPH fixes do it before every force
 * evaluation, so the pair styles of a step accumulate into clean sums.
 */
export const clearSphRates = (s: SimState): void => {
  const a = sphArrays(s, 'Fix sph');
  a.drho.fill(0, 0, s.n);
  a.desph.fill(0, 0, s.n);
};

/**
 * Kernel derivative prefactors of the Lucy-type weight used by the Tait, ideal gas and heat conduction styles
 * (the source's comments name them; the quadric kernel is used by the density summation). The derivative wfd of the
 * weight is lacking a factor of r, which the force expressions recover (see pair_sph_taitwater.cpp).
 */
export const LUCY_DERIV_3D = -25.066903536973515383;
export const LUCY_DERIV_2D = -19.098593171027440292;
/** Quadric kernel prefactors of pair sph/rhosum (3d, 2d). */
export const QUADRIC_3D = 2.1541870227086614782;
export const QUADRIC_2D = 1.5915494309189533576;

/** Native LAMMPS position of every atom: rank[index] = k when native storage order keeps the atom at position k. */
export const nativeRank = (s: SimState): Int32Array => {
  const order = nativeOrder(s);
  const rank = new Int32Array(s.n);
  for (let k = 0; k < s.n; k++) rank[order[k]] = k;
  return rank;
};

/**
 * Whether the ghost at (gx, gy, gz) lies "above and to the right" of the owned atom at (x, y, z): the z, then y, then x
 * order of npair_half_bin_newton (LAMMPS), which keeps an owned-ghost pair in the owned atom's half list.
 */
export const aboveOf = (x: number, y: number, z: number, gx: number, gy: number, gz: number): boolean => {
  if (gz !== z) return gz > z;
  if (gy !== y) return gy > y;
  return gx >= x;
};
