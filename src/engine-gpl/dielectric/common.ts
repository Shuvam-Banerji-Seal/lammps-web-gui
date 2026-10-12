/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/atom_vec_dielectric.cpp (the per-atom dielectric properties and the scaled charge), and the
 * unit conversion constants of src/DIELECTRIC/fix_polarize_bem_gmres.cpp and fix_polarize_bem_icc.cpp at commit
 * b332f28. DIELECTRIC package authors: Trung Nguyen and Monica Olvera de la Cruz (Northwestern University), and Honghao Li (Northwestern). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Shared helpers of the DIELECTRIC styles: the owned per-atom arrays of atom_style dielectric (core atoms.ts stores them
 * as internal custom props), the epsilon0 * E to charge conversion per unit style, and the field arrays a dielectric
 * pair or kspace style exposes to the polarization fixes.
 */
import { DIEL_AREA, DIEL_CURVATURE, DIEL_ED, DIEL_EM, DIEL_EPSILON, DIEL_QSCALED, isDielectricStyle } from '../../engine/atoms';
import { StyleError } from '../../engine/force/types';
import type { SimState } from '../../engine/types';

/** Owned per-atom arrays of atom_style dielectric (read them after each change: a sort or an append can reallocate). */
export interface DielArrays {
  /** Patch area (area/patch in the data file). */
  area: Float64Array;
  ed: Float64Array;
  em: Float64Array;
  epsilon: Float64Array;
  curvature: Float64Array;
  /** Scaled charge the dielectric styles use (q / epsilon for a point charge). */
  qs: Float64Array;
  /** Normal vectors (4 per atom: nx, ny, nz and |n|), the dipole layout of the mu column. */
  mu: Float64Array;
}

/** The dielectric arrays of the state; throws naming the style when the state has no dielectric atoms. */
export const dielArrays = (s: SimState, who: string): DielArrays => {
  if (!isDielectricStyle(s.atomStyle)) throw new StyleError(`${who} requires atom style dielectric (current: ${s.atomStyle})`);
  const g = (name: string): Float64Array => {
    const c = s.custom.get(name);
    if (!c) throw new StyleError(`${who}: atom style dielectric has no ${name} data`);
    return c.data;
  };
  if (!s.mu) throw new StyleError(`${who}: atom style dielectric has no normal vectors`);
  return {
    area: g(DIEL_AREA), ed: g(DIEL_ED), em: g(DIEL_EM), epsilon: g(DIEL_EPSILON), curvature: g(DIEL_CURVATURE),
    qs: g(DIEL_QSCALED), mu: s.mu,
  };
};

/**
 * Conversion of (epsilon0 * E) to a charge per area for the unit style (LAMMPS fix_polarize_bem_gmres.cpp, comments
 * on epsilon0e2q): real 0.000240263377163643 * 4 pi, metal 0.00553386738300813 * 4 pi, si 8.854187812813e-12 * 4 pi,
 * nano 0.000345866711328125 * 4 pi, lj 1. The constants are the SI values of epsilon0 converted to the unit style.
 */
export const epsilon0e2q = (unit: string): number => {
  const FOUR_PI = 4 * Math.PI;
  switch (unit) {
    case 'lj': return 1.0;
    case 'real': return 0.000240263377163643 * FOUR_PI;
    case 'metal': return 0.00553386738300813 * FOUR_PI;
    case 'si': return 8.854187812813e-12 * FOUR_PI;
    case 'nano': return 0.000345866711328125 * FOUR_PI;
    default: throw new StyleError("Only unit styles 'lj', 'real', 'metal', 'si' and 'nano' are supported");
  }
};

/**
 * The electric field array of a dielectric pair style (lj/cut/coul/cut/dielectric, lj/cut/coul/long/dielectric ...):
 * the style name ends in /dielectric and it stores efield; null for any other style.
 */
export const dielectricPairField = (p: { name: string; efield?: Float64Array } | null): Float64Array | null =>
  p && /\/dielectric$/.test(p.name) && p.efield ? p.efield : null;

/** The field arrays of a dielectric pair or kspace style: E per owned atom (3 values), the pair's and kspace's own. */
export interface DielectricField {
  /** Electric field at each owned atom from the pair term (3 per atom). */
  readonly efield: Float64Array;
}

/** Field at owned atom i from the pair term and the kspace term (when the kspace style exposes one). */
export const fieldAt = (pairField: Float64Array, kspaceField: Float64Array | null, i: number, out: Float64Array): void => {
  out[0] = pairField[3 * i]; out[1] = pairField[3 * i + 1]; out[2] = pairField[3 * i + 2];
  if (kspaceField) { out[0] += kspaceField[3 * i]; out[1] += kspaceField[3 * i + 1]; out[2] += kspaceField[3 * i + 2]; }
};
