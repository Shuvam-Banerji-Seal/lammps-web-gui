import { StyleError } from '../types';
import type { Geometry } from '../../domain';

/*
 * Massless M site of the TIP4P water models (the sites of pair styles
 * lj/cut/tip4p/cut, lj/cut/tip4p/long, tip4p/cut, tip4p/long and kspace
 * style pppm/tip4p share this geometry).
 *
 * docs.lammps.org/pair_lj_cut_tip4p.rst: "The M site location is used for
 * all Coulomb interactions instead of the oxygen atom location, also" (the
 * sentence goes on "with all other atom types, while the location of the
 * oxygen atom is used for the Lennard-Jones interactions."). Howto_tip4p.rst:
 * "This site M is located at a fixed distance away from" the oxygen "along
 * the bisector of the HOH angle" (the HOH bisector at qdist from O, the
 * Howto table's "OM distance"); "The forces on M are then" "projected on the
 * oxygen and the two hydrogen atoms."
 * pair_lj_cut_tip4p.rst: "For each TIP4P water molecule in your system, the
 * atom IDs for" the O atom and its H atoms "must be consecutive, with the O
 * atom first."
 *
 * The position and the projection weights below are not written in the docs;
 * they were measured with native LAMMPS (black box, scratch inputs):
 *   M = O + (alpha/2) (d1 + d2),  d1 = H1 - O, d2 = H2 - O (minimum image),
 *   alpha = qdist / (r0 cos(theta0/2)),
 * with r0 the equilibrium OH bond length of btype and theta0 the equilibrium
 * HOH angle of atype (from bond_coeff and angle_coeff, not from the current
 * geometry). The force on M is split as f_O += (1 - alpha) f_M and
 * f_H1 = f_H2 += (alpha/2) f_M. Measured with native LAMMPS (black box): for
 * a distorted water (|d1| = 1.012, |d2| = 0.867 A, theta 80 degrees) the
 * distance |M - O| was 0.16766 A (= (alpha/2)|d1 + d2|, not qdist), and the
 * split fractions were 0.743977 (O) and 0.128012 (each H) for alpha = 0.256023.
 * With bond r0 = 1.0 and angle 110 degrees, qdist = 0.15 gave alpha 0.261517,
 * with r0 = 0.9572, angle 104.52 and qdist = 0.20 alpha 0.341366, and with
 * r0 = 1.2, angle 90, qdist = 0.10 alpha 0.117851: all equal the formula to 1e-12.
 */

export interface Tip4pModel {
  otype: number;
  htype: number;
  /** Projection weight: O gets (1 - alpha), each H gets alpha / 2 of the M force. */
  alpha: number;
}

/** alpha = qdist / (r0 cos(theta0 / 2)); theta0 in degrees. */
export const tip4pAlpha = (qdist: number, r0: number, thetaDeg: number): number =>
  qdist / (r0 * Math.cos((thetaDeg * Math.PI) / 360));

/** Sites of the owned atoms: M positions (3 per atom; the atom's own position for non-O atoms) and the H owned indices (-1 for non-O). */
export interface OwnedSites {
  M: Float64Array;
  h1: Int32Array;
  h2: Int32Array;
}

/**
 * Builds the M sites of the owned atoms. The H atoms are the atoms with IDs
 * O+1 and O+2 (pair_lj_cut_tip4p.rst, see above); a water without them is an error.
 */
export const buildOwnedSites = (
  n: number, x: Float64Array, type: Int32Array, ids: Int32Array, geom: Geometry, model: Tip4pModel,
): OwnedSites => {
  const M = new Float64Array(3 * n);
  const h1 = new Int32Array(n).fill(-1);
  const h2 = new Int32Array(n).fill(-1);
  M.set(x.subarray(0, 3 * n));
  if (n === 0) return { M, h1, h2 };
  const byId = new Map<number, number>();
  for (let i = 0; i < n; i++) byId.set(ids[i], i);
  const d1 = [0, 0, 0], d2 = [0, 0, 0];
  for (let o = 0; o < n; o++) {
    if (type[o] !== model.otype) continue;
    const a = byId.get(ids[o] + 1), b = byId.get(ids[o] + 2);
    if (a === undefined || b === undefined || type[a] !== model.htype || type[b] !== model.htype) {
      throw new StyleError(`TIP4P oxygen atom ${ids[o]} needs H atoms with IDs ${ids[o] + 1} and ${ids[o] + 2} of type ${model.htype}`);
    }
    h1[o] = a; h2[o] = b;
    for (let k = 0; k < 3; k++) {
      d1[k] = x[3 * a + k] - x[3 * o + k];
      d2[k] = x[3 * b + k] - x[3 * o + k];
    }
    geom.minimumImage(d1);
    geom.minimumImage(d2);
    for (let k = 0; k < 3; k++) M[3 * o + k] = x[3 * o + k] + 0.5 * model.alpha * (d1[k] + d2[k]);
  }
  return { M, h1, h2 };
};
