/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BODY/pair_body_rounded_polygon.cpp
 * (damping_friction, tangential_spring, history_friction), src/BODY/pair_body_rounded_polyhedron.cpp (the same three
 * routines, and friction_force with the contact history) and src/BODY/fix_wall_body_polygon.cpp /
 * fix_wall_body_polyhedron.cpp (the wall friction with the history). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Original authors (from the sources): Trung Dac Nguyen (ndactrung@gmail.com). Reference for the tangential spring:
 * Luding, Granular Matter 10, 235 (2008).
 *
 * docs.lammps.org/pair_body_rounded_polygon.html: "The history keyword ... the tangential displacement of the contact
 * is stored" (the keyword adds the contact history of the friction; without it the friction acts at one contact).
 *
 * Per pair of bodies: the contacts accumulate their elastic normal force, contact point, tangential velocity and normal
 * (ContactHistory); then one tangential spring acts on the pair, limited by mu times the sum of the normal forces, and
 * its tangential displacement (the shear) is kept for the next step by the caller, keyed by the pair of atom IDs.
 * The shear is a vector in space, stored as seen from the atom with the lower ID (the caller flips the sign).
 */
import type { PairCompute } from '../../engine/force/types';

export type V3 = [number, number, number];

/** The contact history of one pair of bodies for the current step (native Scratch: fnsum, pcsum, vtsum, nsum). */
export interface ContactHistory {
  fnsum: number;
  pcsum: V3;
  vtsum: V3;
  nsum: V3;
}

export const newContactHistory = (): ContactHistory => ({ fnsum: 0, pcsum: [0, 0, 0], vtsum: [0, 0, 0], nsum: [0, 0, 0] });

/**
 * Add one contact to the history (native damping_friction with the history: fne * pc, sign * fne * vt, sign * fne * n).
 * sign is +1 when the body is the first body of the pair (native shear_i), else -1.
 */
export const accumulateContact = (h: ContactHistory, sign: number, fne: number, pc: V3, vt: V3, n: V3): void => {
  h.fnsum += fne;
  for (let k = 0; k < 3; k++) {
    h.pcsum[k] += fne * pc[k];
    h.vtsum[k] += sign * fne * vt[k];
    h.nsum[k] += sign * fne * n[k];
  }
};

/** The result of the tangential spring of a pair: the friction force on the first body, its contact point, and the new shear. */
export interface HistoryFriction {
  ft: V3;
  pc: V3;
  shear: V3;
}

const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3): number => Math.sqrt(dot(a, a));

/**
 * history_friction + tangential_spring (native) for one pair. S is the shear of the pair as seen from the first body
 * (the body whose contacts are accumulated with sign +1). Returns null when the pair has no contact with a positive
 * normal force (native: the shear is then dropped). update: the tangential velocity advances the shear (native: not
 * in the setup of a run). The force -kt xi is limited to mu times the summed normal forces, and xi is reduced with it.
 */
export const historyFriction = (h: ContactHistory, S: V3, kt: number, mu: number, dt: number, update: boolean): HistoryFriction | null => {
  if (h.fnsum <= 0.0) return null;
  const pc: V3 = [h.pcsum[0] / h.fnsum, h.pcsum[1] / h.fnsum, h.pcsum[2] / h.fnsum];
  let vt: V3 = [h.vtsum[0] / h.fnsum, h.vtsum[1] / h.fnsum, h.vtsum[2] / h.fnsum];
  let n: V3 = [h.nsum[0], h.nsum[1], h.nsum[2]];
  // the tangential velocities of the contacts refer to their own normals
  const nmag = len(n);
  if (nmag > 0.0) {
    const s = 1.0 / nmag;
    n = [s * n[0], s * n[1], s * n[2]];
    const vtn = dot(vt, n);
    vt = [vt[0] - vtn * n[0], vt[1] - vtn * n[1], vt[2] - vtn * n[2]];
  }
  // xi is rotated into the tangent plane of n, keeping its magnitude
  let xi: V3 = [S[0], S[1], S[2]];
  const xin = dot(xi, n);
  const mag = len(xi);
  xi = [xi[0] - xin * n[0], xi[1] - xin * n[1], xi[2] - xin * n[2]];
  const magt = len(xi);
  if (magt > 0.0) {
    const s = mag / magt;
    xi = [s * xi[0], s * xi[1], s * xi[2]];
  }
  if (update) xi = [xi[0] + vt[0] * dt, xi[1] + vt[1] * dt, xi[2] + vt[2] * dt];
  let ft: V3 = [-kt * xi[0], -kt * xi[1], -kt * xi[2]];
  const ftmag = len(ft);
  const ftmax = mu * Math.max(h.fnsum, 0.0);
  if (ftmag > ftmax) {
    const s = ftmax / ftmag;
    ft = [s * ft[0], s * ft[1], s * ft[2]];
    xi = [s * xi[0], s * xi[1], s * xi[2]];
  }
  return { ft, pc, shear: xi };
};

/**
 * The per-atom part of the body pair styles' ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0, facc, del) with newton on
 * (pair_body_rounded_polygon.cpp and pair_body_rounded_polyhedron.cpp compute at b332f28): facc is the force of the pair
 * on body i and del = x_i - x_j; half of the energy and of del (x) facc go to each body.
 */
export const tallyBodyPair = (
  pc: PairCompute, i: number, j: number, e: number, facc: ArrayLike<number>, dx: number, dy: number, dz: number,
): void => {
  if (pc.eatom) { pc.eatom[i] += 0.5 * e; pc.eatom[j] += 0.5 * e; }
  const va = pc.vatom;
  if (!va) return;
  const v0 = 0.5 * dx * facc[0], v1 = 0.5 * dy * facc[1], v2 = 0.5 * dz * facc[2];
  const v3 = 0.5 * dx * facc[1], v4 = 0.5 * dx * facc[2], v5 = 0.5 * dy * facc[2];
  va[6 * i] += v0; va[6 * i + 1] += v1; va[6 * i + 2] += v2; va[6 * i + 3] += v3; va[6 * i + 4] += v4; va[6 * i + 5] += v5;
  va[6 * j] += v0; va[6 * j + 1] += v1; va[6 * j + 2] += v2; va[6 * j + 3] += v3; va[6 * j + 4] += v4; va[6 * j + 5] += v5;
};
