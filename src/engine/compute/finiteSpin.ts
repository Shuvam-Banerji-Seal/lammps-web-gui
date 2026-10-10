import type { SimState } from '../types';
import { isEllipsoid, massOf } from '../atoms';
import { ellipsoidInertia, quatToMatrix } from './asphere';

/*
 * Spin (intrinsic) angular momentum and own moment of inertia of finite-size
 * particles, shared by compute angmom/chunk, inertia/chunk and the variable
 * group functions angmom() and inertia(). docs.lammps.org/compute_angmom_chunk.html:
 * "sphere and line particles via their angular velocity" and "while ellipsoid,
 * superellipsoid, triangle, and body particles add their stored per-particle
 * angular momentum." The spin of a sphere is L = I omega (textbook; the page
 * typesets its formula as math).
 * docs.lammps.org/compute_inertia_chunk.html: "the moment of inertia of the
 * particle's own shape is now added to each chunk via the parallel-axis
 * theorem."
 *
 * Measured with native LAMMPS (black box; spheres of diameter 1 and 2 with
 * density 1 and 0.5, and ellipsoids with half-axes (0.5, 0.75, 1) and
 * (0.4, 0.4, 0.8) in several orientations, with spin set by a Velocities
 * section): the summed angmom(), inertia() and angmom/chunk, inertia/chunk
 * values equal the point-mass sums plus the spin and own-shape terms below,
 * and point particles (radius 0 spheres, ellipsoidflag 0 atoms) add nothing.
 * Textbook physics: a solid sphere of mass m and radius r has I = 2/5 m r^2 in
 * every direction; a solid ellipsoid with half-axes a, b, c has body-frame
 * moments m/5 (b^2+c^2), m/5 (a^2+c^2), m/5 (a^2+b^2), rotated into the lab
 * frame by R(q) I_body R(q)^T.
 */

/** Spin angular momentum and own inertia tensor (inertia/chunk ordering xx,yy,zz,xy,yz,xz) of a finite-size particle. */
export interface FiniteSpin {
  L: [number, number, number];
  T: [number, number, number, number, number, number];
}

/** Spin terms of atom i, or null for a point particle. Sphere spin needs omega; ellipsoid spin is the stored angmom. */
export const finiteSpin = (s: SimState, i: number): FiniteSpin | null => {
  if (isEllipsoid(s, i) && s.quat && s.angmom) {
    const q = s.quat;
    const R = quatToMatrix(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]);
    const [Ia, Ib, Ic] = ellipsoidInertia(s, i);
    // Lab-frame tensor R diag(Ia, Ib, Ic) R^T, entries from the row-major rotation matrix.
    const d = [Ia, Ib, Ic];
    const T = [0, 0, 0, 0, 0, 0] as FiniteSpin['T'];
    // [row, column, inertia/chunk slot]: xx, yy, zz, xy, yz, xz
    const slots: [number, number, number][] = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 3], [1, 2, 4], [0, 2, 5]];
    for (const [a, b, slot] of slots) {
      let t = 0;
      for (let k = 0; k < 3; k++) t += R[3 * a + k] * d[k] * R[3 * b + k];
      T[slot] = t;
    }
    return { L: [s.angmom[3 * i], s.angmom[3 * i + 1], s.angmom[3 * i + 2]], T };
  }
  if (s.radius && s.omega && s.radius[i] > 0) {
    const r = s.radius[i];
    const I = 0.4 * massOf(s, i) * r * r;
    const w = [s.omega[3 * i], s.omega[3 * i + 1], s.omega[3 * i + 2]];
    return { L: [I * w[0], I * w[1], I * w[2]], T: [I, I, I, 0, 0, 0] };
  }
  return null;
};
