/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ASPHERE/atom_vec_tri.cpp (AtomVecTri::data_atom_bonus, the body frame of a triangle read from the
 * Triangles section), with MathExtra::inertia_triangle from src/math_extra.cpp and MathEigen::jacobi3 from
 * src/math_eigen.cpp (decreasing sort, the default) at commit b332f28. The sources carry no contributing-author
 * line; copyright Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * The core reads a Triangles entry through the lazy HOOKS entry 'tri' (registry/tri_hook.ts, the contract in
 * src/engine/triBonus.ts). Steps, in the order of data_atom_bonus: the duplicate-corner check; the area and the
 * mass (density times area; read_data.html: the density "is used in conjunction with the particle volume to set
 * the mass"); the circumradius from the centroid; the inertia tensor of the triangle from inertia_triangle with
 * that mass; its principal moments and axes from the Jacobi diagonalization, right-handed with the third axis
 * flipped if needed; the quaternion of the axes; and the corner points as displacements from the centroid in
 * that basis.
 *
 * The Jacobi step is the jacobi3 port of src/engine-gpl/rigid/math_extra.ts (decreasing eigenvalues) and the
 * quaternion its exyz_to_q, so the frame matches the source to the last bit.
 */
import { jacobi3, exyzToQ } from '../rigid/math_extra';
import { StyleError } from '../../engine/force/types';
import type { TriBody, TriBonusHook, V3 } from '../../engine/triBonus';

/** MathExtra::lensq3 and len3, in the source's order of operations. */
const lensq3 = (v: readonly number[]): number => v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
const len3 = (v: readonly number[]): number => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
/** MathExtra::cross3. */
const cross3 = (a: readonly number[], b: readonly number[]): number[] =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/**
 * MathExtra::inertia_triangle: the inertia tensor (Voigt order xx yy zz yz xz xy) of a triangle of the given mass, from
 * its corners: sv = S v (times3), vtsv = v^T sv (transpose_times3), sum = lensq3 of each corner plus lensq3 of their
 * sum, and the area factor len3 of the cross product of the edges from the first corner.
 */
const inertiaTriangle = (v0: V3, v1: V3, v2: V3, mass: number): number[] => {
  const s = [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]];
  const v = [v0, v1, v2];
  const sv = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) sv[i][j] = s[i][0] * v[0][j] + s[i][1] * v[1][j] + s[i][2] * v[2][j];
  const vtsv = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) vtsv[i][j] = v[0][i] * sv[0][j] + v[1][i] * sv[1][j] + v[2][i] * sv[2][j];
  let sum = lensq3(v0) + lensq3(v1) + lensq3(v2);
  const vvv = [v0[0] + v1[0] + v2[0], v0[1] + v1[1] + v2[1], v0[2] + v1[2] + v2[2]];
  sum += lensq3(vvv);
  const v1mv0 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
  const v2mv0 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
  const a = len3(cross3(v1mv0, v2mv0));
  const inv24 = mass / 24.0;
  return [
    inv24 * a * (sum - vtsv[0][0]),
    inv24 * a * (sum - vtsv[1][1]),
    inv24 * a * (sum - vtsv[2][2]),
    -inv24 * a * vtsv[1][2],
    -inv24 * a * vtsv[0][2],
    -inv24 * a * vtsv[0][1],
  ];
};

/**
 * The triangle body frame of data_atom_bonus, from the corner points (absolute coordinates, already moved near the
 * atom by remap_near) and the density. `given` are the corner points as read, before remap_near: the source takes the
 * edge vectors c2 - c1 and c3 - c1 of the area from them ("MathExtra::sub3(c2, c1, c2mc1)" precedes the remap).
 */
const triBody = (c: [V3, V3, V3], density: number, given: [V3, V3, V3] = c): TriBody => {
  const [c1, c2, c3] = c;
  // atom_vec_tri.cpp data_atom_bonus refuses two equal corner points as an invalid shape
  for (const [p, q] of [[given[0], given[1]], [given[0], given[2]], [given[1], given[2]]] as const) {
    if (p[0] === q[0] && p[1] === q[1] && p[2] === q[2]) throw new StyleError('Invalid shape for triangle in Triangles section of data file (two corner points are equal)');
  }
  const c2mc1 = [given[1][0] - given[0][0], given[1][1] - given[0][1], given[1][2] - given[0][2]];
  const c3mc1 = [given[2][0] - given[0][0], given[2][1] - given[0][1], given[2][2] - given[0][2]];
  const centroid: V3 = [(c1[0] + c2[0] + c3[0]) / 3.0, (c1[1] + c2[1] + c3[1]) / 3.0, (c1[2] + c2[2] + c3[2]) / 3.0];
  // radius: the largest lensq3 of a corner from the centroid, then its square root
  let r2 = 0;
  for (const p of c) r2 = Math.max(r2, lensq3([p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]]));
  const radius = Math.sqrt(r2);
  // rmass = density * area, area = 0.5 len3(c2mc1 x c3mc1)
  const area = 0.5 * len3(cross3(c2mc1, c3mc1));
  const mass = density * area;
  // inertia_triangle with the (remapped) corners, then jacobi3 with its default decreasing sort
  const I6 = inertiaTriangle(c1, c2, c3, mass);
  const tensor = [[I6[0], I6[5], I6[4]], [I6[5], I6[1], I6[3]], [I6[4], I6[3], I6[2]]];
  const { eval: ev, evec } = jacobi3(tensor, true);
  const moments: V3 = [ev[0], ev[1], ev[2]];
  const ex = [evec[0][0], evec[1][0], evec[2][0]], ey = [evec[0][1], evec[1][1], evec[2][1]];
  let ez = [evec[0][2], evec[1][2], evec[2][2]];
  // right-handed: flip the third axis if (ex x ey) . ez < 0
  const norm = cross3(ex, ey);
  if (norm[0] * ez[0] + norm[1] * ez[1] + norm[2] * ez[2] < 0.0) ez = [-ez[0], -ez[1], -ez[2]];
  const q = exyzToQ(ex, ey, ez);
  const quat: [number, number, number, number] = [q[0], q[1], q[2], q[3]];
  // corner displacements from the centroid in the principal basis (transpose_matvec)
  const corners = new Float64Array(9);
  for (let k = 0; k < 3; k++) {
    const d = [c[k][0] - centroid[0], c[k][1] - centroid[1], c[k][2] - centroid[2]];
    corners[3 * k] = ex[0] * d[0] + ex[1] * d[1] + ex[2] * d[2];
    corners[3 * k + 1] = ey[0] * d[0] + ey[1] * d[1] + ey[2] * d[2];
    corners[3 * k + 2] = ez[0] * d[0] + ez[1] * d[1] + ez[2] * d[2];
  }
  return { area, mass, centroid, radius, quat, moments, corners, size: Math.max(len3(c2mc1), len3(c3mc1)) };
};

/** The hook object for the lazy HOOKS entry 'tri' (contract in src/engine/triBonus.ts). */
export const TRI_BODY_HOOK: TriBonusHook = { triBody };
