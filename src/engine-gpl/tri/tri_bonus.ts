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
 * Browser-engine difference: the Jacobi step is the engine's symmetric eigen solver (src/engine/bodies.ts
 * eigenSym3), sorted as jacobi3 sorts. Only the eigenvectors and eigenvalues are taken from it.
 */
import { eigenSym3, rotationToQuat } from '../../engine/bodies';
import { StyleError } from '../../engine/force/types';
import type { TriBody, TriBonusHook, V3 } from '../../engine/triBonus';

/** MathExtra::inertia_triangle: the inertia tensor (Voigt order xx yy zz yz xz xy) of a triangle of the given mass, from its corners. */
const inertiaTriangle = (v0: V3, v1: V3, v2: V3, mass: number): number[] => {
  const v = [v0, v1, v2];
  // S = [[2,1,1],[1,2,1],[1,1,2]]; sv = S v; vtsv = v^T sv (rows are the corners)
  const S = [[2, 1, 1], [1, 2, 1], [1, 1, 2]];
  const vtsv = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let a = 0; a < 3; a++) {
    for (let b = 0; b < 3; b++) {
      let acc = 0;
      for (let k = 0; k < 3; k++) for (let l = 0; l < 3; l++) acc += v[k][a] * S[k][l] * v[l][b];
      vtsv[a][b] = acc;
    }
  }
  const sum = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2] + v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]
    + v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2];
  const vvv = [v0[0] + v1[0] + v2[0], v0[1] + v1[1] + v2[1], v0[2] + v1[2] + v2[2]];
  const total = sum + vvv[0] * vvv[0] + vvv[1] * vvv[1] + vvv[2] * vvv[2];
  const e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]], e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
  const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
  const a = Math.hypot(n[0], n[1], n[2]);
  const inv24 = mass / 24.0;
  return [
    inv24 * a * (total - vtsv[0][0]),
    inv24 * a * (total - vtsv[1][1]),
    inv24 * a * (total - vtsv[2][2]),
    -inv24 * a * vtsv[1][2],
    -inv24 * a * vtsv[0][2],
    -inv24 * a * vtsv[0][1],
  ];
};

/** The triangle body frame of data_atom_bonus, from the corner points (absolute coordinates) and the density. */
const triBody = (c: [V3, V3, V3], density: number): TriBody => {
  const [c1, c2, c3] = c;
  // atom_vec_tri.cpp data_atom_bonus refuses two equal corner points as an invalid shape
  for (const [p, q] of [[c1, c2], [c1, c3], [c2, c3]] as const) {
    if (p[0] === q[0] && p[1] === q[1] && p[2] === q[2]) throw new StyleError('Invalid shape for triangle in Triangles section of data file (two corner points are equal)');
  }
  const c2mc1 = [c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]], c3mc1 = [c3[0] - c1[0], c3[1] - c1[1], c3[2] - c1[2]];
  const norm = [c2mc1[1] * c3mc1[2] - c2mc1[2] * c3mc1[1], c2mc1[2] * c3mc1[0] - c2mc1[0] * c3mc1[2], c2mc1[0] * c3mc1[1] - c2mc1[1] * c3mc1[0]];
  const area = 0.5 * Math.hypot(norm[0], norm[1], norm[2]);
  // rmass = density * area (the mass the inertia uses)
  const mass = density * area;
  const centroid: V3 = [(c1[0] + c2[0] + c3[0]) / 3.0, (c1[1] + c2[1] + c3[1]) / 3.0, (c1[2] + c2[2] + c3[2]) / 3.0];
  // radius: the largest distance from the centroid to a corner
  let r2 = 0;
  for (const p of c) r2 = Math.max(r2, (p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2 + (p[2] - centroid[2]) ** 2);
  const radius = Math.sqrt(r2);
  // inertia_triangle with the absolute corners, then jacobi3 (decreasing eigenvalues)
  const I6 = inertiaTriangle(c1, c2, c3, mass);
  const M = [I6[0], I6[5], I6[4], I6[5], I6[1], I6[3], I6[4], I6[3], I6[2]];
  const eig = eigenSym3(M);
  const order = [0, 1, 2].sort((p, q) => eig.values[q] - eig.values[p]);
  const moments: V3 = [eig.values[order[0]], eig.values[order[1]], eig.values[order[2]]];
  // axes: column k of the eigenvector matrix, k in sort order
  const axis = (k: number): V3 => [eig.vectors[order[k]], eig.vectors[3 + order[k]], eig.vectors[6 + order[k]]];
  const ex = axis(0), ey = axis(1);
  let ez = axis(2);
  // right-handed: flip the third axis if (ex x ey) . ez < 0
  const cr = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
  if (cr[0] * ez[0] + cr[1] * ez[1] + cr[2] * ez[2] < 0) ez = [-ez[0], -ez[1], -ez[2]];
  const R = [ex[0], ey[0], ez[0], ex[1], ey[1], ez[1], ex[2], ey[2], ez[2]];
  const quat = rotationToQuat(R);
  // corner displacements from the centroid in the principal basis: (ex . d, ey . d, ez . d)
  const corners = new Float64Array(9);
  for (let k = 0; k < 3; k++) {
    const d = [c[k][0] - centroid[0], c[k][1] - centroid[1], c[k][2] - centroid[2]];
    for (let b = 0; b < 3; b++) {
      const e = b === 0 ? ex : b === 1 ? ey : ez;
      corners[3 * k + b] = e[0] * d[0] + e[1] * d[1] + e[2] * d[2];
    }
  }
  return { area, mass, centroid, radius, quat, moments, corners };
};

/** The hook object for the lazy HOOKS entry 'tri' (contract in src/engine/triBonus.ts). */
export const TRI_BODY_HOOK: TriBonusHook = { triBody };
