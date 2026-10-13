/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/LATBOLTZ/latboltz_const.h at commit b332f28 (the D3Q15 velocities e15, weights w_lb15, normalizations
 * Ng_lb15 and moment matrix mg_lb15, and the wall bounce-back rows bbl for the orientations 3 and 16 used by the
 * flat walls). Copyright (2003) Sandia Corporation, LAMMPS development team (Frances Mackay, Santtu Ollila, Colin
 * Denniston, Tyson Whitehead). GNU General Public License. This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The D3Q19 tables and the pit (bbl rows other than 3 and 16, od) are not ported: the engine refuses them.
 */

/** MY_SQRT2 and MY_ISQRT2 (math_const.h). */
const MY_SQRT2 = 1.41421356237309504880;
const MY_ISQRT2 = 0.70710678118654752440;

/** 15-velocity lattice propagation vectors e15[i] = (x, y, z). */
export const E15: readonly (readonly [number, number, number])[] = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1],
  [-1, 1, 1], [-1, -1, 1], [1, -1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1],
];

/** 15-velocity weights. */
export const W15: readonly number[] = [
  2. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9.,
  1. / 72., 1. / 72., 1. / 72., 1. / 72., 1. / 72., 1. / 72., 1. / 72., 1. / 72.,
];

/** 15-velocity normalizations. */
export const NG15: readonly number[] = [
  1., 3., 3., 3., 9. / 2., 9. / 2., 9. / 2., 9., 9., 9., 27. / 2., 27. / 2., 27. / 2., 9., 1.,
];

/**
 * 15-velocity moment matrix, MG15[ii * 15 + l] = mg_lb15[ii][l] (row ii is a moment, column l a direction).
 */
export const MG15: Float64Array = Float64Array.from([
  1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
  0., 1., 0., -1., 0., 0., 0., 1., -1., -1., 1., 1., -1., -1., 1.,
  0., 0., 1., 0., -1., 0., 0., 1., 1., -1., -1., 1., 1., -1., -1.,
  0., 0., 0., 0., 0., 1., -1., 1., 1., 1., 1., -1., -1., -1., -1.,
  -1. / 3., 2. / 3., -1. / 3., 2. / 3., -1. / 3., -1. / 3., -1. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3.,
  -1. / 3., -1. / 3., 2. / 3., -1. / 3., 2. / 3., -1. / 3., -1. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3.,
  -1. / 3., -1. / 3., -1. / 3., -1. / 3., -1. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3.,
  0., 0., 0., 0., 0., 0., 0., 1., -1., 1., -1., 1., -1., 1., -1.,
  0., 0., 0., 0., 0., 0., 0., 1., 1., -1., -1., -1., -1., 1., 1.,
  0., 0., 0., 0., 0., 0., 0., 1., -1., -1., 1., -1., 1., 1., -1.,
  0., 0., -1. / 3., 0., 1. / 3., 0., 0., 2. / 3., 2. / 3., -2. / 3., -2. / 3., 2. / 3., 2. / 3., -2. / 3., -2. / 3.,
  0., 0., 0., 0., 0., -1. / 3., 1. / 3., 2. / 3., 2. / 3., 2. / 3., 2. / 3., -2. / 3., -2. / 3., -2. / 3., -2. / 3.,
  0., -1. / 3., 0., 1. / 3., 0., 0., 0., 2. / 3., -2. / 3., -2. / 3., 2. / 3., 2. / 3., -2. / 3., -2. / 3., 2. / 3.,
  0., 0., 0., 0., 0., 0., 0., 1., -1., 1., -1., -1., 1., -1., 1.,
  MY_SQRT2, -MY_ISQRT2, -MY_ISQRT2, -MY_ISQRT2, -MY_ISQRT2, -MY_ISQRT2, -MY_ISQRT2, MY_SQRT2, MY_SQRT2, MY_SQRT2, MY_SQRT2, MY_SQRT2, MY_SQRT2, MY_SQRT2, MY_SQRT2,
]);

/**
 * Wall bounce-back lists (latboltz_const.h, bbl): row [0] is the number of bounce-back directions, the next
 * entries the directions coming from inside the wall, a 0 separator, then the directions that propagate normally.
 * Only the flat walls are used: orientation 3 (bottom wall, +z normal) and 16 (top wall, -z normal).
 */
export const BBL_WALL_BOTTOM: readonly number[] = [5, 5, 7, 8, 9, 10, 0, 1, 2, 3, 4, 6, 11, 12, 13, 14];
export const BBL_WALL_TOP: readonly number[] = [5, 6, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10];
