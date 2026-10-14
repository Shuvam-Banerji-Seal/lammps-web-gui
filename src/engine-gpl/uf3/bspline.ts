/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ML-UF3/uf3_bspline_basis2.cpp, src/ML-UF3/uf3_bspline_basis2.h, src/ML-UF3/uf3_bspline_basis3.cpp and
 * src/ML-UF3/uf3_bspline_basis3.h (the constructors: the cubic B-spline coefficients of one basis function on its knot
 * span, and the quadratic derivative form). Authors credited in the C++ headers: Ajinkya Hire, Hendrik Krass,
 * Matthias Rupp and Richard Hennig. Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * The constant expressions are the source's, transliterated mechanically (square -> sq, cube -> cb, the same
 * operation order), so the values are the native ones bit for bit. knots[off + n] is the C++ knots[n] of a pointer
 * offset by off.
 */

const sq = (x: number): number => x * x;
const cb = (x: number): number => x * x * x;

/** uf3_bspline_basis2 constructor: 9 constants of the quadratic (derivative) basis on knots[off..off+3]. */
export function uf3Basis2Constants(knots: ArrayLike<number>, off: number, coefficient: number): Float64Array {
  const K = new Float64Array(9);
  let c0 = 0, c1 = 0, c2 = 0;
  c0 = coefficient
    * (sq(knots[off + 0])
    / (sq(knots[off + 0]) - knots[off + 0] * knots[off + 1] - knots[off + 0] * knots[off + 2] + knots[off + 1] * knots[off + 2]));
  c1 = coefficient *
    (-2 * knots[off + 0] /
    (sq(knots[off + 0]) - knots[off + 0] * knots[off + 1] - knots[off + 0] * knots[off + 2] + knots[off + 1] * knots[off + 2]));
  c2 = coefficient *
    (1 / (sq(knots[off + 0]) - knots[off + 0] * knots[off + 1] - knots[off + 0] * knots[off + 2] + knots[off + 1] * knots[off + 2]));
  K[0] = c0;
  K[1] = c1;
  K[2] = c2;
  c0 = coefficient *
    (-knots[off + 1] * knots[off + 3] /
    (sq(knots[off + 1]) - knots[off + 1] * knots[off + 2] - knots[off + 1] * knots[off + 3] + knots[off + 2] * knots[off + 3]) -
    knots[off + 0] * knots[off + 2] /
    (knots[off + 0] * knots[off + 1] - knots[off + 0] * knots[off + 2] - knots[off + 1] * knots[off + 2] + sq(knots[off + 2])));
  c1 = coefficient *
    (knots[off + 1] /
    (sq(knots[off + 1]) - knots[off + 1] * knots[off + 2] - knots[off + 1] * knots[off + 3] + knots[off + 2] * knots[off + 3]) +
    knots[off + 3] /
    (sq(knots[off + 1]) - knots[off + 1] * knots[off + 2] - knots[off + 1] * knots[off + 3] + knots[off + 2] * knots[off + 3]) +
    knots[off + 0] /
    (knots[off + 0] * knots[off + 1] - knots[off + 0] * knots[off + 2] - knots[off + 1] * knots[off + 2] + sq(knots[off + 2])) +
    knots[off + 2] /
    (knots[off + 0] * knots[off + 1] - knots[off + 0] * knots[off + 2] - knots[off + 1] * knots[off + 2] + sq(knots[off + 2])));
  c2 = coefficient *
    (-1 / (sq(knots[off + 1]) - knots[off + 1] * knots[off + 2] - knots[off + 1] * knots[off + 3] + knots[off + 2] * knots[off + 3]) -
    1 / (knots[off + 0] * knots[off + 1] - knots[off + 0] * knots[off + 2] - knots[off + 1] * knots[off + 2] + sq(knots[off + 2])));
  K[3] = c0;
  K[4] = c1;
  K[5] = c2;
  c0 = coefficient *
    (sq(knots[off + 3]) /
    (knots[off + 1] * knots[off + 2] - knots[off + 1] * knots[off + 3] - knots[off + 2] * knots[off + 3] + sq(knots[off + 3])));
  c1 = coefficient *
    (-2 * knots[off + 3] /
    (knots[off + 1] * knots[off + 2] - knots[off + 1] * knots[off + 3] - knots[off + 2] * knots[off + 3] + sq(knots[off + 3])));
  c2 = coefficient *
    (1 / (knots[off + 1] * knots[off + 2] - knots[off + 1] * knots[off + 3] - knots[off + 2] * knots[off + 3] + sq(knots[off + 3])));
  K[6] = c0;
  K[7] = c1;
  K[8] = c2;
  return K;
}

/** uf3_bspline_basis3 constructor: 16 constants of the cubic basis on knots[off..off+4]. */
export function uf3Basis3Constants(knots: ArrayLike<number>, off: number, coefficient: number): Float64Array {
  const K = new Float64Array(16);
  let c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  c0 = coefficient *
    (-cb(knots[off + 0]) /
    (-cb(knots[off + 0]) + sq(knots[off + 0]) * knots[off + 1] + sq(knots[off + 0]) * knots[off + 2] +
    sq(knots[off + 0]) * knots[off + 3] - knots[off + 0] * knots[off + 1] * knots[off + 2] -
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3]));
  c1 = coefficient *
    (3 * sq(knots[off + 0]) /
    (-cb(knots[off + 0]) + sq(knots[off + 0]) * knots[off + 1] + sq(knots[off + 0]) * knots[off + 2] +
    sq(knots[off + 0]) * knots[off + 3] - knots[off + 0] * knots[off + 1] * knots[off + 2] -
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3]));
  c2 = coefficient *
    (-3 * knots[off + 0] /
    (-cb(knots[off + 0]) + sq(knots[off + 0]) * knots[off + 1] + sq(knots[off + 0]) * knots[off + 2] +
    sq(knots[off + 0]) * knots[off + 3] - knots[off + 0] * knots[off + 1] * knots[off + 2] -
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3]));
  c3 = coefficient *
    (1 /
    (-cb(knots[off + 0]) + sq(knots[off + 0]) * knots[off + 1] + sq(knots[off + 0]) * knots[off + 2] +
    sq(knots[off + 0]) * knots[off + 3] - knots[off + 0] * knots[off + 1] * knots[off + 2] -
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3]));
  K[0] = c0;
  K[1] = c1;
  K[2] = c2;
  K[3] = c3;
  c0 = coefficient *
    (sq(knots[off + 1]) * knots[off + 4] /
    (-cb(knots[off + 1]) + sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 4] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    knots[off + 2] * knots[off + 3] * knots[off + 4]) +
    sq(knots[off + 0]) * knots[off + 2] /
    (-sq(knots[off + 0]) * knots[off + 1] + sq(knots[off + 0]) * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] -
    knots[off + 0] * sq(knots[off + 2]) - knots[off + 0] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 3] + sq(knots[off + 2]) * knots[off + 3]) +
    knots[off + 0] * knots[off + 1] * knots[off + 3] /
    (-knots[off + 0] * sq(knots[off + 1]) + knots[off + 0] * knots[off + 1] * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 3] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * sq(knots[off + 3]) + knots[off + 2] * sq(knots[off + 3])));
  c1 = coefficient *
    (-sq(knots[off + 1]) /
    (-cb(knots[off + 1]) + sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 4] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    knots[off + 2] * knots[off + 3] * knots[off + 4]) -
    2 * knots[off + 1] * knots[off + 4] /
    (-cb(knots[off + 1]) + sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 4] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    knots[off + 2] * knots[off + 3] * knots[off + 4]) -
    sq(knots[off + 0]) /
    (-sq(knots[off + 0]) * knots[off + 1] + sq(knots[off + 0]) * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] -
    knots[off + 0] * sq(knots[off + 2]) - knots[off + 0] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 3] + sq(knots[off + 2]) * knots[off + 3]) -
    2 * knots[off + 0] * knots[off + 2] /
    (-sq(knots[off + 0]) * knots[off + 1] + sq(knots[off + 0]) * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] -
    knots[off + 0] * sq(knots[off + 2]) - knots[off + 0] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 3] + sq(knots[off + 2]) * knots[off + 3]) -
    knots[off + 0] * knots[off + 1] /
    (-knots[off + 0] * sq(knots[off + 1]) + knots[off + 0] * knots[off + 1] * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 3] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * sq(knots[off + 3]) + knots[off + 2] * sq(knots[off + 3])) -
    knots[off + 0] * knots[off + 3] /
    (-knots[off + 0] * sq(knots[off + 1]) + knots[off + 0] * knots[off + 1] * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 3] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * sq(knots[off + 3]) + knots[off + 2] * sq(knots[off + 3])) -
    knots[off + 1] * knots[off + 3] /
    (-knots[off + 0] * sq(knots[off + 1]) + knots[off + 0] * knots[off + 1] * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 3] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * sq(knots[off + 3]) + knots[off + 2] * sq(knots[off + 3])));
  c2 = coefficient *
    (2 * knots[off + 1] /
    (-cb(knots[off + 1]) + sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 4] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    knots[off + 2] * knots[off + 3] * knots[off + 4]) +
    knots[off + 4] /
    (-cb(knots[off + 1]) + sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 4] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    knots[off + 2] * knots[off + 3] * knots[off + 4]) +
    2 * knots[off + 0] /
    (-sq(knots[off + 0]) * knots[off + 1] + sq(knots[off + 0]) * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] -
    knots[off + 0] * sq(knots[off + 2]) - knots[off + 0] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 3] + sq(knots[off + 2]) * knots[off + 3]) +
    knots[off + 2] /
    (-sq(knots[off + 0]) * knots[off + 1] + sq(knots[off + 0]) * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] -
    knots[off + 0] * sq(knots[off + 2]) - knots[off + 0] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 3] + sq(knots[off + 2]) * knots[off + 3]) +
    knots[off + 0] /
    (-knots[off + 0] * sq(knots[off + 1]) + knots[off + 0] * knots[off + 1] * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 3] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * sq(knots[off + 3]) + knots[off + 2] * sq(knots[off + 3])) +
    knots[off + 1] /
    (-knots[off + 0] * sq(knots[off + 1]) + knots[off + 0] * knots[off + 1] * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 3] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * sq(knots[off + 3]) + knots[off + 2] * sq(knots[off + 3])) +
    knots[off + 3] /
    (-knots[off + 0] * sq(knots[off + 1]) + knots[off + 0] * knots[off + 1] * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 3] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * sq(knots[off + 3]) + knots[off + 2] * sq(knots[off + 3])));
  c3 = coefficient *
    (-1 /
    (-cb(knots[off + 1]) + sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 4] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    knots[off + 2] * knots[off + 3] * knots[off + 4]) -
    1 /
    (-sq(knots[off + 0]) * knots[off + 1] + sq(knots[off + 0]) * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] -
    knots[off + 0] * sq(knots[off + 2]) - knots[off + 0] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * knots[off + 2] * knots[off + 3] + sq(knots[off + 2]) * knots[off + 3]) -
    1 /
    (-knots[off + 0] * sq(knots[off + 1]) + knots[off + 0] * knots[off + 1] * knots[off + 2] +
    knots[off + 0] * knots[off + 1] * knots[off + 3] - knots[off + 0] * knots[off + 2] * knots[off + 3] +
    sq(knots[off + 1]) * knots[off + 3] - knots[off + 1] * knots[off + 2] * knots[off + 3] -
    knots[off + 1] * sq(knots[off + 3]) + knots[off + 2] * sq(knots[off + 3])));
  K[4] = c0;
  K[5] = c1;
  K[6] = c2;
  K[7] = c3;
  c0 = coefficient *
    (-knots[off + 0] * sq(knots[off + 3]) /
    (-knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] +
    knots[off + 0] * knots[off + 2] * knots[off + 3] - knots[off + 0] * sq(knots[off + 3]) +
    knots[off + 1] * knots[off + 2] * knots[off + 3] - knots[off + 1] * sq(knots[off + 3]) -
    knots[off + 2] * sq(knots[off + 3]) + cb(knots[off + 3])) -
    knots[off + 1] * knots[off + 3] * knots[off + 4] /
    (-sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] -
    knots[off + 1] * sq(knots[off + 3]) - knots[off + 1] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * knots[off + 3] * knots[off + 4] + sq(knots[off + 3]) * knots[off + 4]) -
    knots[off + 2] * sq(knots[off + 4]) /
    (-knots[off + 1] * sq(knots[off + 2]) + knots[off + 1] * knots[off + 2] * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    sq(knots[off + 2]) * knots[off + 4] - knots[off + 2] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * sq(knots[off + 4]) + knots[off + 3] * sq(knots[off + 4])));
  c1 = coefficient *
    (2 * knots[off + 0] * knots[off + 3] /
    (-knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] +
    knots[off + 0] * knots[off + 2] * knots[off + 3] - knots[off + 0] * sq(knots[off + 3]) +
    knots[off + 1] * knots[off + 2] * knots[off + 3] - knots[off + 1] * sq(knots[off + 3]) -
    knots[off + 2] * sq(knots[off + 3]) + cb(knots[off + 3])) +
    sq(knots[off + 3]) /
    (-knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] +
    knots[off + 0] * knots[off + 2] * knots[off + 3] - knots[off + 0] * sq(knots[off + 3]) +
    knots[off + 1] * knots[off + 2] * knots[off + 3] - knots[off + 1] * sq(knots[off + 3]) -
    knots[off + 2] * sq(knots[off + 3]) + cb(knots[off + 3])) +
    knots[off + 1] * knots[off + 3] /
    (-sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] -
    knots[off + 1] * sq(knots[off + 3]) - knots[off + 1] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * knots[off + 3] * knots[off + 4] + sq(knots[off + 3]) * knots[off + 4]) +
    knots[off + 1] * knots[off + 4] /
    (-sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] -
    knots[off + 1] * sq(knots[off + 3]) - knots[off + 1] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * knots[off + 3] * knots[off + 4] + sq(knots[off + 3]) * knots[off + 4]) +
    knots[off + 3] * knots[off + 4] /
    (-sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] -
    knots[off + 1] * sq(knots[off + 3]) - knots[off + 1] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * knots[off + 3] * knots[off + 4] + sq(knots[off + 3]) * knots[off + 4]) +
    2 * knots[off + 2] * knots[off + 4] /
    (-knots[off + 1] * sq(knots[off + 2]) + knots[off + 1] * knots[off + 2] * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    sq(knots[off + 2]) * knots[off + 4] - knots[off + 2] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * sq(knots[off + 4]) + knots[off + 3] * sq(knots[off + 4])) +
    sq(knots[off + 4]) /
    (-knots[off + 1] * sq(knots[off + 2]) + knots[off + 1] * knots[off + 2] * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    sq(knots[off + 2]) * knots[off + 4] - knots[off + 2] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * sq(knots[off + 4]) + knots[off + 3] * sq(knots[off + 4])));
  c2 = coefficient *
    (-knots[off + 0] /
    (-knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] +
    knots[off + 0] * knots[off + 2] * knots[off + 3] - knots[off + 0] * sq(knots[off + 3]) +
    knots[off + 1] * knots[off + 2] * knots[off + 3] - knots[off + 1] * sq(knots[off + 3]) -
    knots[off + 2] * sq(knots[off + 3]) + cb(knots[off + 3])) -
    2 * knots[off + 3] /
    (-knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] +
    knots[off + 0] * knots[off + 2] * knots[off + 3] - knots[off + 0] * sq(knots[off + 3]) +
    knots[off + 1] * knots[off + 2] * knots[off + 3] - knots[off + 1] * sq(knots[off + 3]) -
    knots[off + 2] * sq(knots[off + 3]) + cb(knots[off + 3])) -
    knots[off + 1] /
    (-sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] -
    knots[off + 1] * sq(knots[off + 3]) - knots[off + 1] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * knots[off + 3] * knots[off + 4] + sq(knots[off + 3]) * knots[off + 4]) -
    knots[off + 3] /
    (-sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] -
    knots[off + 1] * sq(knots[off + 3]) - knots[off + 1] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * knots[off + 3] * knots[off + 4] + sq(knots[off + 3]) * knots[off + 4]) -
    knots[off + 4] /
    (-sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] -
    knots[off + 1] * sq(knots[off + 3]) - knots[off + 1] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * knots[off + 3] * knots[off + 4] + sq(knots[off + 3]) * knots[off + 4]) -
    knots[off + 2] /
    (-knots[off + 1] * sq(knots[off + 2]) + knots[off + 1] * knots[off + 2] * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    sq(knots[off + 2]) * knots[off + 4] - knots[off + 2] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * sq(knots[off + 4]) + knots[off + 3] * sq(knots[off + 4])) -
    2 * knots[off + 4] /
    (-knots[off + 1] * sq(knots[off + 2]) + knots[off + 1] * knots[off + 2] * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    sq(knots[off + 2]) * knots[off + 4] - knots[off + 2] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * sq(knots[off + 4]) + knots[off + 3] * sq(knots[off + 4])));
  c3 = coefficient *
    (1 /
    (-knots[off + 0] * knots[off + 1] * knots[off + 2] + knots[off + 0] * knots[off + 1] * knots[off + 3] +
    knots[off + 0] * knots[off + 2] * knots[off + 3] - knots[off + 0] * sq(knots[off + 3]) +
    knots[off + 1] * knots[off + 2] * knots[off + 3] - knots[off + 1] * sq(knots[off + 3]) -
    knots[off + 2] * sq(knots[off + 3]) + cb(knots[off + 3])) +
    1 /
    (-sq(knots[off + 1]) * knots[off + 2] + sq(knots[off + 1]) * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] -
    knots[off + 1] * sq(knots[off + 3]) - knots[off + 1] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * knots[off + 3] * knots[off + 4] + sq(knots[off + 3]) * knots[off + 4]) +
    1 /
    (-knots[off + 1] * sq(knots[off + 2]) + knots[off + 1] * knots[off + 2] * knots[off + 3] +
    knots[off + 1] * knots[off + 2] * knots[off + 4] - knots[off + 1] * knots[off + 3] * knots[off + 4] +
    sq(knots[off + 2]) * knots[off + 4] - knots[off + 2] * knots[off + 3] * knots[off + 4] -
    knots[off + 2] * sq(knots[off + 4]) + knots[off + 3] * sq(knots[off + 4])));
  K[8] = c0;
  K[9] = c1;
  K[10] = c2;
  K[11] = c3;
  c0 = coefficient *
    (cb(knots[off + 4]) /
    (-knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] +
    knots[off + 1] * knots[off + 3] * knots[off + 4] - knots[off + 1] * sq(knots[off + 4]) +
    knots[off + 2] * knots[off + 3] * knots[off + 4] - knots[off + 2] * sq(knots[off + 4]) - knots[off + 3] * sq(knots[off + 4]) +
    cb(knots[off + 4])));
  c1 = coefficient *
    (-3 * sq(knots[off + 4]) /
    (-knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] +
    knots[off + 1] * knots[off + 3] * knots[off + 4] - knots[off + 1] * sq(knots[off + 4]) +
    knots[off + 2] * knots[off + 3] * knots[off + 4] - knots[off + 2] * sq(knots[off + 4]) - knots[off + 3] * sq(knots[off + 4]) +
    cb(knots[off + 4])));
  c2 = coefficient *
    (3 * knots[off + 4] /
    (-knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] +
    knots[off + 1] * knots[off + 3] * knots[off + 4] - knots[off + 1] * sq(knots[off + 4]) +
    knots[off + 2] * knots[off + 3] * knots[off + 4] - knots[off + 2] * sq(knots[off + 4]) - knots[off + 3] * sq(knots[off + 4]) +
    cb(knots[off + 4])));
  c3 = coefficient *
    (-1 /
    (-knots[off + 1] * knots[off + 2] * knots[off + 3] + knots[off + 1] * knots[off + 2] * knots[off + 4] +
    knots[off + 1] * knots[off + 3] * knots[off + 4] - knots[off + 1] * sq(knots[off + 4]) +
    knots[off + 2] * knots[off + 3] * knots[off + 4] - knots[off + 2] * sq(knots[off + 4]) - knots[off + 3] * sq(knots[off + 4]) +
    cb(knots[off + 4])));
  K[12] = c0;
  K[13] = c1;
  K[14] = c2;
  K[15] = c3;
  return K;
}
