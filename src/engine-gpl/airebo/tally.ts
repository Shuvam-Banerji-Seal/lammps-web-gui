/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/pair.cpp (Pair::ev_tally with newton on, Pair::v_tally2, Pair::v_tally3, Pair::v_tally4 and
 * Pair::ev_tally4, their per-atom parts), at commit b332f28; the calls are the ones of
 * src/MANYBODY/pair_airebo.cpp. The global virial of AIREBO comes from the force field's sum of x.f
 * (virial_fdotr_compute), so only the per-atom virial (compute stress/atom) is tallied here.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */

const add = (va: Float64Array, a: number, w: number, v0: number, v1: number, v2: number, v3: number, v4: number, v5: number): void => {
  const o = 6 * a;
  va[o] += w * v0; va[o + 1] += w * v1; va[o + 2] += w * v2;
  va[o + 3] += w * v3; va[o + 4] += w * v4; va[o + 5] += w * v5;
};

/** Pair::v_tally2 (and the virial of Pair::ev_tally): d (x) d fpair, half to i and half to j. */
export const vTally2 = (va: Float64Array, i: number, j: number, fpair: number, dx: number, dy: number, dz: number): void => {
  const v0 = dx * dx * fpair, v1 = dy * dy * fpair, v2 = dz * dz * fpair;
  const v3 = dx * dy * fpair, v4 = dx * dz * fpair, v5 = dy * dz * fpair;
  add(va, i, 0.5, v0, v1, v2, v3, v4, v5);
  add(va, j, 0.5, v0, v1, v2, v3, v4, v5);
};

/** Pair::v_tally3: drik (x) fi + drjk (x) fj, a third to each of i, j and k. */
export const vTally3 = (
  va: Float64Array, i: number, j: number, k: number,
  fi: ArrayLike<number>, fj: ArrayLike<number>, drik: ArrayLike<number>, drjk: ArrayLike<number>,
): void => {
  const v0 = drik[0] * fi[0] + drjk[0] * fj[0];
  const v1 = drik[1] * fi[1] + drjk[1] * fj[1];
  const v2 = drik[2] * fi[2] + drjk[2] * fj[2];
  const v3 = drik[0] * fi[1] + drjk[0] * fj[1];
  const v4 = drik[0] * fi[2] + drjk[0] * fj[2];
  const v5 = drik[1] * fi[2] + drjk[1] * fj[2];
  const third = 1.0 / 3.0;
  add(va, i, third, v0, v1, v2, v3, v4, v5);
  add(va, j, third, v0, v1, v2, v3, v4, v5);
  add(va, k, third, v0, v1, v2, v3, v4, v5);
};

/** Pair::v_tally4 (and the virial of Pair::ev_tally4): drim (x) fi + drjm (x) fj + drkm (x) fk, a quarter to each atom. */
export const vTally4 = (
  va: Float64Array, i: number, j: number, k: number, m: number,
  fi: ArrayLike<number>, fj: ArrayLike<number>, fk: ArrayLike<number>,
  drim: ArrayLike<number>, drjm: ArrayLike<number>, drkm: ArrayLike<number>,
): void => {
  const v0 = drim[0] * fi[0] + drjm[0] * fj[0] + drkm[0] * fk[0];
  const v1 = drim[1] * fi[1] + drjm[1] * fj[1] + drkm[1] * fk[1];
  const v2 = drim[2] * fi[2] + drjm[2] * fj[2] + drkm[2] * fk[2];
  const v3 = drim[0] * fi[1] + drjm[0] * fj[1] + drkm[0] * fk[1];
  const v4 = drim[0] * fi[2] + drjm[0] * fj[2] + drkm[0] * fk[2];
  const v5 = drim[1] * fi[2] + drjm[1] * fj[2] + drkm[1] * fk[2];
  add(va, i, 0.25, v0, v1, v2, v3, v4, v5);
  add(va, j, 0.25, v0, v1, v2, v3, v4, v5);
  add(va, k, 0.25, v0, v1, v2, v3, v4, v5);
  add(va, m, 0.25, v0, v1, v2, v3, v4, v5);
};
