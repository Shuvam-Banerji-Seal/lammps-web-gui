/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the helpers of the pair styles in
 * src/EXTRA-PAIR (pair_lj_cut_sphere.cpp, pair_lj_expand_sphere.cpp, pair_lj_expand_coul_long.cpp,
 * pair_lj_gromacs_coul_gromacs.cpp, pair_lj_pirani.cpp, pair_morse_smooth_linear.cpp, pair_nm_cut*.cpp) and of
 * src/math_special.h (MathSpecial::powint) and src/pair.cpp (Pair::mix_energy, Pair::mix_distance).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2;
 * this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import type { MixRule } from '../../engine/force/types';
import { StyleError } from '../../engine/force/types';
import { typeBounds } from '../../engine/force/types';
import { pow as libmPow } from '../libm/pow';

/** math_special.h MathSpecial::powint: repeated squaring, as the native loop does it. */
export const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

/** Pair::mix_energy (pair.cpp at b332f28): sixthpower gives 0 when a diameter is 0. */
export const mixEnergyNative = (rule: MixRule, eps1: number, eps2: number, sig1: number, sig2: number): number => {
  if (rule === 'geometric' || rule === 'arithmetic') return Math.sqrt(eps1 * eps2);
  if (sig1 !== 0.0 && sig2 !== 0.0) {
    return (2.0 * Math.sqrt(eps1 * eps2) * powint(sig1, 3) * powint(sig2, 3) / (powint(sig1, 6) + powint(sig2, 6)));
  }
  return 0.0;
};

/** Pair::mix_distance (pair.cpp at b332f28); the sixthpower root is the C library pow. */
export const mixDistanceNative = (rule: MixRule, sig1: number, sig2: number): number => {
  if (rule === 'geometric') return Math.sqrt(sig1 * sig2);
  if (rule === 'arithmetic') return 0.5 * (sig1 + sig2);
  return libmPow(0.5 * (powint(sig1, 6) + powint(sig2, 6)), 1.0 / 6.0);
};

/**
 * utils::bounds for a pair_coeff I J pair, then the loop of the native coeff(): every I <= J in the ranges is
 * visited with j from max(jlo, i); a range with no such pair is "Incorrect args for pair coefficients".
 */
export const forEachTypePair = (iw: string, jw: string, ntypes: number, fn: (i: number, j: number) => void): void => {
  const [ilo, ihi] = typeBounds(iw, ntypes);
  const [jlo, jhi] = typeBounds(jw, ntypes);
  let count = 0;
  for (let i = ilo; i <= ihi; i++) {
    for (let j = Math.max(jlo, i); j <= jhi; j++) {
      fn(i, j);
      count++;
    }
  }
  if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
};

/** The radius of an owned or ghost atom (ghosts carry the radius of their owner). */
export const radiusOfIndex = (radius: Float64Array, owner: Int32Array, k: number): number => radius[owner[k]];
