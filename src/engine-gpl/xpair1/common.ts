/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the helpers of the pair styles in
 * src/EXTRA-PAIR (pair.cpp mix_energy and mix_distance, math_special.h powint, and the bounds and coefficient-store
 * steps every pair_<style>.cpp coeff() repeats). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { PairParams, StyleError, type MixRule } from '../../engine/force/types';
import { typeBounds } from '../../engine/force/types';
import { pow as libmPow } from '../libm/pow';

/** MathSpecial::powint (math_special.h): x^n for an integer n by repeated squaring; 0 for x == 0 (n != 0). */
export const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

/** Pair::mix_energy (pair.cpp, b332f28): sqrt(eps1 eps2) except the sixth-power rule. */
export const mixEnergy = (rule: MixRule, eps1: number, eps2: number, sig1: number, sig2: number): number => {
  if (rule === 'geometric' || rule === 'arithmetic') return Math.sqrt(eps1 * eps2);
  if (sig1 !== 0.0 && sig2 !== 0.0) {
    return (2.0 * Math.sqrt(eps1 * eps2) * powint(sig1, 3) * powint(sig2, 3) / (powint(sig1, 6) + powint(sig2, 6)));
  }
  return 0.0;
};

/** Pair::mix_distance (pair.cpp, b332f28). */
export const mixDistance = (rule: MixRule, sig1: number, sig2: number): number => {
  if (rule === 'geometric') return Math.sqrt(sig1 * sig2);
  if (rule === 'arithmetic') return 0.5 * (sig1 + sig2);
  return libmPow(0.5 * (powint(sig1, 6) + powint(sig2, 6)), 1.0 / 6.0);
};

/**
 * The coefficient store of a native coeff(): every I <= J of the two type ranges is set (native loops
 * "for (j = MAX(jlo,i); j <= jhi; j++)"), and a range that sets no pair is the error the caller names.
 */
export const setPairRange = (p: PairParams, iw: string, jw: string, values: readonly number[], emptyMsg: string, ntypes: number): void => {
  const [ilo, ihi] = typeBounds(iw, ntypes);
  const [jlo, jhi] = typeBounds(jw, ntypes);
  let count = 0;
  for (let i = ilo; i <= ihi; i++) {
    for (let j = Math.max(jlo, i); j <= jhi; j++) {
      p.set(i, j, values);
      count++;
    }
  }
  if (count === 0) throw new StyleError(emptyMsg);
};

/** Allocated pair-table size (ntypes+1)^2, as a zeroed table. */
export const table = (ntypes: number): Float64Array<ArrayBuffer> => new Float64Array((ntypes + 1) * (ntypes + 1));

/**
 * Settings reset of a native settings() command: every I <= J pair that a coeff() set takes the new global cutoff
 * (pair_beck.cpp, pair_born_gauss.cpp, ...). The column names the cutoff column of the store.
 */
export const resetSetCutoffs = (p: PairParams | undefined, column: string, cutGlobal: number, ntypes: number, j0: (i: number) => number = (i) => i): void => {
  if (!p) return;
  const c = p.p(column);
  for (let i = 1; i <= ntypes; i++) {
    for (let j = j0(i); j <= ntypes; j++) {
      if (p.isSet(i, j)) {
        c[p.idx(i, j)] = cutGlobal;
        c[p.idx(j, i)] = cutGlobal;
      }
    }
  }
};

/** Mirrors a per-pair value of an I <= J table into its J, I order (native "x[j][i] = x[i][j]" in init_one). */
export const mirror = (t: Float64Array, nt: number, i: number, j: number): void => { t[j * nt + i] = t[i * nt + j]; };
