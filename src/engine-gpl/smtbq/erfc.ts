/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SMTBQ/pair_smtbq.cpp (PairSMTBQ::tabqeq, which calls the C library erfc; this file replaces that
 * library call) at commit b332f28. Authors credited in the C++ file headers: N. Salles, O. Politano, E. Amzallag,
 * R. Tetot (SMTBQ). Copyright (2015) Universite de Bourgogne and Sandia Corporation (LAMMPS). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * erfc(x) for the SMTBQ tables: the series of erf for x < 2 (the alternating series has no cancellation problem
 * there) and the continued fraction for erfc at x >= 2 (Abramowitz and Stegun 7.1.14): direct evaluation, accurate
 * to about 1e-15 absolute; no table.
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** erfc(x) for real x. */
export function erfc(x: number): number {
  if (Number.isNaN(x)) return x;
  if (x < 0) return 2 - erfc(-x);
  if (x < 2) {
    // erf(x) = 2/sqrt(pi) sum_n (-1)^n x^(2n+1) / (n! (2n+1))
    let term = x;
    let sum = x;
    const x2 = x * x;
    for (let n = 1; n < 400; n++) {
      term *= -x2 / n;
      const t = term / (2 * n + 1);
      sum += t;
      if (Math.abs(t) < 1e-18 * Math.abs(sum)) break;
    }
    return 1 - TWO_OVER_SQRTPI * sum;
  }
  // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
  let f = x;
  for (let k = 80; k >= 1; k--) f = x + (k / 2) / f;  // 80 levels: relative change below 1e-16 against 3000 levels for x >= 2
  return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
}
