/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MGPT/mgpt_splinetab.cpp and src/MGPT/mgpt_splinetab.h at commit b332f28 (the MGPT package).
 * Fast MGPT algorithm: Tomas Oppelstrup (LLNL, 2015), from the matrix MGPT v4.4 FORTRAN routine of John Moriarty
 * (LLNL, 2006). Copyright (2015) Lawrence Livermore National Security, LLC, LLNL-CODE-674031, distributed under the
 * GNU General Public License version 2; the LLNL notice is kept in NOTICE.llnl next to this file.
 * This file is GPL-2.0 too (see src/engine-gpl/LICENSE).
 *
 * Piecewise cubic splines on an equidistant grid: makespline builds one cubic per interval (a C^2 function, second
 * derivatives at the ends from a second-order one-sided formula), evalspline evaluates the value and the first and
 * second derivatives.
 */

/** mgpt_splinetab.cpp trisolve: the three-band system A (n rows of 3) solved in place into y. */
const trisolve = (n: number, A: Float64Array, y: Float64Array): void => {
  // Backward elimination
  for (let i = n - 1; i > 0; i--) {
    const q = A[3 * (i - 1) + 2] / A[3 * i + 1];
    A[3 * (i - 1) + 1] = A[3 * (i - 1) + 1] - q * A[3 * i];
    y[i - 1] = y[i - 1] - q * y[i];
  }
  // Forward substitution
  y[0] = y[0] / A[1];
  for (let i = 1; i < n; i++) y[i] = (y[i] - A[3 * i] * y[i - 1]) / A[3 * i + 1];
};

/**
 * mgpt_splinetab.cpp makespline: the coefficients of the ntab-1 interpolating cubics of the table tab (point i at
 * tab[off + stride*i]) into C (4 per interval: value, then the coefficients of t, t^2, t^3 for t in [0, 1)).
 */
export const makespline = (ntab: number, stride: number, tab: Float64Array, off: number, C: Float64Array): void => {
  const n = 3 * (ntab - 1);
  const A = new Float64Array(3 * n);
  const y = new Float64Array(n);
  const t = (i: number): number => tab[off + stride * i];

  // Second order second derivative approximation at end points.
  const hLeft = 2.0 * t(0) - 5.0 * t(1) + 4.0 * t(2) - 1.0 * t(3);
  const hRight = 2.0 * t(ntab - 1) - 5.0 * t(ntab - 2) + 4.0 * t(ntab - 3) - 1.0 * t(ntab - 4);

  A[0] = 0.0; A[1] = 0.0; A[2] = 2.0; y[0] = hLeft;
  let j = 0;
  for (let i = 1; i < ntab - 1; i++) {
    j = 3 * (i - 1);
    const d = t(i) - t(i - 1);
    A[3 * (j + 1)] = 1.0; A[3 * (j + 1) + 1] = 1.0; A[3 * (j + 1) + 2] = 1.0; y[j + 1] = d;
    A[3 * (j + 2)] = 1.0; A[3 * (j + 2) + 1] = 2.0; A[3 * (j + 2) + 2] = -1.0; y[j + 2] = -d;
    A[3 * (j + 3)] = 2.0; A[3 * (j + 3) + 1] = 2.0; A[3 * (j + 3) + 2] = -2.0; y[j + 3] = 2.0 * d;
  }

  j = 3 * (ntab - 2);
  const d = t(ntab - 1) - t(ntab - 2);
  A[3 * (j + 1)] = 1.0; A[3 * (j + 1) + 1] = 1.0; A[3 * (j + 1) + 2] = 1.0; y[j + 1] = d;
  A[3 * (j + 2)] = 2.0; A[3 * (j + 2) + 1] = 6.0; A[3 * (j + 2) + 2] = 0.0; y[j + 2] = hRight;

  trisolve(n, A, y);

  for (let i = 0; i < ntab - 1; i++) {
    C[4 * i] = t(i);
    C[4 * i + 1] = y[3 * i];
    C[4 * i + 2] = y[3 * i + 1];
    C[4 * i + 3] = y[3 * i + 2];
  }
};

/** The value, first and second derivative of a spline (evalspline's y, dy, d2y). */
export interface SplineValue {
  y: number;
  dy: number;
  d2y: number;
}

/**
 * mgpt_splinetab.cpp evalspline: the spline of C (n intervals over [x0, x1], as made by makespline from n+1 points)
 * at x, written into out. Outside [x0, x1] the first or last cubic is extended.
 */
export const evalspline = (n: number, x0: number, x1: number, C: Float64Array, x: number, out: SplineValue): void => {
  const dxinv = n / (x1 - x0);
  let xhat = (x - x0) / (x1 - x0) * n;

  // C++ (int) truncation toward zero
  let idx = Math.trunc(xhat);
  if (idx < 0) idx = 0;
  if (idx > n - 1) idx = n - 1;
  xhat = xhat - idx;
  const p = 4 * idx;

  const t1 = C[p + 2] + xhat * C[p + 3];
  const t2 = C[p + 1] + xhat * t1;

  const t3 = t1 + xhat * C[p + 3];

  out.y = C[p] + xhat * t2;
  out.dy = (t2 + xhat * t3) * dxinv;
  out.d2y = 2.0 * (t3 + xhat * C[p + 3]) * (dxinv * dxinv);
};
