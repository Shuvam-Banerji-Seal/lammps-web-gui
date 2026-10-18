/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the spline and splint
 * routines of src/MOLECULE/bond_table.cpp, angle_table.cpp (BondTable::spline, splint; the same code in
 * both files), and the cyclic spline of src/MOLECULE/dihedral_table.cpp (solve_cyc_tridiag, taken from the
 * GNU Scientific Library tridiag.c by Gerard Jungman, cyc_spline, cyc_splint, cyc_lin), at commit b332f28.
 * Copyright (2003) Sandia Corporation (GNU General Public License); the spline code is "modified from Numerical
 * Recipes" (dihedral_table.cpp: "the cyclic tridiagonal matrix solver was borrowed from the tridiag.c written
 * by Gerard Jungman for GSL"; contributing author of dihedral_table.cpp: Andrew Jewett).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The operations are kept in the order of the C++ (the products and quotients are left to right as written).
 */
import { StyleError } from '../../engine/force/types';

const GSL_SUCCESS = 0;
const GSL_EZERODIV = 12;

/**
 * Non-periodic cubic spline through (x, y) with first derivatives yp1, ypn at the ends (values above 0.99e300 mean
 * a natural end); the second derivatives go to y2 (bond_table.cpp BondTable::spline).
 */
export const spline = (x: Float64Array, y: Float64Array, n: number, yp1: number, ypn: number, y2: Float64Array): void => {
  let p: number, qn: number, sig: number, un: number;
  const u = new Float64Array(n);
  if (yp1 > 0.99e300) {
    y2[0] = 0.0;
    u[0] = 0.0;
  } else {
    y2[0] = -0.5;
    u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
  }
  for (let i = 1; i < n - 1; i++) {
    sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    p = sig * y2[i - 1] + 2.0;
    y2[i] = (sig - 1.0) / p;
    u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
  }
  if (ypn > 0.99e300) {
    qn = 0.0;
    un = 0.0;
  } else {
    qn = 0.5;
    un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
  }
  y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
  for (let k = n - 2; k >= 0; k--) y2[k] = y2[k] * y2[k + 1] + u[k];
};

/** Value of the spline (xa, ya, y2a) at x (bond_table.cpp BondTable::splint). */
export const splint = (xa: Float64Array, ya: Float64Array, y2a: Float64Array, n: number, x: number): number => {
  let klo = 0;
  let khi = n - 1;
  while (khi - klo > 1) {
    const k = (khi + klo) >> 1;
    if (xa[k] > x) khi = k;
    else klo = k;
  }
  const h = xa[khi] - xa[klo];
  const a = (xa[khi] - x) / h;
  const b = (x - xa[klo]) / h;
  return a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6.0;
};

/**
 * Solves the cyclic tridiagonal system (diag, offdiag, b) of size N >= 3 (dihedral_table.cpp solve_cyc_tridiag,
 * with the strides of 1). Returns the solution, or null when a pivot is zero (the native then reports
 * "Matrix must be positive definite").
 */
const solveCycTridiag = (diag: Float64Array, offdiag: Float64Array, b: Float64Array, N: number): Float64Array | null => {
  const delta = new Float64Array(N), gamma = new Float64Array(N), alpha = new Float64Array(N);
  const c = new Float64Array(N), z = new Float64Array(N), x = new Float64Array(N);
  let status = GSL_SUCCESS;
  let sum = 0.0;
  // factor
  alpha[0] = diag[0];
  gamma[0] = offdiag[0] / alpha[0];
  delta[0] = offdiag[N - 1] / alpha[0];
  if (alpha[0] === 0) status = GSL_EZERODIV;
  for (let i = 1; i < N - 2; i++) {
    alpha[i] = diag[i] - offdiag[i - 1] * gamma[i - 1];
    gamma[i] = offdiag[i] / alpha[i];
    delta[i] = -delta[i - 1] * offdiag[i - 1] / alpha[i];
    if (alpha[i] === 0) status = GSL_EZERODIV;
  }
  for (let i = 0; i < N - 2; i++) sum += alpha[i] * delta[i] * delta[i];
  alpha[N - 2] = diag[N - 2] - offdiag[N - 3] * gamma[N - 3];
  gamma[N - 2] = (offdiag[N - 2] - offdiag[N - 3] * delta[N - 3]) / alpha[N - 2];
  alpha[N - 1] = diag[N - 1] - sum - alpha[N - 2] * gamma[N - 2] * gamma[N - 2];
  // update
  z[0] = b[0];
  for (let i = 1; i < N - 1; i++) z[i] = b[i] - z[i - 1] * gamma[i - 1];
  sum = 0.0;
  for (let i = 0; i < N - 2; i++) sum += delta[i] * z[i];
  z[N - 1] = b[N - 1] - sum - gamma[N - 2] * z[N - 2];
  for (let i = 0; i < N; i++) c[i] = z[i] / alpha[i];
  // backsubstitution
  x[N - 1] = c[N - 1];
  x[N - 2] = c[N - 2] - gamma[N - 2] * x[N - 1];
  for (let i = N - 3, j = 0; j <= N - 3; j++, i--) {
    x[i] = c[i] - gamma[i] * x[i + 1] - delta[i] * x[N - 1];
  }
  return status === GSL_SUCCESS ? x : null;
};

/**
 * Second derivatives of the periodic cubic spline through (xa, ya) with period `period` (dihedral_table.cpp
 * cyc_spline). N must be at least 3: the native code indexes one entry before the first for N = 2.
 */
export const cycSpline = (xa: Float64Array, ya: Float64Array, n: number, period: number): Float64Array => {
  if (n < 3) throw new StyleError(`cyclic spline needs at least 3 table entries, got ${n}`);
  const diag = new Float64Array(n), offdiag = new Float64Array(n), rhs = new Float64Array(n);
  let xaIm1: number, xaIp1: number;
  for (let i = 0; i < n; i++) {
    let im1 = i - 1;
    if (im1 < 0) {
      im1 += n;
      xaIm1 = xa[im1] - period;
    } else xaIm1 = xa[im1];
    let ip1 = i + 1;
    if (ip1 >= n) {
      ip1 -= n;
      xaIp1 = xa[ip1] + period;
    } else xaIp1 = xa[ip1];
    diag[i] = (xaIp1 - xaIm1) / 3.0;
    offdiag[i] = (xaIp1 - xa[i]) / 6.0;
    rhs[i] = ((ya[ip1] - ya[i]) / (xaIp1 - xa[i])) - ((ya[i] - ya[im1]) / (xa[i] - xaIm1));
  }
  const y2a = solveCycTridiag(diag, offdiag, rhs, n);
  if (y2a === null) throw new StyleError('Internal Cyclic Spline Error: Matrix must be positive definite.');
  return y2a;
};

/** The bracketing search of the cyclic routines: returns [klo, khi, xlo, xhi] with the periodic wrap applied. */
const cycBracket = (xa: Float64Array, n: number, period: number, x: number): [number, number, number, number] => {
  let klo = -1;
  let khi = n;
  let xlo = xa[n - 1] - period;
  let xhi = xa[0] + period;
  while (khi - klo > 1) {
    const k = (khi + klo) >> 1;
    if (xa[k] > x) {
      khi = k;
      xhi = xa[k];
    } else {
      klo = k;
      xlo = xa[k];
    }
  }
  if (khi === n) khi = 0;
  if (klo === -1) klo = n - 1;
  return [klo, khi, xlo, xhi];
};

/** Periodic cubic spline value at x (dihedral_table.cpp cyc_splint). */
export const cycSplint = (xa: Float64Array, ya: Float64Array, y2a: Float64Array, n: number, period: number, x: number): number => {
  const [klo, khi, xlo, xhi] = cycBracket(xa, n, period, x);
  const h = xhi - xlo;
  const a = (xhi - x) / h;
  const b = (x - xlo) / h;
  return a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6.0;
};

/** Periodic linear interpolation at x (dihedral_table.cpp cyc_lin). */
export const cycLin = (xa: Float64Array, ya: Float64Array, n: number, period: number, x: number): number => {
  const [klo, khi, xlo, xhi] = cycBracket(xa, n, period, x);
  const h = xhi - xlo;
  const a = (xhi - x) / h;
  const b = (x - xlo) / h;
  return a * ya[klo] + b * ya[khi];
};
