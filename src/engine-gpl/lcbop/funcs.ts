/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_lcbop.h (f_c, f_c_LR), src/MANYBODY/pair_lcbop.cpp (g_decompose_x,
 * gSpline, hSpline, F_conj), at commit b332f28. Authors credited in the C++ file headers: Dominik Wojt
 * (Wroclaw University of Technology), based on pair_airebo by Ase Henry (MIT).
 * Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { exp } from '../libm/exp';
import { cos, sin } from '../libm/sincos';
import { pow } from '../libm/pow';
import type { LcbopParams, LcbopTables } from './params';

/** A double out-parameter (the C++ `double *dX`). */
export interface Ref { v: number }

/** MathConst::MY_PI */
export const MY_PI = 3.14159265358979323846;

/** The C++ MIN and MAX macros (the first operand is returned on a tie). */
export const cmin = (a: number, b: number): number => (a < b ? a : b);
export const cmax = (a: number, b: number): number => (a > b ? a : b);

/**
 * Short-range cutoff f_C (pair_lcbop.h f_c): returns the cutoff and writes its derivative to dX.
 * Like the C++, every branch sets dX.
 */
export const f_c = (p: LcbopParams, Xij: number, Xmin: number, Xmax: number, dX: Ref): number => {
  let cutoff: number;
  const t = (Xij - Xmin) / (Xmax - Xmin);
  if (t <= 0.0) {
    cutoff = 1.0;
    dX.v = 0.0;
  } else if (t >= 1.0) {
    cutoff = 0.0;
    dX.v = 0.0;
  } else {
    const z = t * t * t - 1;
    cutoff = exp(p.gamma_1 * t * t * t / z);
    dX.v = cutoff * (-3 * p.gamma_1 * t * t) / z / z / (Xmax - Xmin);
  }
  return cutoff;
};

/**
 * Long-range cutoff f_C_LR (pair_lcbop.h f_c_LR). Below Xmin the derivative is NOT written: the C++ says
 * "this way the derivative is inherited from previous cut off function call", so dX keeps the value of
 * the f_c call made on the same ref just before.
 */
export const f_c_LR = (Xij: number, Xmin: number, Xmax: number, dX: Ref): number => {
  let cutoff: number;
  const t = (Xij - Xmin) / (Xmax - Xmin);
  if (t <= 0.0) {
    cutoff = 1.0;
  } else if (t >= 1.0) {
    cutoff = 0.0;
    dX.v = 0.0;
  } else {
    cutoff = (1.0 + cos(MY_PI * t)) / 2.0;
    dX.v = -MY_PI * sin(MY_PI * t) / 2 / (Xmax - Xmin);
  }
  return cutoff;
};

/** PairLCBOP::gSpline: G(cos) and its derivative into dgdc. */
export const gSpline = (T: LcbopTables, x: number, dgdc: Ref): number => {
  // g_decompose_x
  let i = 0;
  while ((i < 6 - 1) && !(x < T.gX[i + 1])) i++;
  const x_n = x - T.gX[i];
  let sum = 0;
  dgdc.v = 0;
  let pow_x_n = 1.0;
  for (let j = 0; j < 5; j++) {
    sum += T.gC[j * 5 + i] * pow_x_n;
    dgdc.v += T.gC[(j + 1) * 5 + i] * (j + 1) * pow_x_n;
    pow_x_n *= x_n;
  }
  sum += T.gC[5 * 5 + i] * pow_x_n;
  return sum;
};

/** PairLCBOP::hSpline: H(delta) and its derivative into dhdx. */
export const hSpline = (p: LcbopParams, x: number, dhdx: Ref): number => {
  if (x < -p.d) {
    const z = p.kappa * (x + p.d);
    const y = pow(z, 10.0);
    const w = pow(1 + y, -0.1);
    dhdx.v = p.kappa * p.L * w / (1 + y);
    return p.L * (1 + z * w);
  }
  if (x > p.d) {
    dhdx.v = p.R_1;
    return p.R_0 + p.R_1 * (x - p.d);
  }
  let result = 1 + p.C_1 * x;
  dhdx.v = p.C_1 * result;
  let pow_x = x * x;
  result += 0.5 * p.C_1 * p.C_1 * pow_x;
  pow_x *= x; // == x^3
  dhdx.v += 4 * p.C_4 * pow_x;
  pow_x *= x; // == x^4
  result += p.C_4 * pow_x;
  pow_x *= x; // == x^5
  dhdx.v += 6 * p.C_6 * pow_x;
  pow_x *= x; // == x^6 (the C++ comment says x^5)
  result += p.C_6 * pow_x;
  return result;
};

/** Derivatives of F_conj with respect to N_ij, N_ji and N_conj_ij (the C++ out-parameters). */
export interface FConjD { dij: number; dji: number; dconj: number }

/**
 * PairLCBOP::F_conj: the bicubic F_conj(N_ij, N_ji) interpolated for N_conj = 0 and 1 and blended by N_conj.
 * The cell index of N truncates with floor; the C++ casts a negative floor to size_t (which the MIN then
 * clamps to 2 on the usual x86-64 build), so a negative N takes the last cell.
 */
export const F_conj = (T: LcbopTables, N_ij: number, N_ji: number, N_conj_ij: number, out: FConjD): number => {
  const fi = Math.floor(N_ij), fj = Math.floor(N_ji);
  const N_ij_int = fi < 0 ? 2 : cmin(fi, 2); // 2 is the highest number of field
  const N_ji_int = fj < 0 ? 2 : cmin(fj, 2);
  const x = N_ij - N_ij_int;
  const y = N_ji - N_ji_int;
  const f0 = T.F_conj_field[N_ij_int][N_ji_int][0];
  const f1 = T.F_conj_field[N_ij_int][N_ji_int][1];
  let F_0 = 0, F_1 = 0;
  let dF_0_dx = 0, dF_0_dy = 0;
  let dF_1_dx = 0, dF_1_dy = 0;
  if (N_conj_ij < 1) {
    // term-by-term in the C++ order: l, r, then the x and y derivative sums
    let l = (1 - y) * (1 - x);
    let r = (f0.f_00 + x * x * f0.f_x_10 + y * y * f0.f_y_01);
    F_0 += l * r; dF_0_dx += -(1 - y) * r + l * 2 * x * f0.f_x_10; dF_0_dy += -(1 - x) * r + l * 2 * y * f0.f_y_01;
    l = (1 - y) * x;
    r = (f0.f_10 + (1 - x) * (1 - x) * f0.f_x_00 + y * y * f0.f_y_11);
    F_0 += l * r; dF_0_dx += (1 - y) * r - l * 2 * (1 - x) * f0.f_x_00; dF_0_dy += -x * r + l * 2 * y * f0.f_y_11;
    l = y * (1 - x);
    r = (f0.f_01 + x * x * f0.f_x_11 + (1 - y) * (1 - y) * f0.f_y_00);
    F_0 += l * r; dF_0_dx += -y * r + l * 2 * x * f0.f_x_11; dF_0_dy += (1 - x) * r - l * 2 * (1 - y) * f0.f_y_00;
    l = y * x;
    r = (f0.f_11 + (1 - x) * (1 - x) * f0.f_x_01 + (1 - y) * (1 - y) * f0.f_y_10);
    F_0 += l * r; dF_0_dx += y * r - l * 2 * (1 - x) * f0.f_x_01; dF_0_dy += x * r - l * 2 * (1 - y) * f0.f_y_10;
  }
  if (N_conj_ij > 0) {
    let l = (1 - y) * (1 - x);
    let r = (f0.f_00 + x * x * f1.f_x_10 + y * y * f1.f_y_01);
    F_1 += l * r; dF_1_dx += -(1 - y) * r + l * 2 * x * f1.f_x_10; dF_1_dy += -(1 - x) * r + l * 2 * y * f1.f_y_01;
    l = (1 - y) * x;
    r = (f1.f_10 + (1 - x) * (1 - x) * f1.f_x_00 + y * y * f1.f_y_11);
    F_1 += l * r; dF_1_dx += (1 - y) * r - l * 2 * (1 - x) * f1.f_x_00; dF_1_dy += -x * r + l * 2 * y * f1.f_y_11;
    l = y * (1 - x);
    r = (f1.f_01 + x * x * f1.f_x_11 + (1 - y) * (1 - y) * f1.f_y_00);
    F_1 += l * r; dF_1_dx += -y * r + l * 2 * x * f1.f_x_11; dF_1_dy += (1 - x) * r - l * 2 * (1 - y) * f1.f_y_00;
    l = y * x;
    r = (f1.f_11 + (1 - x) * (1 - x) * f1.f_x_01 + (1 - y) * (1 - y) * f1.f_y_10);
    F_1 += l * r; dF_1_dx += y * r - l * 2 * (1 - x) * f1.f_x_01; dF_1_dy += x * r - l * 2 * (1 - y) * f1.f_y_10;
  }
  const result = (1 - N_conj_ij) * F_0 + N_conj_ij * F_1;
  out.dij = (1 - N_conj_ij) * dF_0_dx + N_conj_ij * dF_1_dx;
  out.dji = (1 - N_conj_ij) * dF_0_dy + N_conj_ij * dF_1_dy;
  out.dconj = -F_0 + F_1;
  return result;
};
