/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_airebo.h (Sp5th, Sp, Sp2 inline functions),
 * src/MANYBODY/pair_airebo.cpp (Spbicubic, Sptricubic, Sptricubic_patch_adjust,
 * Sptricubic_patch_coeffs, Spbicubic_patch_adjust, Spbicubic_patch_coeffs) and
 * src/MATH/math_special.h (powint), at commit b332f28.
 * Authors credited in the C++ file headers: Ase Henry (MIT); bugfixes and optimizations:
 * Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer (Temple U), Markus Hoehnerbach
 * (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M: Thomas C. O'Connor (JHU) 2014.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { CINV_BI, CINV_TRI } from './cinv';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** Output slot for a derivative (the C++ code passes double * for it). */
export interface Ref { v: number }

/** MathSpecial::powint: integer power by repeated squaring (same rounding as LAMMPS). */
export const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

const PI = 3.14159265358979323846;

/** Fifth order spline by Horner's rule; coeffs[off..off+5]; derivative into d. */
export const sp5th = (x: number, coeffs: Float64Array, off: number, d: Ref): number => {
  let f = coeffs[off + 5] * x;
  let dd = 5.0 * coeffs[off + 5] * x;
  f += coeffs[off + 4];
  dd += 4.0 * coeffs[off + 4];
  f *= x;
  dd *= x;
  f += coeffs[off + 3];
  dd += 3.0 * coeffs[off + 3];
  f *= x;
  dd *= x;
  f += coeffs[off + 2];
  dd += 2.0 * coeffs[off + 2];
  f *= x;
  dd *= x;
  f += coeffs[off + 1];
  dd += coeffs[off + 1];
  f *= x;
  f += coeffs[off];
  d.v = dd;
  return f;
};

/** Cutoff function S(t) of pair_airebo.h "Sp": returns the cutoff, derivative into dX. */
export const sp = (xij: number, xmin: number, xmax: number, dX: Ref): number => {
  let cutoff: number;
  const t = (xij - xmin) / (xmax - xmin);
  if (t <= 0.0) {
    cutoff = 1.0;
    dX.v = 0.0;
  } else if (t >= 1.0) {
    cutoff = 0.0;
    dX.v = 0.0;
  } else {
    cutoff = 0.5 * (1.0 + libmCos(t * PI));
    dX.v = (-0.5 * PI * libmSin(t * PI)) / (xmax - xmin);
  }
  return cutoff;
};

/** LJ cutoff function Sp2 of pair_airebo.h. */
export const sp2 = (xij: number, xmin: number, xmax: number, dX: Ref): number => {
  let cutoff: number;
  const t = (xij - xmin) / (xmax - xmin);
  if (t <= 0.0) {
    cutoff = 1.0;
    dX.v = 0.0;
  } else if (t >= 1.0) {
    cutoff = 0.0;
    dX.v = 0.0;
  } else {
    cutoff = 1.0 - t * t * (3.0 - 2.0 * t);
    dX.v = (6.0 * (t * t - t)) / (xmax - xmin);
  }
  return cutoff;
};

/** Bicubic spline: coeffs[off + 4*i + j] (16 values); df[0] = d/dx, df[1] = d/dy. */
export const spbicubic = (x: number, y: number, coeffs: Float64Array, off: number, df: Float64Array): number => {
  let f = 0.0;
  df[0] = 0.0;
  df[1] = 0.0;
  let xn = 1.0;
  let xn1 = 0.0;
  for (let i = 0; i < 4; i++) {
    let yn = 1.0;
    let yn1 = 0.0;
    for (let j = 0; j < 4; j++) {
      const c = coeffs[off + i * 4 + j];
      f += c * xn * yn;
      if (i > 0) df[0] += c * i * xn1 * yn;
      if (j > 0) df[1] += c * j * xn * yn1;
      yn1 = yn;
      yn *= y;
    }
    xn1 = xn;
    xn *= x;
  }
  return f;
};

/** Tricubic spline: coeffs[off + 16*i + 4*j + k] (64 values); df = d/dx, d/dy, d/dz. */
export const sptricubic = (
  x: number, y: number, z: number, coeffs: Float64Array, off: number, df: Float64Array,
): number => {
  let f = 0.0;
  df[0] = 0.0;
  df[1] = 0.0;
  df[2] = 0.0;
  let xn = 1.0;
  let xn1 = 0.0;
  for (let i = 0; i < 4; i++) {
    const ir = i;
    let yn = 1.0;
    let yn1 = 0.0;
    for (let j = 0; j < 4; j++) {
      const jr = j;
      let zn = 1.0;
      let zn1 = 0.0;
      for (let k = 0; k < 4; k++) {
        const kr = k;
        const c = coeffs[off + 16 * i + 4 * j + k];
        f += c * xn * yn * zn;
        if (i > 0) df[0] += c * ir * xn1 * yn * zn;
        if (j > 0) df[1] += c * jr * xn * yn1 * zn;
        if (k > 0) df[2] += c * kr * xn * yn * zn1;
        zn1 = zn;
        zn *= z;
      }
      yn1 = yn;
      yn *= y;
    }
    xn1 = xn;
    xn *= x;
  }
  return f;
};

const BINOM = [1, 1, 2, 6];

/** Sptricubic_patch_adjust: dl is the 64-value patch at offset off; dir R, M or L. */
const sptricubicPatchAdjust = (dl: Float64Array, off: number, wid: number, lo: number, dir: string): void => {
  let rowOuterL = 16, rowInnerL = 1, colL = 4;
  if (dir === 'R') {
    rowOuterL = 4;
    colL = 16;
  } else if (dir === 'M') {
    colL = 4;
  } else if (dir === 'L') {
    rowInnerL = 4;
    colL = 1;
  }
  for (let rowOuter = 0; rowOuter < 4; rowOuter++) {
    for (let rowInner = 0; rowInner < 4; rowInner++) {
      for (let col = 0; col < 4; col++) {
        let acc = 0;
        for (let k = col; k < 4; k++) {
          acc += dl[off + rowOuterL * rowOuter + rowInnerL * rowInner + colL * k]
            * powint(wid, -k) * powint(-lo, k - col) * BINOM[k] / BINOM[col] / BINOM[k - col];
        }
        dl[off + rowOuterL * rowOuter + rowInnerL * rowInner + colL * col] = acc;
      }
    }
  }
};

/**
 * Sptricubic_patch_coeffs: the 64 coefficients of one tricubic cell from its 8 corner values
 * y, y1 (d/dx), y2 (d/dy), y3 (d/dz), written to dl[off..off+63].
 */
export const sptricubicPatchCoeffs = (
  xmin: number, xmax: number, ymin: number, ymax: number, zmin: number, zmax: number,
  y: Float64Array, y1: Float64Array, y2: Float64Array, y3: Float64Array,
  dl: Float64Array, off: number,
): void => {
  const dx = xmax - xmin;
  const dy = ymax - ymin;
  const dz = zmax - zmin;
  const x = new Float64Array(32);
  for (let i = 0; i < 8; i++) {
    x[i + 0 * 8] = y[i];
    x[i + 1 * 8] = y1[i] * dx;
    x[i + 2 * 8] = y2[i] * dy;
    x[i + 3 * 8] = y3[i] * dz;
  }
  for (let i = 0; i < 64; i++) {
    let s = 0;
    for (let k = 0; k < 32; k++) s += x[k] * CINV_TRI[i * 32 + k];
    dl[off + i] = s;
  }
  sptricubicPatchAdjust(dl, off, dx, xmin, 'R');
  sptricubicPatchAdjust(dl, off, dy, ymin, 'M');
  sptricubicPatchAdjust(dl, off, dz, zmin, 'L');
};

/** Spbicubic_patch_adjust: dl is the 16-value patch at offset off. */
const spbicubicPatchAdjust = (dl: Float64Array, off: number, wid: number, lo: number, dir: string): void => {
  const rowL = dir === 'R' ? 1 : 4;
  const colL = dir === 'L' ? 1 : 4;
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      let acc = 0;
      for (let k = col; k < 4; k++) {
        acc += dl[off + rowL * row + colL * k] * powint(wid, -k) * powint(-lo, k - col)
          * BINOM[k] / BINOM[col] / BINOM[k - col];
      }
      dl[off + rowL * row + colL * col] = acc;
    }
  }
};

/** Spbicubic_patch_coeffs: 16 coefficients of one bicubic cell (values y, derivatives y1, y2) into dl[off..]. */
export const spbicubicPatchCoeffs = (
  xmin: number, xmax: number, ymin: number, ymax: number,
  y: Float64Array, y1: Float64Array, y2: Float64Array, dl: Float64Array, off: number,
): void => {
  const dx = xmax - xmin;
  const dy = ymax - ymin;
  const x = new Float64Array(12);
  for (let i = 0; i < 4; i++) {
    x[i + 0 * 4] = y[i];
    x[i + 1 * 4] = y1[i] * dx;
    x[i + 2 * 4] = y2[i] * dy;
  }
  for (let i = 0; i < 16; i++) {
    let s = 0;
    for (let k = 0; k < 12; k++) s += x[k] * CINV_BI[i * 12 + k];
    dl[off + i] = s;
  }
  spbicubicPatchAdjust(dl, off, dx, xmin, 'R');
  spbicubicPatchAdjust(dl, off, dy, ymin, 'L');
};
