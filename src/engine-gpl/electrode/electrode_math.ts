/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ELECTRODE/electrode_math.h (the erfc approximation), src/KSPACE/ewald_const.h (its constants) and the
 * dense matrix inversion of src/ELECTRODE/fix_electrode_conp.cpp (dgetrf/dgetri, here a Gauss-Jordan elimination
 * with partial pivoting) at commit b332f28. ELECTRODE contributing authors: Ludwig Ahrens-Iwers (TUHH),
 * Shern Tee (UQ), Robert Meissner (TUHH), and the LAMMPS developers. Copyright Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */

/** ewald_const.h: EWALD_P and A1..A5 of the Abramowitz-Stegun 7.1.26 erfc approximation. */
export const EWALD_P = 0.3275911;
export const A1 = 0.254829592;
export const A2 = -0.284496736;
export const A3 = 1.421413741;
export const A4 = -1.453152027;
export const A5 = 1.061405429;

import { exp as libmExp } from '../libm/exp';
import { erf as libmErf, erfc as libmErfc } from '../libm/erf';

/** math_const.h values used by the ELECTRODE sources. */
export const MY_PI = Math.PI;
export const MY_PI2 = Math.PI / 2;
export const MY_2PI = 2 * Math.PI;
export const MY_4PI = 4 * Math.PI;
export const MY_PIS = 1.77245385090551602729; // math_const.h (sqrt(pi) rounded once; Math.sqrt(Math.PI) is one ulp below)
export const MY_SQRT2 = Math.SQRT2;

/** electrode_math.h: "erfc(ERFCMAX) < machine epsilon(double)". */
const ERFCMAX = 5.8;

/** electrode_math.h safe_erfc: the approximate erfc used by every electrode pair term. */
export const safeErfc = (x: number): number => {
  if (x > ERFCMAX) return 0.0;
  const expm2 = libmExp(-x * x);
  const t = 1.0 / (1.0 + EWALD_P * x);
  return t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
};

/**
 * electrode_math.h safe_derfcr: returns d/dr erfc(r) at x = r (with the cut-off erfc set to 0 beyond ERFCMAX);
 * the erfc value is written to out.erfc.
 */
export const safeDerfcr = (x: number, out: { erfc: number }): number => {
  if (x > ERFCMAX) {
    out.erfc = 0.0;
    return 0.0;
  }
  const x2 = x * x;
  const expm2 = libmExp(-x2);
  const t = 1.0 / (1.0 + EWALD_P * x);
  out.erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
  return -out.erfc - 2.0 * expm2 * x / MY_PIS;
};

/**
 * In-place inverse of an n x n row-major matrix by Gauss-Jordan elimination with partial pivoting (the role of
 * dgetrf + dgetri in fix_electrode_conp.cpp). Returns false if the matrix is singular.
 */
export const invertDense = (a: Float64Array, n: number): boolean => {
  const m = Float64Array.from(a);
  const inv = new Float64Array(n * n);
  for (let i = 0; i < n; i++) inv[i * n + i] = 1;
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r * n + c]) > Math.abs(m[p * n + c])) p = r;
    if (m[p * n + c] === 0) return false;
    if (p !== c) {
      for (let k = 0; k < n; k++) {
        const t1 = m[c * n + k]; m[c * n + k] = m[p * n + k]; m[p * n + k] = t1;
        const t2 = inv[c * n + k]; inv[c * n + k] = inv[p * n + k]; inv[p * n + k] = t2;
      }
    }
    const d = m[c * n + c];
    for (let k = 0; k < n; k++) { m[c * n + k] /= d; inv[c * n + k] /= d; }
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = m[r * n + c];
      if (f === 0) continue;
      for (let k = 0; k < n; k++) {
        m[r * n + k] -= f * m[c * n + k];
        inv[r * n + k] -= f * inv[c * n + k];
      }
    }
  }
  a.set(inv);
  return true;
};

/** erf(x) for the Slab2d correction: the C library's, as native slab_2d.cpp calls erf() (../libm/erf.ts). */
export const erfAccurate = (x: number): number => libmErf(x);
