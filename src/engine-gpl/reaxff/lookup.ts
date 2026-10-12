/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/reaxff_lookup.cpp (Tridiagonal_Solve, Natural_Cubic_Spline,
 * Complete_Cubic_Spline, Init_Lookup_Tables), at commit b332f28. Authors credited in the C++
 * file headers: Hasan Metin Aktulga (Purdue University, now Lawrence Berkeley National
 * Laboratory); heavily modified and adapted for LAMMPS by the LAMMPS developers. Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The C++ code passes sub-arrays (&h[1], &coef[1]); here the same arrays are used with explicit
 * offsets. Table index r (1..tabulate) holds the cubic coefficients of the interval starting at r*dx.
 */
import type { ControlParams, ReaxParam } from './ffield';
import { lrVdwCoulomb } from './nonbonded';

export interface CubicCoef { a: Float64Array; b: Float64Array; c: Float64Array; d: Float64Array }

export interface LRTable {
  dx: number;
  inv_dx: number;
  n: number;
  H: CubicCoef; vdW: CubicCoef; CEvd: CubicCoef; ele: CubicCoef; CEclmb: CubicCoef;
}

export interface LookupTables {
  /** indexed tmin * numAtomTypes + tmax; null when neither type is present in the system. */
  table: (LRTable | null)[];
}

const newCoef = (n: number): CubicCoef => ({
  a: new Float64Array(n), b: new Float64Array(n), c: new Float64Array(n), d: new Float64Array(n),
});

/** Tridiagonal_Solve on arrays of length n (the LAMMPS arrays c and d are overwritten, as in C). */
const tridiagonalSolve = (a: Float64Array, b: Float64Array, c: Float64Array, d: Float64Array, x: Float64Array, n: number): void => {
  c[0] /= b[0];
  d[0] /= b[0];
  for (let i = 1; i < n; i++) {
    const id = b[i] - c[i - 1] * a[i];
    c[i] /= id;
    d[i] = (d[i] - d[i - 1] * a[i]) / id;
  }
  x[n - 1] = d[n - 1];
  for (let i = n - 2; i >= 0; i--) x[i] = d[i] - c[i] * x[i + 1];
};

/**
 * Natural_Cubic_Spline: n points; the C++ arrays h, f are sub-arrays starting at hOff / fOff
 * (sub-index k is global index off + k), and the coefficients of interval i (1 <= i <= n-1) go to
 * sub-index i-1 of coef (global cOff + i - 1).
 */
export const naturalCubicSpline = (
  h: Float64Array, hOff: number, f: Float64Array, fOff: number, coef: CubicCoef, cOff: number, n: number,
): void => {
  const hh = (k: number) => h[hOff + k];
  const ff = (k: number) => f[fOff + k];
  const a = new Float64Array(n), b = new Float64Array(n), c = new Float64Array(n), d = new Float64Array(n), v = new Float64Array(n);
  a[0] = a[1] = a[n - 1] = 0;
  for (let i = 2; i < n - 1; i++) a[i] = hh(i - 1);
  b[0] = b[n - 1] = 0;
  for (let i = 1; i < n - 1; i++) b[i] = 2 * (hh(i - 1) + hh(i));
  c[0] = c[n - 2] = c[n - 1] = 0;
  for (let i = 1; i < n - 2; i++) c[i] = hh(i);
  d[0] = d[n - 1] = 0;
  for (let i = 1; i < n - 1; i++) d[i] = 6 * ((ff(i + 1) - ff(i)) / hh(i) - (ff(i) - ff(i - 1)) / hh(i - 1));
  v[0] = 0;
  v[n - 1] = 0;
  tridiagonalSolve(a.subarray(1), b.subarray(1), c.subarray(1), d.subarray(1), v.subarray(1), n - 2);
  for (let i = 1; i < n; i++) {
    const g = cOff + i - 1;
    coef.d[g] = (v[i] - v[i - 1]) / (6 * hh(i - 1));
    coef.c[g] = v[i] / 2;
    coef.b[g] = (ff(i) - ff(i - 1)) / hh(i - 1) + hh(i - 1) * (2 * v[i] + v[i - 1]) / 6;
    coef.a[g] = ff(i);
  }
};

/** Complete_Cubic_Spline (clamped ends v0 and vlast); the C++ expression f[n-2]/h[n-2] is kept as written. */
export const completeCubicSpline = (
  h: Float64Array, hOff: number, f: Float64Array, fOff: number, v0: number, vlast: number, coef: CubicCoef, cOff: number, n: number,
): void => {
  const hh = (k: number) => h[hOff + k];
  const ff = (k: number) => f[fOff + k];
  const a = new Float64Array(n), b = new Float64Array(n), c = new Float64Array(n), d = new Float64Array(n), v = new Float64Array(n);
  a[0] = 0;
  for (let i = 1; i < n; i++) a[i] = hh(i - 1);
  b[0] = 2 * hh(0);
  for (let i = 1; i < n; i++) b[i] = 2 * (hh(i - 1) + hh(i));
  c[n - 1] = 0;
  for (let i = 0; i < n - 1; i++) c[i] = hh(i);
  d[0] = (6 * (ff(1) - ff(0))) / hh(0) - 6 * v0;
  d[n - 1] = 6 * vlast - 6 * (ff(n - 1) - ff(n - 2) / hh(n - 2));
  for (let i = 1; i < n - 1; i++) d[i] = 6 * ((ff(i + 1) - ff(i)) / hh(i) - (ff(i) - ff(i - 1)) / hh(i - 1));
  tridiagonalSolve(a, b, c, d, v, n);
  for (let i = 1; i < n; i++) {
    const g = cOff + i - 1;
    coef.d[g] = (v[i] - v[i - 1]) / (6 * hh(i - 1));
    coef.c[g] = v[i] / 2;
    coef.b[g] = (ff(i) - ff(i - 1)) / hh(i - 1) + hh(i - 1) * (2 * v[i] + v[i - 1]) / 6;
    coef.a[g] = ff(i);
  }
};

/**
 * Init_Lookup_Tables: the spline tables of the pairs of types present in the system. Values are
 * sampled on r = dr, 2dr, ... tabulate*dr (dr = nonb_cut / tabulate) and the two end conditions
 * come from the first and last samples. Arrays of length tabulate+2 are used as in the C++ code.
 */
export const initLookupTables = (param: ReaxParam, ctrl: ControlParams, T: Float64Array, typesPresent: Uint8Array): LookupTables => {
  const nt = param.numAtomTypes;
  const table: (LRTable | null)[] = new Array(nt * nt).fill(null);
  const tab = ctrl.tabulate;
  const dr = ctrl.nonbCut / tab;
  const len = tab + 2;
  for (let i = 0; i < nt; i++) {
    if (!typesPresent[i]) continue;
    for (let j = i; j < nt; j++) {
      if (!typesPresent[j]) continue;
      const h = new Float64Array(len), fh = new Float64Array(len), fvdw = new Float64Array(len);
      const fCEvd = new Float64Array(len), fele = new Float64Array(len), fCEclmb = new Float64Array(len);
      const H = newCoef(len), vdW = newCoef(len), CEvd = newCoef(len), ele = newCoef(len), CEclmb = newCoef(len);
      const dx = dr;
      let v0_vdw = 0, v0_ele = 0, vlast_vdw = 0, vlast_ele = 0;
      for (let r = 1; r <= tab; r++) {
        const y = lrVdwCoulomb(param, ctrl, T, i, j, r * dr);
        h[r] = dx;
        fh[r] = y.H;
        fvdw[r] = y.e_vdW;
        fCEvd[r] = y.CEvd;
        fele[r] = y.e_ele;
        fCEclmb[r] = y.CEclmb;
      }
      // init the start-end points
      h[tab + 1] = dx;
      v0_vdw = fCEvd[1];
      v0_ele = fCEclmb[1];
      fh[tab + 1] = fh[tab];
      fvdw[tab + 1] = fvdw[tab];
      fCEvd[tab + 1] = fCEvd[tab];
      fele[tab + 1] = fele[tab];
      fCEclmb[tab + 1] = fCEclmb[tab];
      vlast_vdw = fCEvd[tab];
      vlast_ele = fele[tab];
      // sub-arrays &h[1] etc.: index 0 of a sub-array is the global index 1
      naturalCubicSpline(h, 1, fh, 1, H, 1, tab + 1);
      completeCubicSpline(h, 1, fvdw, 1, v0_vdw, vlast_vdw, vdW, 1, tab + 1);
      naturalCubicSpline(h, 1, fCEvd, 1, CEvd, 1, tab + 1);
      completeCubicSpline(h, 1, fele, 1, v0_ele, vlast_ele, ele, 1, tab + 1);
      naturalCubicSpline(h, 1, fCEclmb, 1, CEclmb, 1, tab + 1);
      table[i * nt + j] = { dx, inv_dx: tab / ctrl.nonbCut, n: len, H, vdW, CEvd, ele, CEclmb };
    }
  }
  return { table };
};
