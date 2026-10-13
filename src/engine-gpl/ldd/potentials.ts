/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BOCS/ldd_potential.h, ldd_potential.cpp
 * (table reader, table index and the spline evaluation helpers), ldd_potential_quadratic.cpp,
 * ldd_potential_tablespline.cpp and ldd_potential_tablegradspline.cpp (the "quadratic", "table/spline" and
 * "table/gradspline" embedding potentials of pair_style ldd). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/
 * LICENSE). Contributed by Michael R. DeLyser and Maria C. Lesniewski (Pennsylvania State University), from the BOCS
 * package sources; the spline is from Numerical Recipes (second edition), as the source states.
 *
 * u(rho) is the potential of a local density rho; f(rho) = -du/drho is the force-like derivative used in the pair
 * force. "table/gradspline" takes its f from the spline of u (its gradient), "table/spline" splines the f column.
 */
import { StyleError } from '../../engine/force/types';

/** Values beyond these in the spline end conditions mean "second derivative zero" (ldd_potential_tablespline.cpp). */
const BIG = 0.99e30;
const BIGGER = 0.999e30;
const SMALL = 1.0e-8;

export interface LddPotential {
  readonly nCoeffs: number;
  u(rho: number): number;
  f(rho: number): number;
}

/** ldd_potential_quadratic.cpp: u = a rho^2 + b rho + c, the arguments "a b c" after the keyword. */
export class LddPotentialQuadratic implements LddPotential {
  readonly nCoeffs = 3;
  private readonly c: Float64Array;

  constructor(args: string[]) {
    if (args.length < 3) throw new StyleError('ldd potential quadratic needs 3 coefficients: a b c');
    const v = args.slice(0, 3).map(Number);
    if (!v.every((x) => Number.isFinite(x))) throw new StyleError('ldd potential quadratic needs 3 numeric coefficients: a b c');
    this.c = new Float64Array(v);
  }

  u(rho: number): number {
    return this.c[0] * rho * rho + this.c[1] * rho + this.c[2];
  }

  f(rho: number): number {
    return -(2.0 * this.c[0] * rho + this.c[1]);
  }
}

/** The tabulated potential (LddPotential::read_table_file, get_table_index, calc_A_table, spline, splint). */
abstract class LddTablePotential implements LddPotential {
  readonly nCoeffs = 1;
  protected r = new Float64Array(0);
  protected uT = new Float64Array(0);
  protected fT = new Float64Array(0);
  protected u2 = new Float64Array(0);
  protected f2 = new Float64Array(0);
  protected dr = 0;

  /** Reads "r u f" lines (LddPotential::read_table_file) and builds the second derivatives of u and f. */
  constructor(text: string, keyword: string, splineF: boolean) {
    const rv: number[] = [], uv: number[] = [], fv: number[] = [];
    let lineNo = 0;
    for (const raw of text.split(/\r?\n/)) {
      lineNo++;
      const w = raw.replace(/#.*$/, '').trim();
      if (w === '') continue;
      const v = w.split(/\s+/).map(Number);
      if (v.length < 3 || v.slice(0, 3).some((x) => !Number.isFinite(x))) {
        throw new StyleError(`unable to read rho u f from line ${lineNo} of the ldd ${keyword} table`);
      }
      rv.push(v[0]);
      uv.push(v[1]);
      fv.push(v[2]);
    }
    const n = rv.length;
    if (n < 2) throw new StyleError(`the ldd ${keyword} table needs at least two points`);
    this.r = Float64Array.from(rv);
    this.uT = Float64Array.from(uv);
    this.fT = Float64Array.from(fv);
    this.dr = this.r[1] - this.r[0];
    for (let i = 0; i < n - 1; i++) {
      if (Math.abs(this.r[i + 1] - this.r[i] - this.dr) > this.dr / 100.0) {
        throw new StyleError(`ldd ${keyword} table: the rho grid is not uniform at point ${i + 1}`);
      }
    }
    this.u2 = new Float64Array(n);
    this.f2 = new Float64Array(n);
    spline(this.r, this.uT, n, this.u2);
    if (splineF) spline(this.r, this.fT, n, this.f2);
  }

  protected get n(): number { return this.r.length; }

  protected index(rho: number): number {
    const n = this.n;
    if (rho > this.r[n - 1]) throw new StyleError(`local density = ${rho} > ${this.r[n - 1]} = table hi`);
    if (rho < this.r[0]) throw new StyleError(`local density = ${rho} < ${this.r[0]} = table lo`);
    return Math.floor((rho - this.r[0]) / this.dr);
  }

  protected splint(idx: number, y: Float64Array, y2: Float64Array, rho: number): number {
    const a = (this.r[idx + 1] - rho) / this.dr, b = 1.0 - a;
    const x0 = this.r[idx], x1 = this.r[idx + 1];
    if (Math.abs(rho - x0) <= SMALL) return y[idx];
    if (Math.abs(x1 - rho) <= SMALL) return y[idx + 1];
    return a * y[idx] + b * y[idx + 1] + ((a * a * a - a) * y2[idx] + (b * b * b - b) * y2[idx + 1]) * this.dr * this.dr / 6.0;
  }

  protected dsplint(idx: number, y: Float64Array, y2: Float64Array, rho: number): number {
    const a = (this.r[idx + 1] - rho) / this.dr, b = 1.0 - a;
    return 1 / this.dr * (y[idx + 1] - y[idx]) - (3 * a * a - 1) / 6 * this.dr * y2[idx] + (3 * b * b - 1) / 6 * this.dr * y2[idx + 1];
  }

  abstract u(rho: number): number;
  abstract f(rho: number): number;
}

/** Calculates the second derivatives at the knots (Numerical Recipes spline with zero end curvature). */
const spline = (x: Float64Array, y: Float64Array, n: number, y2: Float64Array): void => {
  const u = new Float64Array(n);
  const yp1 = BIGGER, ypn = BIGGER;
  if (yp1 > BIG) {
    y2[0] = u[0] = 0.0;
  } else {
    y2[0] = -0.5;
    u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
  }
  for (let i = 1; i < n - 1; i++) {
    const sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    const p = sig * y2[i - 1] + 2.0;
    y2[i] = (sig - 1.0) / p;
    u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
  }
  let qn = 0, un = 0;
  if (ypn > BIG) {
    qn = un = 0.0;
  } else {
    qn = 0.5;
    un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
  }
  y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
  for (let j = n - 2; j > 0; j--) y2[j] = y2[j] * y2[j + 1] + u[j];
};

/** ldd_potential_tablespline.cpp: u and f are both splined. */
export class LddPotentialTableSpline extends LddTablePotential {
  constructor(text: string) {
    super(text, 'table/spline', true);
  }

  u(rho: number): number {
    const n = this.n;
    if (rho === this.r[n - 1]) return this.uT[n - 1];
    return this.splint(this.index(rho), this.uT, this.u2, rho);
  }

  f(rho: number): number {
    const n = this.n;
    if (rho === this.r[n - 1]) return this.fT[n - 1];
    return this.splint(this.index(rho), this.fT, this.f2, rho);
  }
}

/** ldd_potential_tablegradspline.cpp: u is splined and f = -du/drho from the spline (the gradient of the table). */
export class LddPotentialTableGradSpline extends LddTablePotential {
  constructor(text: string) {
    super(text, 'table/gradspline', false);
  }

  u(rho: number): number {
    const n = this.n;
    if (rho === this.r[n - 1]) return this.uT[n - 1];
    return this.splint(this.index(rho), this.uT, this.u2, rho);
  }

  f(rho: number): number {
    const n = this.n;
    if (rho === this.r[n - 1]) return this.fT[n - 1];
    return -this.dsplint(this.index(rho), this.uT, this.u2, rho);
  }
}
