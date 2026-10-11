import { StyleError } from './types';
import { parseNum } from './util';

/*
 * Tabulated angular functions for pair_style sw/angle/table (docs.lammps.org/pair_sw_angle_table.html). Its tabulated
 * angular potential "has to be of the format as used in the angle_style table command" (docs.lammps.org/angle_table.html),
 * so the section reader below is the angle_style table one:
 *
 *   "The parameter "N" is required and its value is the number of table entries that follow."
 *   "The "FP" parameter is optional.  If used, it is followed by two values fplo and fphi, which are the
 *    second derivatives at the innermost and outermost angle settings."
 *   "The "EQ" parameter is also optional."
 *   "On each line, the first value is the index from 1 to N, the second value is the angle value (in degrees), the third
 *    value is the energy (in energy units), and the fourth is -dE/d(theta) (also in energy units)."
 *   "The angle values must increase from one line to the next." "The angle values must also begin with 0.0 and end with 180.0"
 *
 * Interpolation (docs.lammps.org/angle_table.html): "For the linear style, the angle is used to find 2 surrounding table
 * values from which an energy or its derivative is computed by linear interpolation." "For the spline style, a cubic
 * spline coefficients are computed and stored at each of the N values in the table."
 *
 * Measured with native LAMMPS (black box), sw/angle/table with a custom angle file (nodes 0, 45, 90, 135, 180): the
 * energy of a grid point of a table with Nsw != Nfile equals the cubic spline through the Nfile energies whose end first
 * derivatives are the negated table derivatives (E'(0) = -D(0), E'(180) = -D(180), D = the fourth column); the derivative
 * column of such a grid is the cubic spline through the fourth column whose end first derivatives are the first
 * differences (D1 - D0)/(x1 - x0) and (DN - DN-1)/(xN - xN-1). Energies are then linear in the grid. The FP values did
 * not change any energy or force of that style (measured). With Nsw = Nfile the grid is the file nodes themselves; the
 * spline style is measured the same way (energy spline with the -D end slopes, derivative spline with the difference
 * end slopes).
 */

export interface AngleSection {
  n: number;
  /** angles in degrees, increasing from 0 to 180 */
  x: Float64Array;
  /** energies (energy units) */
  e: Float64Array;
  /** -dE/d(theta) in energy units per degree */
  d: Float64Array;
}

/** Parameter line of one section (keyword followed by values); FP and EQ are accepted, N is required. */
const parseParams = (line: string, filename: string, keyword: string): number => {
  const t = line.trim().split(/\s+/);
  let n = 0;
  let k = 0;
  while (k < t.length) {
    const key = t[k++];
    if (key === 'N') n = parseNum(t[k++], 'table parameter N');
    else if (key === 'FP') {
      parseNum(t[k++], 'table parameter FP fplo');
      parseNum(t[k++], 'table parameter FP fphi');
    } else if (key === 'EQ') parseNum(t[k++], 'table parameter EQ');
    else throw new StyleError(`angle table ${filename} section ${keyword}: unknown parameter '${key}' (N, FP and EQ are accepted)`);
  }
  if (!Number.isInteger(n) || n < 2) throw new StyleError(`angle table ${filename} section ${keyword} needs an integer N >= 2, got N = ${n}`);
  return n;
};

/** Reads the section whose keyword matches; other sections are skipped. */
export const readAngleSection = (text: string, filename: string, keyword: string): AngleSection => {
  const lines = text.split(/\r?\n/);
  let k = 0;
  while (k < lines.length) {
    const header = lines[k].trim();
    k++;
    if (header === '' || header.startsWith('#')) continue;
    const kw = header.split(/\s+/)[0];
    let params = '';
    while (k < lines.length) {
      const l = lines[k].trim();
      k++;
      if (l !== '' && !l.startsWith('#')) {
        params = l;
        break;
      }
    }
    if (params === '') throw new StyleError(`angle table ${filename}: section ${kw} has no parameter line`);
    const n = parseParams(params, filename, kw);
    const x = new Float64Array(n), e = new Float64Array(n), d = new Float64Array(n);
    let got = 0;
    while (got < n && k < lines.length) {
      const l = lines[k].trim();
      k++;
      if (l === '' || l.startsWith('#')) continue;
      const t = l.split(/\s+/);
      if (t.length < 4) throw new StyleError(`angle table ${filename}: section ${kw} data line '${l}' needs index, angle, energy and derivative`);
      const idx = parseNum(t[0], 'table index');
      if (!Number.isInteger(idx) || idx !== got + 1) throw new StyleError(`angle table ${filename}: section ${kw} line '${l}' should have index ${got + 1}`);
      x[got] = parseNum(t[1], 'table angle');
      e[got] = parseNum(t[2], 'table energy');
      d[got] = parseNum(t[3], 'table derivative');
      got++;
    }
    if (got < n) throw new StyleError(`angle table ${filename}: section ${kw} has ${got} of ${n} data lines`);
    if (kw === keyword) {
      if (x[0] !== 0 || x[n - 1] !== 180) {
        throw new StyleError(`angle table ${filename}: section ${kw} must begin with angle 0.0 and end with 180.0 (got ${x[0]} and ${x[n - 1]})`);
      }
      for (let m = 1; m < n; m++) {
        if (!(x[m] > x[m - 1])) throw new StyleError(`angle table ${filename}: section ${kw} angles must increase from one line to the next`);
      }
      return { n, x, e, d };
    }
  }
  throw new StyleError(`angle table ${filename} has no section with keyword '${keyword}'`);
};

/** Complete cubic spline through (x, y) with end first derivatives d0, dn; returns the second derivatives. */
export const splineSecond = (x: Float64Array, y: Float64Array, d0: number, dn: number): Float64Array => {
  const n = x.length;
  const M = new Float64Array(n);
  const h = new Float64Array(n - 1);
  for (let k = 0; k < n - 1; k++) h[k] = x[k + 1] - x[k];
  const lower = new Float64Array(n), diag = new Float64Array(n), upper = new Float64Array(n), rhs = new Float64Array(n);
  diag[0] = 2 * h[0];
  upper[0] = h[0];
  rhs[0] = 6 * ((y[1] - y[0]) / h[0] - d0);
  for (let k = 1; k < n - 1; k++) {
    lower[k] = h[k - 1];
    diag[k] = 2 * (h[k - 1] + h[k]);
    upper[k] = h[k];
    rhs[k] = 6 * ((y[k + 1] - y[k]) / h[k] - (y[k] - y[k - 1]) / h[k - 1]);
  }
  lower[n - 1] = h[n - 2];
  diag[n - 1] = 2 * h[n - 2];
  rhs[n - 1] = 6 * (dn - (y[n - 1] - y[n - 2]) / h[n - 2]);
  for (let k = 1; k < n; k++) {
    const w = lower[k] / diag[k - 1];
    diag[k] -= w * upper[k - 1];
    rhs[k] -= w * rhs[k - 1];
  }
  M[n - 1] = rhs[n - 1] / diag[n - 1];
  for (let k = n - 2; k >= 0; k--) M[k] = (rhs[k] - upper[k] * M[k + 1]) / diag[k];
  return M;
};

/** Interval index k with x[k] <= q < x[k+1], clamped to [0, n-2]. */
const interval = (x: Float64Array, q: number): number => {
  let lo = 0, hi = x.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (x[mid] > q) hi = mid;
    else lo = mid;
  }
  return lo;
};

/** Cubic spline value at q from its nodes x, values y and second derivatives M (clamped outside). */
const splineValue = (x: Float64Array, y: Float64Array, M: Float64Array, q: number): number => {
  const last = x.length - 1;
  if (q <= x[0]) return y[0];
  if (q >= x[last]) return y[last];
  const lo = interval(x, q), h = x[lo + 1] - x[lo];
  const a = (x[lo + 1] - q) / h, b = (q - x[lo]) / h;
  return a * y[lo] + b * y[lo + 1] + ((a * a * a - a) * M[lo] + (b * b * b - b) * M[lo + 1]) * (h * h) / 6;
};

/** A table as the pair style evaluates it: energy and -dE/d(theta) (per degree) at any angle in [0, 180]. */
export class AngleTable {
  private constructor(
    readonly style: 'linear' | 'spline',
    private readonly x: Float64Array,
    private readonly e: Float64Array,
    private readonly d: Float64Array,
    private readonly eM: Float64Array | null,
    private readonly dM: Float64Array | null,
  ) {}

  /** Builds the table of Nsw entries from a file section (Nsw = the sw entry's N). */
  static build(sec: AngleSection, nsw: number, style: 'linear' | 'spline', where: string): AngleTable {
    const n = sec.n;
    const d0 = sec.d[0], dn = sec.d[n - 1];
    // derivative end slopes of the derivative column: first differences of the first and last two values
    const fd0 = (sec.d[1] - sec.d[0]) / (sec.x[1] - sec.x[0]);
    const fdn = (sec.d[n - 1] - sec.d[n - 2]) / (sec.x[n - 1] - sec.x[n - 2]);
    if (nsw === n) {
      if (style === 'spline') {
        return new AngleTable(style, sec.x, sec.e, sec.d, splineSecond(sec.x, sec.e, -d0, -dn), splineSecond(sec.x, sec.d, fd0, fdn));
      }
      return new AngleTable(style, sec.x, sec.e, sec.d, null, null);
    }
    if (style === 'spline') {
      throw new StyleError(`${where}: the spline style of the angle table is only supported when N of the sw entry equals N of the table (${n}), got ${nsw}`);
    }
    // Nsw != Nfile: the grid is uniform in angle from 0 to 180 with Nsw points (measured, see the header)
    const eM = splineSecond(sec.x, sec.e, -d0, -dn);
    const dM = splineSecond(sec.x, sec.d, fd0, fdn);
    const x = new Float64Array(nsw), e = new Float64Array(nsw), d = new Float64Array(nsw);
    for (let k = 0; k < nsw; k++) {
      x[k] = (180 * k) / (nsw - 1);
      e[k] = splineValue(sec.x, sec.e, eM, x[k]);
      d[k] = splineValue(sec.x, sec.d, dM, x[k]);
    }
    return new AngleTable('linear', x, e, d, null, null);
  }

  /** Energy at the angle theta (degrees). */
  energy(theta: number): number {
    if (this.eM) return splineValue(this.x, this.e, this.eM, theta);
    const x = this.x;
    const k = interval(x, theta);
    const t = (theta - x[k]) / (x[k + 1] - x[k]);
    return this.e[k] + t * (this.e[k + 1] - this.e[k]);
  }

  /** -dE/d(theta) (energy units per degree) at the angle theta (degrees). */
  derivative(theta: number): number {
    if (this.dM) return splineValue(this.x, this.d, this.dM, theta);
    const x = this.x;
    const k = interval(x, theta);
    const t = (theta - x[k]) / (x[k + 1] - x[k]);
    return this.d[k] + t * (this.d[k + 1] - this.d[k]);
  }
}
