/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MESONT/pair_mesocnt.cpp and src/MESONT/pair_mesocnt.h at commit b332f28 (read_file, read_data,
 * spline_coeff, the spline evaluation functions, legendre, gl_init_nodes and gl_init_weights; contributing author
 * Philipp Kloza, University of Cambridge). Copyright (2003) Sandia Corporation (GNU General Public License). This
 * file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The mesocnt potential tables: a header (four table sizes, then r, sigma, delta1, delta2), the 1D tables uinf and
 * gamma, and the 2D tables phi and usemi, each given as rows "x y value" (the x index outer, the y index inner).
 * Tables are cubic-spline coefficients (finite-difference end slopes, as the C++ computes them); the evaluators
 * extrapolate or clamp as the C++ does.
 */
import { StyleError } from '../../engine/force/types';
import { cos as libmCos } from '../libm/sincos';

/** Parsed mesocnt potential file with its spline coefficients. */
export interface MesocntTables {
  uinfPoints: number;
  gammaPoints: number;
  phiPoints: number;
  usemiPoints: number;
  rAng: number;
  sigAng: number;
  delta1: number;
  delta2: number;
  hstartUinf: number;
  delhUinf: number;
  uinfCoeff: Float64Array;
  hstartGamma: number;
  delhGamma: number;
  gammaCoeff: Float64Array;
  hstartPhi: number;
  psistartPhi: number;
  delhPhi: number;
  delpsiPhi: number;
  phiCoeff: Float64Array;
  hstartUsemi: number;
  xistartUsemi: number;
  delhUsemi: number;
  delxiUsemi: number;
  usemiCoeff: Float64Array;
}

/** 10^k for k = 0..22: exact doubles, so m / 10^k is correctly rounded (the same value strtod gives). */
const POW10: number[] = Array.from({ length: 23 }, (_, k) => Number(`1e${k}`));

/**
 * A decimal "[-+]ddd[.ddd]" token parsed without allocating: exact when the digits form an integer below 2^53 and at
 * most 22 digits follow the point, then one IEEE division gives the correctly rounded value (as Number() does).
 * Anything else (exponents, more digits, other characters) returns undefined and the caller uses Number().
 */
export const parseDecimal = (t: string, s: number, e: number): number | undefined => {
  let p = s;
  let neg = false;
  const c0 = t.charCodeAt(p);
  if (c0 === 45 || c0 === 43) {
    neg = c0 === 45;
    p++;
  }
  let m = 0, k = 0, anyDigit = false, seenDot = false;
  for (; p < e; p++) {
    const c = t.charCodeAt(p);
    if (c >= 48 && c <= 57) {
      m = m * 10 + (c - 48);
      if (m > 9007199254740991) return undefined;
      anyDigit = true;
      if (seenDot) k++;
    } else if (c === 46 && !seenDot) seenDot = true;
    else return undefined;
  }
  if (!anyDigit || k > 22) return undefined;
  const v = m / POW10[k];
  return neg ? -v : v;
};

/**
 * Reads the whitespace-separated numbers of a potential file after its title line, one at a time, without building
 * per-line arrays (the file has about two million rows). Lines that start with '#' are comments.
 */
class NumberReader {
  private pos: number;
  constructor(private readonly text: string) {
    const nl = text.indexOf('\n');
    this.pos = nl < 0 ? text.length : nl + 1;
  }

  next(what: string): number {
    const t = this.text, len = t.length;
    let p = this.pos;
    for (;;) {
      while (p < len && t.charCodeAt(p) <= 32) p++;
      if (p >= len) throw new StyleError(`Error parsing ${what} of the mesocnt potential file`);
      if (t.charCodeAt(p) !== 35) break;
      while (p < len && t.charCodeAt(p) !== 10) p++;
    }
    const s = p;
    while (p < len && t.charCodeAt(p) > 32) p++;
    this.pos = p;
    const fast = parseDecimal(t, s, p);
    if (fast !== undefined) return fast;
    const v = Number(t.slice(s, p));
    if (!Number.isFinite(v)) throw new StyleError(`Error parsing ${what} of the mesocnt potential file`);
    return v;
  }
}

/**
 * PairMesoCNT::read_file: the header and the four tables, read in file order (one row per data point, as the C++
 * PotentialFileReader::next_values does). The spacings (delh, delpsi, delxi) come from the second row of a table
 * for x and from the second row of the first x-block for y, as read_data does.
 */
let parsedText: string | null = null;
let parsedTables: MesocntTables | null = null;

/** The parsed tables of the last file read (the same file is read once per pair_coeff of a run). */
export const readMesocntFile = (text: string): MesocntTables => {
  if (parsedTables && parsedText === text) return parsedTables;
  parsedTables = parseMesocntFile(text);
  parsedText = text;
  return parsedTables;
};

const parseMesocntFile = (text: string): MesocntTables => {
  const rd = new NumberReader(text);
  const uinfPoints = Math.trunc(rd.next('table sizes')), gammaPoints = Math.trunc(rd.next('table sizes'));
  const phiPoints = Math.trunc(rd.next('table sizes')), usemiPoints = Math.trunc(rd.next('table sizes'));
  const rAng = rd.next('header'), sigAng = rd.next('header'), delta1 = rd.next('header'), delta2 = rd.next('header');
  if (![uinfPoints, gammaPoints, phiPoints, usemiPoints].every((v) => Number.isFinite(v) && v > 1)) {
    throw new StyleError('Error parsing mesocnt potential file header: table sizes');
  }
  const read1 = (n: number): { start: number; d: number; data: Float64Array } => {
    const data = new Float64Array(n);
    let start = 0, d = 0, x = 0, xtemp = 0;
    for (let i = 0; i < n; i++) {
      if (i > 0) xtemp = x;
      x = rd.next('data');
      data[i] = rd.next('data');
      if (i === 0) start = x;
      else if (i === 1) d = x - xtemp;
    }
    return { start, d, data };
  };
  const read2 = (n: number): { xstart: number; ystart: number; dx: number; dy: number; data: Float64Array } => {
    const data = new Float64Array(n * n);
    let xstart = 0, ystart = 0, dx = 0, dy = 0, x = 0, y = 0, xtemp = 0, ytemp = 0;
    for (let i = 0; i < n; i++) {
      if (i > 0) xtemp = x;
      for (let j = 0; j < n; j++) {
        if (j > 0) ytemp = y;
        x = rd.next('data');
        y = rd.next('data');
        data[i * n + j] = rd.next('data');
        if (i === 0 && j === 0) ystart = y;
        if (j === 1) dy = y - ytemp;
      }
      if (i === 0) xstart = x;
      else if (i === 1) dx = x - xtemp;
    }
    return { xstart, ystart, dx, dy, data };
  };
  const uinf = read1(uinfPoints);
  const gamma = read1(gammaPoints);
  const phi = read2(phiPoints);
  const usemi = read2(usemiPoints);
  return {
    uinfPoints, gammaPoints, phiPoints, usemiPoints, rAng, sigAng, delta1, delta2,
    hstartUinf: uinf.start, delhUinf: uinf.d, uinfCoeff: splineCoeff1(uinf.data, uinf.d, uinfPoints),
    hstartGamma: gamma.start, delhGamma: gamma.d, gammaCoeff: splineCoeff1(gamma.data, gamma.d, gammaPoints),
    hstartPhi: phi.xstart, psistartPhi: phi.ystart, delhPhi: phi.dx, delpsiPhi: phi.dy,
    phiCoeff: splineCoeff2(phi.data, phi.dx, phi.dy, phiPoints),
    hstartUsemi: usemi.xstart, xistartUsemi: usemi.ystart, delhUsemi: usemi.dx, delxiUsemi: usemi.dy,
    usemiCoeff: splineCoeff2(usemi.data, usemi.dx, usemi.dy, usemiPoints),
  };
};

/** The 4x4 cubic Hermite basis matrix for spacing h (ax of spline_coeff). */
const axMatrix = (dx: number): number[][] => {
  const dxInv = 1.0 / dx, dxsqInv = dxInv * dxInv, dxcbInv = dxInv * dxsqInv;
  return [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [-3 * dxsqInv, -2 * dxInv, 3 * dxsqInv, -dxInv],
    [2 * dxcbInv, dxsqInv, -2 * dxcbInv, dxsqInv],
  ];
};

/** Off-diagonal and diagonal entries of the tridiagonal system of spline_coeff (b[i][i-1], b[i][i], b[i][i+1]). */
const tridiag = (h: number, n: number) => (i: number, j: number): number => {
  if (j === i - 1) return i > 1 ? h : 0;
  if (j === i) return 4 * h;
  if (j === i + 1) return i < n - 2 ? h : 0;
  return 0;
};

/**
 * PairMesoCNT::spline_coeff (1D): the coefficients of the cubic on each interval, flat with four values per
 * interval: row i (i = 1..n-1) holds g[i][0..3] at 4*i + k.
 */
export const splineCoeff1 = (u: Float64Array, dx: number, n: number): Float64Array => {
  const g = new Float64Array(4 * n);
  const p = new Float64Array(n);
  const b = tridiag(dx, n);
  const bprime = new Float64Array(n), dprime = new Float64Array(n);
  const dxInv = 1.0 / dx;
  p[0] = (u[1] - u[0]) * dxInv;
  p[n - 1] = (u[n - 1] - u[n - 2]) * dxInv;
  bprime[1] = b(1, 1);
  for (let i = 2; i < n - 1; i++) bprime[i] = b(i, i) - b(i, i - 1) * b(i - 1, i) / bprime[i - 1];
  for (let i = 1; i < n - 1; i++) {
    let d = 3 * (u[i + 1] - u[i - 1]);
    if (i === 1) d -= dx * p[i - 1];
    if (i === n - 2) d -= dx * p[i + 1];
    dprime[i] = d;
    if (i !== 1) dprime[i] -= b(i, i - 1) * dprime[i - 1] / bprime[i - 1];
  }
  p[n - 2] = dprime[n - 2] / bprime[n - 2];
  for (let i = n - 3; i > 0; i--) p[i] = (dprime[i] - b(i, i + 1) * p[i + 1]) / bprime[i];
  const ax = axMatrix(dx);
  for (let i = 1; i < n; i++) {
    const k = [u[i - 1], p[i - 1], u[i], p[i]];
    for (let j = 0; j < 4; j++) {
      let acc = 0;
      for (let l = 0; l < 4; l++) acc += ax[j][l] * k[l];
      g[4 * i + j] = acc;
    }
  }
  return g;
};

/**
 * PairMesoCNT::spline_coeff (2D): bicubic coefficients at [((i*n + j)*4 + l)*4 + m] for i, j = 1..n-1 (the cell with
 * lower corner (i-1, j-1)). The x sweep uses dx and the y sweep dy, as the C++ does.
 */
export const splineCoeff2 = (u: Float64Array, dx: number, dy: number, n: number): Float64Array => {
  const g = new Float64Array(n * n * 16);
  const p = new Float64Array(n * n), q = new Float64Array(n * n), s = new Float64Array(n * n);
  const U = (i: number, j: number): number => u[i * n + j];
  const bprime = new Float64Array(n), dprime = new Float64Array(n);
  const dxInv = 1.0 / dx, dyInv = 1.0 / dy;
  // boundary finite differences
  for (let i = 0; i < n; i++) {
    p[i] = (U(1, i) - U(0, i)) * dxInv;
    p[(n - 1) * n + i] = (U(n - 1, i) - U(n - 2, i)) * dxInv;
  }
  for (let i = 0; i < n; i++) {
    q[i * n] = (U(i, 1) - U(i, 0)) * dyInv;
    q[i * n + n - 1] = (U(i, n - 1) - U(i, n - 2)) * dyInv;
  }
  s[0] = (p[1] - p[0]) * dyInv;
  s[n - 1] = (p[n - 1] - p[n - 2]) * dyInv;
  s[(n - 1) * n] = (p[(n - 1) * n + 1] - p[(n - 1) * n]) * dyInv;
  s[(n - 1) * n + n - 1] = (p[(n - 1) * n + n - 1] - p[(n - 1) * n + n - 2]) * dyInv;
  // sweep in x (spacing dx): p along each column j
  let b = tridiag(dx, n);
  bprime[1] = b(1, 1);
  for (let i = 2; i < n - 1; i++) bprime[i] = b(i, i) - b(i, i - 1) * b(i - 1, i) / bprime[i - 1];
  // the forward sweep runs over rows (all columns j at once), so that memory is read in order
  const dp2 = new Float64Array(n * n);
  for (let i = 1; i < n - 1; i++) {
    for (let j = 0; j < n; j++) {
      let d = 3 * (U(i + 1, j) - U(i - 1, j));
      if (i === 1) d -= dx * p[(i - 1) * n + j];
      if (i === n - 2) d -= dx * p[(i + 1) * n + j];
      if (i !== 1) d -= b(i, i - 1) * dp2[(i - 1) * n + j] / bprime[i - 1];
      dp2[i * n + j] = d;
    }
  }
  for (let j = 0; j < n; j++) p[(n - 2) * n + j] = dp2[(n - 2) * n + j] / bprime[n - 2];
  for (let i = n - 3; i > 0; i--) {
    for (let j = 0; j < n; j++) p[i * n + j] = (dp2[i * n + j] - b(i, i + 1) * p[(i + 1) * n + j]) / bprime[i];
  }
  // s on the two boundary columns j = 0 and j = n-1 (x sweep, spacing dx)
  for (let j = 0; j < n; j += n - 1) {
    for (let i = 1; i < n - 1; i++) {
      let d = 3 * (q[(i + 1) * n + j] - q[(i - 1) * n + j]);
      if (i === 1) d -= dx * s[(i - 1) * n + j];
      if (i === n - 2) d -= dx * s[(i + 1) * n + j];
      dprime[i] = d;
      if (i !== 1) dprime[i] -= b(i, i - 1) * dprime[i - 1] / bprime[i - 1];
    }
    s[(n - 2) * n + j] = dprime[n - 2] / bprime[n - 2];
    for (let i = n - 3; i > 0; i--) s[i * n + j] = (dprime[i] - b(i, i + 1) * s[(i + 1) * n + j]) / bprime[i];
  }
  // sweep in y (spacing dy)
  b = tridiag(dy, n);
  bprime[1] = b(1, 1);
  for (let i = 2; i < n - 1; i++) bprime[i] = b(i, i) - b(i, i - 1) * b(i - 1, i) / bprime[i - 1];
  for (let i = 0; i < n; i++) {
    for (let j = 1; j < n - 1; j++) {
      let d = 3 * (U(i, j + 1) - U(i, j - 1));
      if (j === 1) d -= dy * q[i * n + j - 1];
      if (j === n - 2) d -= dy * q[i * n + j + 1];
      dprime[j] = d;
      if (j !== 1) dprime[j] -= b(j, j - 1) * dprime[j - 1] / bprime[j - 1];
    }
    q[i * n + n - 2] = dprime[n - 2] / bprime[n - 2];
    for (let j = n - 3; j > 0; j--) q[i * n + j] = (dprime[j] - b(j, j + 1) * q[i * n + j + 1]) / bprime[j];
  }
  for (let i = 0; i < n; i++) {
    for (let j = 1; j < n - 1; j++) {
      let d = 3 * (p[i * n + j + 1] - p[i * n + j - 1]);
      if (j === 1) d -= dy * s[i * n + j - 1];
      if (j === n - 2) d -= dy * s[i * n + j + 1];
      dprime[j] = d;
      if (j !== 1) dprime[j] -= b(j, j - 1) * dprime[j - 1] / bprime[j - 1];
    }
    s[i * n + n - 2] = dprime[n - 2] / bprime[n - 2];
    for (let j = n - 3; j > 0; j--) s[i * n + j] = (dprime[j] - b(j, j + 1) * s[i * n + j + 1]) / bprime[j];
  }
  const ax = axMatrix(dx), ay = axMatrix(dy);
  // g = ax K ay^T per cell, with K the 4x4 corner data (the C++ triple sum, reordered as two 4x4 products)
  const K = new Float64Array(16), T = new Float64Array(16);
  for (let i = 1; i < n; i++) {
    for (let j = 1; j < n; j++) {
      K[0] = U(i - 1, j - 1); K[1] = q[(i - 1) * n + j - 1]; K[2] = U(i - 1, j); K[3] = q[(i - 1) * n + j];
      K[4] = p[(i - 1) * n + j - 1]; K[5] = s[(i - 1) * n + j - 1]; K[6] = p[(i - 1) * n + j]; K[7] = s[(i - 1) * n + j];
      K[8] = U(i, j - 1); K[9] = q[i * n + j - 1]; K[10] = U(i, j); K[11] = q[i * n + j];
      K[12] = p[i * n + j - 1]; K[13] = s[i * n + j - 1]; K[14] = p[i * n + j]; K[15] = s[i * n + j];
      // T[l][o] = sum_a ax[l][a] K[a][o]
      for (let l = 0; l < 4; l++) {
        for (let o = 0; o < 4; o++) {
          T[4 * l + o] = ax[l][0] * K[o] + ax[l][1] * K[4 + o] + ax[l][2] * K[8 + o] + ax[l][3] * K[12 + o];
        }
      }
      const base = (i * n + j) * 16;
      // g[l][m] = sum_o T[l][o] ay[m][o]
      for (let l = 0; l < 4; l++) {
        for (let m = 0; m < 4; m++) {
          g[base + l * 4 + m] = T[4 * l] * ay[m][0] + T[4 * l + 1] * ay[m][1] + T[4 * l + 2] * ay[m][2] + T[4 * l + 3] * ay[m][3];
        }
      }
    }
  }
  return g;
};

/** Heaviside step: 1 for x > 0, else 0 (pair_mesocnt.h). */
const heaviside = (x: number): number => (x > 0 ? 1.0 : 0.0);
/** Quintic switch s5 (pair_mesocnt.h). */
export const s5 = (x: number): number => {
  const x2 = x * x;
  return heaviside(-x) + heaviside(x) * heaviside(1 - x) * (1 - x2 * x * (6 * x2 - 15 * x + 10));
};
/** Derivative of s5 (pair_mesocnt.h). */
export const ds5 = (x: number): number => {
  const x2 = x * x;
  return -30 * heaviside(x) * heaviside(1 - x) * x2 * (x2 - 2 * x + 1);
};
/** Cubic switch s (pair_mesocnt.h). */
export const sCubic = (x: number): number => heaviside(-x) + heaviside(x) * heaviside(1 - x) * (1 - x * x * (3 - 2 * x));
/** Derivative of the cubic switch ds (pair_mesocnt.h). */
export const dsCubic = (x: number): number => 6 * heaviside(x) * heaviside(1 - x) * x * (x - 1);

/** PairMesoCNT::spline (1D): value, linear extrapolation below the table, constant above it. */
export const spline1 = (x0: number, xstart: number, dx: number, c: Float64Array, size: number): number => {
  let i = Math.ceil((x0 - xstart) / dx);
  let x = x0;
  if (i < 1) return c[4] + c[5] * (x - xstart);
  if (i > size - 1) {
    i = size - 1;
    x = xstart + (size - 1) * dx;
  }
  const xlo = xstart + (i - 1) * dx;
  const xbar = x - xlo;
  return c[4 * i] + xbar * (c[4 * i + 1] + xbar * (c[4 * i + 2] + xbar * c[4 * i + 3]));
};

/** PairMesoCNT::dspline (1D): derivative, constant below the table. */
export const dspline1 = (x0: number, xstart: number, dx: number, c: Float64Array, size: number): number => {
  let i = Math.ceil((x0 - xstart) / dx);
  let x = x0;
  if (i < 1) return c[5];
  if (i > size - 1) {
    i = size - 1;
    x = xstart + (size - 1) * dx;
  }
  const xlo = xstart + (i - 1) * dx;
  const xbar = x - xlo;
  return c[4 * i + 1] + xbar * (2 * c[4 * i + 2] + 3 * xbar * c[4 * i + 3]);
};

/** The clamped cell (i, j) and clamped x, y of a bicubic table (the constant extrapolation of the C++). */
const cell2 = (
  x0: number, y0: number, xstart: number, ystart: number, dx: number, dy: number, size: number,
): { i: number; j: number; x: number; y: number } => {
  let i = Math.ceil((x0 - xstart) / dx);
  let j = Math.ceil((y0 - ystart) / dy);
  let x = x0, y = y0;
  if (i < 1) {
    i = 1;
    x = xstart;
  } else if (i > size - 1) {
    i = size - 1;
    x = xstart + (size - 1) * dx;
  }
  if (j < 1) {
    j = 1;
    y = ystart;
  } else if (j > size - 1) {
    j = size - 1;
    y = ystart + (size - 1) * dy;
  }
  return { i, j, x, y };
};

/** PairMesoCNT::spline (2D): bicubic value. */
export const spline2 = (
  x0: number, y0: number, xstart: number, ystart: number, dx: number, dy: number, c: Float64Array, size: number,
): number => {
  const { i, j, x, y } = cell2(x0, y0, xstart, ystart, dx, dy, size);
  const xbar = x - (xstart + (i - 1) * dx), ybar = y - (ystart + (j - 1) * dy);
  const base = (i * size + j) * 16;
  const yl = (l: number): number => {
    const o = base + l * 4;
    return c[o] + ybar * (c[o + 1] + ybar * (c[o + 2] + ybar * c[o + 3]));
  };
  const y0v = yl(0), y1v = yl(1), y2v = yl(2), y3v = yl(3);
  return y0v + xbar * (y1v + xbar * (y2v + xbar * y3v));
};

/** PairMesoCNT::dxspline: bicubic partial derivative in x. */
export const dxspline2 = (
  x0: number, y0: number, xstart: number, ystart: number, dx: number, dy: number, c: Float64Array, size: number,
): number => {
  const { i, j, x, y } = cell2(x0, y0, xstart, ystart, dx, dy, size);
  const xbar = x - (xstart + (i - 1) * dx), ybar = y - (ystart + (j - 1) * dy);
  const base = (i * size + j) * 16;
  const yl = (l: number): number => {
    const o = base + l * 4;
    return c[o] + ybar * (c[o + 1] + ybar * (c[o + 2] + ybar * c[o + 3]));
  };
  const y1 = yl(1), y2 = yl(2), y3 = yl(3);
  return y1 + xbar * (2 * y2 + 3 * xbar * y3);
};

/** PairMesoCNT::dyspline: bicubic partial derivative in y. */
export const dyspline2 = (
  x0: number, y0: number, xstart: number, ystart: number, dx: number, dy: number, c: Float64Array, size: number,
): number => {
  const { i, j, x, y } = cell2(x0, y0, xstart, ystart, dx, dy, size);
  const xbar = x - (xstart + (i - 1) * dx), ybar = y - (ystart + (j - 1) * dy);
  const base = (i * size + j) * 16;
  const yd = (l: number): number => {
    const o = base + l * 4;
    return c[o + 1] + ybar * (2 * c[o + 2] + 3 * ybar * c[o + 3]);
  };
  const y0v = yd(0), y1v = yd(1), y2v = yd(2), y3v = yd(3);
  return y0v + xbar * (y1v + xbar * (y2v + xbar * y3v));
};

/** PairMesoCNT::legendre: the n-th Legendre polynomial at x (the three-term recurrence, as the C++). */
export const legendre = (n: number, x: number): number => {
  if (n === 0) return 1.0;
  if (n === 1) return x;
  const lc = new Float64Array(n + 1);
  lc[0] = 1.0;
  lc[1] = x;
  for (let i = 2; i <= n; i++) lc[i] = ((2 * i - 1) * x * lc[i - 1] - (i - 1) * lc[i - 2]) / i;
  return lc[n];
};

const BISECTION_STEPS = 1000000;
const BISECTION_EPS = 1.0e-15;

/** PairMesoCNT::gl_init_nodes: the roots of the quad-th Legendre polynomial (bisection, as the C++). */
export const glNodes = (quad: number): Float64Array => {
  const nodes = new Float64Array(quad);
  let kStart: number, kEnd: number, kOffset: number;
  if (quad % 2) {
    kStart = 1;
    kEnd = (quad - 1) / 2 + 1;
    kOffset = 2;
    nodes[kEnd - 1] = 0.0;
  } else {
    kStart = 0;
    kEnd = quad / 2;
    kOffset = 1;
  }
  let root = 0;
  for (let k = kStart; k < kEnd; k++) {
    const theta = (Math.ceil(0.5 * quad) - 0.25 - k) * Math.PI / (quad + 0.5);
    let a = libmCos((Math.ceil(0.5 * quad) - k) * Math.PI / (quad + 1.0));
    let b = libmCos(theta);
    let c = 0;
    let iter = 0;
    do {
      c = 0.5 * (a + b);
      if (legendre(quad, c) === 0.0) break;
      if (legendre(quad, a) * legendre(quad, c) < 0) b = c;
      else a = c;
      iter++;
    } while (Math.abs(a - b) >= BISECTION_EPS && iter <= BISECTION_STEPS);
    nodes[kEnd + root] = c;
    nodes[kEnd - root - kOffset] = -c;
    root++;
  }
  return nodes;
};

/** PairMesoCNT::gl_init_weights: the Gauss-Legendre weights of the nodes from glNodes. */
export const glWeights = (quad: number, nodes: Float64Array): Float64Array => {
  const w = new Float64Array(quad);
  for (let i = 0; i < quad; i++) {
    const x = nodes[i];
    const dlegendre = quad * (x * legendre(quad, x) - legendre(quad - 1, x)) / (x * x - 1.0);
    w[i] = 2.0 / ((1.0 - x * x) * dlegendre * dlegendre);
  }
  return w;
};
