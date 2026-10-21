/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/amoeba_kspace.cpp (lattice, moduli, bspline, dftmod, bspline_fill, bsplgen, cmp_to_fmp,
 * cart_to_frac, fphi_to_cphi, frac_to_cart, grid_mpole, fphi_mpole, kewald's factorable), src/AMOEBA/amoeba_multipole.cpp
 * (multipole_kspace) and src/AMOEBA/amoeba_convolution.cpp (the forward FFT scaled by 1/N, the backward FFT unscaled)
 * at commit b332f28. AMOEBA package authors: the LAMMPS developers (Sandia Corporation, GNU General Public License).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Smooth particle-mesh Ewald for the permanent atomic multipoles of pair amoeba (multipole_kspace): the fractional
 * multipoles, the B-spline moments of order bsorder on an nfft grid (bspline_fill, bsplgen), the grid, the 3d FFT with
 * the convolution by the Ewald influence function (bsmod moduli from dftmod), the potential gathered back with the B-spline
 * derivatives (fphi_mpole), the energy, the forces and the torques on the sites. The grid is a single periodic grid (one
 * process): the native ghost-cell reduction and the periodic wrap are the same indices modulo nfft. Both FFTs are
 * unnormalized (amoeba_convolution.cpp: the 1/N scaling is compiled out, SCALE 0).
 * Literature (amoeba_multipole.cpp): C. Sagui, L. G. Pedersen and T. A. Darden, J. Chem. Phys. 120, 73 (2004).
 */
import { StyleError } from '../../engine/force/types';
import { fft3d } from '../../engine/force/fft';
import { powint } from './hal';
import { ELECTRIC, torque2force, type MpoleSites } from './multipole';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';

/** pi (amoeba_kspace.cpp MY_PI). */
const MY_PI = Math.PI;

/** Grid sizes factorable by 2, 3 and 5 (pair_amoeba.cpp init: factors[] = {2, 3, 5}). */
export const factorable = (n: number): boolean => {
  let m = n;
  while (m > 1) {
    let i = 0;
    const factors = [2, 3, 5];
    for (; i < factors.length; i++) {
      if (m % factors[i] === 0) {
        m /= factors[i];
        break;
      }
    }
    if (i === factors.length) return false;
  }
  return true;
};

/** Grid size from the box length as kewald does without pme-grid: int(L * 1.2 - 1e-8) + 1, rounded up to factorable. */
export const defaultGridSize = (prd: number): number => {
  let n = Math.trunc(prd * 1.2 - 1.0e-8) + 1;
  while (!factorable(n)) n++;
  return n;
};

/** bspline (amoeba_kspace.cpp): the B-spline values c[0..n-1] of order n at x. */
const bspline = (x: number, n: number, c: Float64Array): void => {
  c[0] = 1.0 - x;
  c[1] = x;
  for (let k = 3; k <= n; k++) {
    const denom = 1.0 / (k - 1);
    c[k - 1] = (x * c[k - 2]) * denom;
    for (let i = 1; i <= k - 2; i++) {
      c[k - i - 1] = ((x + i) * c[k - i - 2] + (k - i - x) * c[k - i - 1]) * denom;
    }
    c[0] = ((1.0 - x) * c[0]) * denom;
  }
};

/** dftmod (amoeba_kspace.cpp): the squared modulus of the DFT of the B-spline, with the zeta correction. */
const dftmod = (bsmod: Float64Array, bsarray: Float64Array, nfft: number, order: number): void => {
  let factor = (2.0 * MY_PI) / nfft;
  for (let i = 0; i < nfft; i++) {
    let sum1 = 0.0, sum2 = 0.0;
    for (let j = 0; j < nfft; j++) {
      const arg = factor * i * j;
      sum1 += bsarray[j] * libmCos(arg);
      sum2 += bsarray[j] * libmSin(arg);
    }
    bsmod[i] = sum1 * sum1 + sum2 * sum2;
  }
  const eps = 1.0e-7;
  if (bsmod[0] < eps) bsmod[0] = 0.5 * bsmod[1];
  for (let i = 1; i < nfft - 1; i++) if (bsmod[i] < eps) bsmod[i] = 0.5 * (bsmod[i - 1] + bsmod[i + 1]);
  if (bsmod[nfft - 1] < eps) bsmod[nfft - 1] = 0.5 * bsmod[nfft - 2];
  const jcut = 50;
  const order2 = 2 * order;
  const half = Math.floor(nfft / 2);
  for (let i = 1; i <= nfft; i++) {
    let k = i - 1;
    if (i > half) k -= nfft;
    let zeta: number;
    if (k === 0) zeta = 1.0;
    else {
      let sum1 = 1.0, sum2 = 1.0;
      factor = (MY_PI * k) / nfft;
      for (let j = 1; j <= jcut; j++) {
        const arg = factor / (factor + MY_PI * j);
        sum1 += powint(arg, order);
        sum2 += powint(arg, order2);
      }
      for (let j = 1; j <= jcut; j++) {
        const arg = factor / (factor - MY_PI * j);
        sum1 += powint(arg, order);
        sum2 += powint(arg, order2);
      }
      zeta = sum2 / sum1;
    }
    bsmod[i - 1] *= zeta * zeta;
  }
};

/** moduli (amoeba_kspace.cpp): the inverse-DFT moduli of the B-spline along each grid direction. */
export const pmeModuli = (nfft: readonly [number, number, number], order: number): Float64Array[] => {
  const maxfft = Math.max(nfft[0], nfft[1], nfft[2]);
  const arr = new Float64Array(order);
  bspline(0.0, order, arr);
  const bsarray = new Float64Array(maxfft);
  for (let i = 0; i < order; i++) bsarray[i + 1] = arr[i];
  return nfft.map((nf) => {
    const out = new Float64Array(nf);
    dftmod(out, bsarray, nf, order);
    return out;
  });
};

/**
 * bsplgen (amoeba_kspace.cpp): the B-spline values and the first three derivatives of an atom at fractional offset w,
 * for the order-bsorder spline, as thetai[i*4 + j] (i the spline point, j the derivative order 0..3).
 */
const bsplgen = (w: number, bsorder: number, thetai: Float64Array, off: number): void => {
  const level = 4;
  const bsbuild: number[][] = [];
  for (let i = 0; i < bsorder; i++) bsbuild.push(new Array<number>(bsorder).fill(0.0));
  bsbuild[1][1] = w;
  bsbuild[0][1] = 1.0 - w;
  bsbuild[2][2] = 0.5 * w * bsbuild[1][1];
  bsbuild[1][2] = 0.5 * ((1.0 + w) * bsbuild[0][1] + (2.0 - w) * bsbuild[1][1]);
  bsbuild[0][2] = 0.5 * (1.0 - w) * bsbuild[0][1];
  for (let i = 4; i <= bsorder; i++) {
    const k = i - 1;
    const denom = 1.0 / k;
    bsbuild[i - 1][i - 1] = denom * w * bsbuild[k - 1][k - 1];
    for (let j = 1; j <= i - 2; j++) {
      bsbuild[i - j - 1][i - 1] = denom * ((w + j) * bsbuild[i - j - 2][k - 1] + (i - j - w) * bsbuild[i - j - 1][k - 1]);
    }
    bsbuild[0][i - 1] = denom * (1.0 - w) * bsbuild[0][k - 1];
  }
  let k = bsorder - 1;
  bsbuild[bsorder - 1][k - 1] = bsbuild[bsorder - 2][k - 1];
  for (let i = bsorder - 1; i >= 2; i--) bsbuild[i - 1][k - 1] = bsbuild[i - 2][k - 1] - bsbuild[i - 1][k - 1];
  bsbuild[0][k - 1] = -bsbuild[0][k - 1];
  if (level === 4) {
    k = bsorder - 2;
    bsbuild[bsorder - 2][k - 1] = bsbuild[bsorder - 3][k - 1];
    for (let i = bsorder - 2; i >= 2; i--) bsbuild[i - 1][k - 1] = bsbuild[i - 2][k - 1] - bsbuild[i - 1][k - 1];
    bsbuild[0][k - 1] = -bsbuild[0][k - 1];
    bsbuild[bsorder - 1][k - 1] = bsbuild[bsorder - 2][k - 1];
    for (let i = bsorder - 1; i >= 2; i--) bsbuild[i - 1][k - 1] = bsbuild[i - 2][k - 1] - bsbuild[i - 1][k - 1];
    bsbuild[0][k - 1] = -bsbuild[0][k - 1];
    k = bsorder - 3;
    bsbuild[bsorder - 3][k - 1] = bsbuild[bsorder - 4][k - 1];
    for (let i = bsorder - 3; i >= 2; i--) bsbuild[i - 1][k - 1] = bsbuild[i - 2][k - 1] - bsbuild[i - 1][k - 1];
    bsbuild[0][k - 1] = -bsbuild[0][k - 1];
    bsbuild[bsorder - 2][k - 1] = bsbuild[bsorder - 3][k - 1];
    for (let i = bsorder - 2; i >= 2; i--) bsbuild[i - 1][k - 1] = bsbuild[i - 2][k - 1] - bsbuild[i - 1][k - 1];
    bsbuild[0][k - 1] = -bsbuild[0][k - 1];
    bsbuild[bsorder - 1][k - 1] = bsbuild[bsorder - 2][k - 1];
    for (let i = bsorder - 1; i >= 2; i--) bsbuild[i - 1][k - 1] = bsbuild[i - 2][k - 1] - bsbuild[i - 1][k - 1];
    bsbuild[0][k - 1] = -bsbuild[0][k - 1];
  }
  for (let i = 1; i <= bsorder; i++) {
    for (let j = 1; j <= level; j++) thetai[off + (i - 1) * 4 + (j - 1)] = bsbuild[i - 1][bsorder - j];
  }
};

/** The PME grid of one step: sizes, B-spline order, moduli, and the reciprocal lattice (diagonal for an orthogonal box). */
export interface PmeGrid {
  nfft: [number, number, number];
  order: number;
  bsmod: Float64Array[];
  /** Box origin and lengths (orthogonal box). */
  lo: [number, number, number];
  prd: [number, number, number];
  /** Reciprocal lattice (lattice(): recip = h_inv), 3x3 row-major. */
  recip: Float64Array;
}

export const makePmeGrid = (
  nfft: [number, number, number], order: number, lo: [number, number, number], prd: [number, number, number],
): PmeGrid => {
  const recip = new Float64Array(9);
  for (let d = 0; d < 3; d++) recip[4 * d] = 1.0 / prd[d];
  return { nfft, order, bsmod: pmeModuli(nfft, order), lo, prd, recip };
};

/** bspline_fill (amoeba_kspace.cpp): the grid index of each atom and its B-spline coefficients in each direction. */
export interface PmeFill {
  /** igrid per atom (3 values). */
  igrid: Int32Array;
  /** thetai per direction, n * order * 4 values. */
  theta: [Float64Array, Float64Array, Float64Array];
}

export const pmeFill = (x: Float64Array, n: number, g: PmeGrid): PmeFill => {
  const eps = 1.0e-8;
  const igrid = new Int32Array(3 * n);
  const theta: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(n * g.order * 4), new Float64Array(n * g.order * 4), new Float64Array(n * g.order * 4),
  ];
  const stride = g.order * 4;
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < 3; d++) {
      // domain->x2lamda: lamda = h_inv (x - lo). Native wraps owned atoms only on reneighboring steps, so between them
      // lamda can lie outside [0, 1); the index is then the truncated one (bspline_fill) and the grid index wraps periodically
      const lam = g.recip[4 * d] * (x[3 * i + d] - g.lo[d]);
      const fr = g.nfft[d] * lam;
      const ifr = Math.trunc(fr - eps);
      igrid[3 * i + d] = ifr;
      bsplgen(fr - ifr, g.order, theta[d], i * stride);
    }
  }
  return { igrid, theta };
};

/** The condensed multipole order of cmp (amoeba_kspace.cpp): m, dx, dy, dz, qxx, qyy, qzz, 2qxy, 2qxz, 2qyz. */
const CMP_FROM_RPOLE = [0, 1, 2, 3, 4, 8, 12, 5, 6, 9];

const QI1 = [0, 1, 2, 0, 0, 1];
const QI2 = [0, 1, 2, 1, 2, 2];

/** cart_to_frac (amoeba_kspace.cpp): the 10x10 transform from Cartesian to fractional multipoles. */
const cartToFrac = (g: PmeGrid): number[][] => {
  const a: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let d = 0; d < 3; d++) a[d][i] = g.nfft[d] * g.recip[3 * i + d];
  const ctf: number[][] = Array.from({ length: 10 }, () => new Array<number>(10).fill(0.0));
  ctf[0][0] = 1.0;
  for (let i = 1; i < 4; i++) for (let j = 1; j < 4; j++) ctf[i][j] = a[i - 1][j - 1];
  for (let i1 = 0; i1 < 3; i1++) {
    const k = QI1[i1];
    for (let i2 = 0; i2 < 6; i2++) {
      const i = QI1[i2], j = QI2[i2];
      ctf[i1 + 4][i2 + 4] = a[k][i] * a[k][j];
    }
  }
  for (let i1 = 3; i1 < 6; i1++) {
    const k = QI1[i1], m = QI2[i1];
    for (let i2 = 0; i2 < 6; i2++) {
      const i = QI1[i2], j = QI2[i2];
      ctf[i1 + 4][i2 + 4] = a[k][i] * a[m][j] + a[k][j] * a[m][i];
    }
  }
  return ctf;
};

/** frac_to_cart (amoeba_kspace.cpp): the 10x10 transform from fractional to Cartesian potentials. */
const fracToCart = (g: PmeGrid): number[][] => {
  const a: number[][] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let d = 0; d < 3; d++) a[i][d] = g.nfft[d] * g.recip[3 * i + d];
  const ftc: number[][] = Array.from({ length: 10 }, () => new Array<number>(10).fill(0.0));
  ftc[0][0] = 1.0;
  for (let i = 1; i < 4; i++) for (let j = 1; j < 4; j++) ftc[i][j] = a[i - 1][j - 1];
  for (let i1 = 0; i1 < 3; i1++) {
    const k = QI1[i1];
    for (let i2 = 0; i2 < 3; i2++) {
      const i = QI1[i2];
      ftc[i1 + 4][i2 + 4] = a[k][i] * a[k][i];
    }
    for (let i2 = 3; i2 < 6; i2++) {
      const i = QI1[i2], j = QI2[i2];
      ftc[i1 + 4][i2 + 4] = 2.0 * a[k][i] * a[k][j];
    }
  }
  for (let i1 = 3; i1 < 6; i1++) {
    const k = QI1[i1], m = QI2[i1];
    for (let i2 = 0; i2 < 3; i2++) {
      const i = QI1[i2];
      ftc[i1 + 4][i2 + 4] = a[k][i] * a[m][i];
    }
    for (let i2 = 3; i2 < 6; i2++) {
      const i = QI1[i2], j = QI2[i2];
      ftc[i1 + 4][i2 + 4] = a[k][i] * a[m][j] + a[m][i] * a[k][j];
    }
  }
  return ftc;
};

/** Output of the permanent multipole reciprocal sum. */
export interface MpoleKspaceResult {
  e: number;
  virial: Float64Array;
}

/** cmp (condensed Cartesian, 10 per site) from the rotated multipoles rpole (13 per site), as multipole_kspace builds it. */
export const cmpFromRpole = (rp: Float64Array, n: number): Float64Array => {
  const cmp = new Float64Array(10 * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 7; k++) cmp[10 * i + k] = rp[13 * i + CMP_FROM_RPOLE[k]];
    cmp[10 * i + 7] = 2.0 * rp[13 * i + 5];
    cmp[10 * i + 8] = 2.0 * rp[13 * i + 6];
    cmp[10 * i + 9] = 2.0 * rp[13 * i + 9];
  }
  return cmp;
};

/** cmp_to_fmp (amoeba_kspace.cpp): the fractional multipoles (10 per site) of the grid g. */
export const cmpToFmp = (cmp: Float64Array, n: number, g: PmeGrid): Float64Array => {
  const ctf = cartToFrac(g);
  const fmp = new Float64Array(10 * n);
  for (let i = 0; i < n; i++) {
    const o = 10 * i;
    fmp[o] = ctf[0][0] * cmp[o];
    for (let j = 1; j < 4; j++) {
      fmp[o + j] = 0.0;
      for (let k = 1; k < 4; k++) fmp[o + j] += ctf[j][k] * cmp[o + k];
    }
    for (let j = 4; j < 10; j++) {
      fmp[o + j] = 0.0;
      for (let k = 4; k < 10; k++) fmp[o + j] += ctf[j][k] * cmp[o + k];
    }
  }
  return fmp;
};

/** fphi_to_cphi (amoeba_kspace.cpp): the Cartesian potentials (10 per site) from the 20 fractional terms. */
export const fphiToCphi = (fphi: Float64Array, n: number, g: PmeGrid): Float64Array => {
  const ftc = fracToCart(g);
  const cphi = new Float64Array(10 * n);
  for (let i = 0; i < n; i++) {
    const o = 10 * i, p = 20 * i;
    cphi[o] = ftc[0][0] * fphi[p];
    for (let j = 1; j < 4; j++) {
      cphi[o + j] = 0.0;
      for (let k = 1; k < 4; k++) cphi[o + j] += ftc[j][k] * fphi[p + k];
    }
    for (let j = 4; j < 10; j++) {
      cphi[o + j] = 0.0;
      for (let k = 4; k < 10; k++) cphi[o + j] += ftc[j][k] * fphi[p + k];
    }
  }
  return cphi;
};

/** grid_mpole (amoeba_kspace.cpp): spreads the fractional moments (10 per site) onto the periodic real grid. */
export const spreadMpole = (fmp: Float64Array, n: number, fill: PmeFill, g: PmeGrid): Float64Array => {
  const [nfft1, nfft2, nfft3] = g.nfft;
  const order = g.order;
  const nlpts = Math.floor((order - 1) / 2);
  const grid = new Float64Array(nfft1 * nfft2 * nfft3);
  const wrap = (v: number, nf: number): number => ((v % nf) + nf) % nf;
  const th = fill.theta, ig = fill.igrid;
  for (let m = 0; m < n; m++) {
    const fm = (k: number): number => fmp[10 * m + k];
    let kz = ig[3 * m + 2] - nlpts;
    for (let kb = 0; kb < order; kb++) {
      const v0 = th[2][m * order * 4 + kb * 4], v1 = th[2][m * order * 4 + kb * 4 + 1], v2 = th[2][m * order * 4 + kb * 4 + 2];
      let jy = ig[3 * m + 1] - nlpts;
      for (let jb = 0; jb < order; jb++) {
        const u0 = th[1][m * order * 4 + jb * 4], u1 = th[1][m * order * 4 + jb * 4 + 1], u2 = th[1][m * order * 4 + jb * 4 + 2];
        const term0 = fm(0) * u0 * v0 + fm(2) * u1 * v0 + fm(3) * u0 * v1 + fm(5) * u2 * v0 + fm(6) * u0 * v2 + fm(9) * u1 * v1;
        const term1 = fm(1) * u0 * v0 + fm(7) * u1 * v0 + fm(8) * u0 * v1;
        const term2 = fm(4) * u0 * v0;
        let ix = ig[3 * m] - nlpts;
        for (let ib = 0; ib < order; ib++) {
          const t0 = th[0][m * order * 4 + ib * 4], t1 = th[0][m * order * 4 + ib * 4 + 1], t2 = th[0][m * order * 4 + ib * 4 + 2];
          grid[(wrap(kz, nfft3) * nfft2 + wrap(jy, nfft2)) * nfft1 + wrap(ix, nfft1)] += term0 * t0 + term1 * t1 + term2 * t2;
          ix++;
        }
        jy++;
      }
      kz++;
    }
  }
  return grid;
};

/** fphi_mpole (amoeba_kspace.cpp): the 20 potential terms (value and derivatives) per site from the real grid. */
export const gatherMpole = (pgrid: Float64Array, n: number, fill: PmeFill, g: PmeGrid): Float64Array => {
  const [nfft1, nfft2, nfft3] = g.nfft;
  const order = g.order;
  const nlpts = Math.floor((order - 1) / 2);
  const wrap = (v: number, nf: number): number => ((v % nf) + nf) % nf;
  const th = fill.theta, ig = fill.igrid;
  const fphi = new Float64Array(20 * n);
  for (let m = 0; m < n; m++) {
    const tuv = new Float64Array(20);
    let kz = ig[3 * m + 2] - nlpts;
    for (let kb = 0; kb < order; kb++) {
      const v0 = th[2][m * order * 4 + kb * 4], v1 = th[2][m * order * 4 + kb * 4 + 1];
      const v2 = th[2][m * order * 4 + kb * 4 + 2], v3 = th[2][m * order * 4 + kb * 4 + 3];
      const tu = new Float64Array(10); // tu00, tu10, tu01, tu20, tu11, tu02, tu30, tu21, tu12, tu03
      let jy = ig[3 * m + 1] - nlpts;
      for (let jb = 0; jb < order; jb++) {
        const u0 = th[1][m * order * 4 + jb * 4], u1 = th[1][m * order * 4 + jb * 4 + 1];
        const u2 = th[1][m * order * 4 + jb * 4 + 2], u3 = th[1][m * order * 4 + jb * 4 + 3];
        let t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
        let ix = ig[3 * m] - nlpts;
        for (let ib = 0; ib < order; ib++) {
          const tq = pgrid[(wrap(kz, nfft3) * nfft2 + wrap(jy, nfft2)) * nfft1 + wrap(ix, nfft1)];
          t0 += tq * th[0][m * order * 4 + ib * 4];
          t1 += tq * th[0][m * order * 4 + ib * 4 + 1];
          t2 += tq * th[0][m * order * 4 + ib * 4 + 2];
          t3 += tq * th[0][m * order * 4 + ib * 4 + 3];
          ix++;
        }
        tu[0] += t0 * u0;
        tu[1] += t1 * u0;
        tu[2] += t0 * u1;
        tu[3] += t2 * u0;
        tu[4] += t1 * u1;
        tu[5] += t0 * u2;
        tu[6] += t3 * u0;
        tu[7] += t2 * u1;
        tu[8] += t1 * u2;
        tu[9] += t0 * u3;
        jy++;
      }
      tuv[0] += tu[0] * v0;
      tuv[1] += tu[1] * v0;
      tuv[2] += tu[2] * v0;
      tuv[3] += tu[0] * v1;
      tuv[4] += tu[3] * v0;
      tuv[5] += tu[5] * v0;
      tuv[6] += tu[0] * v2;
      tuv[7] += tu[4] * v0;
      tuv[8] += tu[1] * v1;
      tuv[9] += tu[2] * v1;
      tuv[10] += tu[6] * v0;
      tuv[11] += tu[9] * v0;
      tuv[12] += tu[0] * v3;
      tuv[13] += tu[7] * v0;
      tuv[14] += tu[3] * v1;
      tuv[15] += tu[8] * v0;
      tuv[16] += tu[5] * v1;
      tuv[17] += tu[1] * v2;
      tuv[18] += tu[2] * v2;
      tuv[19] += tu[4] * v1;
      kz++;
    }
    for (let k = 0; k < 20; k++) fphi[20 * m + k] = tuv[k];
  }
  return fphi;
};

/**
 * The influence function of the PME sums (expterm of amoeba_induce.cpp udirect1 and umutual1, and of multipole_kspace):
 * exp(-pi^2 h^2 / aewald^2) / (pi V h^2 bsmod1 bsmod2 bsmod3) on the grid, 0 at h = 0 and where exp underflows.
 */
export const influence = (g: PmeGrid, aewald: number): Float64Array => {
  const [nfft1, nfft2, nfft3] = g.nfft;
  const volbox = g.prd[0] * g.prd[1] * g.prd[2];
  const volterm = MY_PI * volbox;
  const pterm = (MY_PI / aewald) * (MY_PI / aewald);
  const nhalf1 = Math.floor((nfft1 + 1) / 2), nhalf2 = Math.floor((nfft2 + 1) / 2), nhalf3 = Math.floor((nfft3 + 1) / 2);
  const bs1 = g.bsmod[0], bs2 = g.bsmod[1], bs3 = g.bsmod[2];
  const w = new Float64Array(nfft1 * nfft2 * nfft3);
  let nn = 0;
  for (let k = 0; k < nfft3; k++) {
    for (let j = 0; j < nfft2; j++) {
      for (let i = 0; i < nfft1; i++) {
        const r1 = i >= nhalf1 ? i - nfft1 : i;
        const r2 = j >= nhalf2 ? j - nfft2 : j;
        const r3 = k >= nhalf3 ? k - nfft3 : k;
        const h1 = g.recip[0] * r1 + g.recip[1] * r2 + g.recip[2] * r3;
        const h2 = g.recip[3] * r1 + g.recip[4] * r2 + g.recip[5] * r3;
        const h3 = g.recip[6] * r1 + g.recip[7] * r2 + g.recip[8] * r3;
        const hsq = h1 * h1 + h2 * h2 + h3 * h3;
        const term = -pterm * hsq;
        if (term > -50.0 && hsq !== 0.0) {
          const denom = volterm * hsq * bs1[i] * bs2[j] * bs3[k];
          w[nn] = libmExp(term) / denom;
        }
        nn++;
      }
    }
  }
  return w;
};

/**
 * Forward FFT of a real grid (unscaled), multiplication by the influence function w, backward FFT (unscaled). Returns the
 * forward transform (for the virial of the permanent sum) and the real part of the result.
 */
export const convolveReal = (grid: Float64Array, g: PmeGrid, w: Float64Array): { fwd: Float64Array; pot: Float64Array } => {
  const [nfft1, nfft2, nfft3] = g.nfft;
  const N = nfft1 * nfft2 * nfft3;
  const c = new Float64Array(2 * N);
  for (let i = 0; i < N; i++) c[2 * i] = grid[i];
  fft3d(c, nfft1, nfft2, nfft3, -1);
  const fwd = Float64Array.from(c);
  for (let i = 0; i < N; i++) {
    c[2 * i] *= w[i];
    c[2 * i + 1] *= w[i];
  }
  fft3d(c, nfft1, nfft2, nfft3, 1);
  const pot = new Float64Array(N);
  for (let i = 0; i < N; i++) pot[i] = c[2 * i];
  return { fwd, pot };
};

/**
 * grid_uind (amoeba_kspace.cpp): the fractional induced dipoles (uind, uinp) spread onto a complex grid, uind in the real
 * part and uinp in the imaginary part (the two channels of the 4d INDUCE_GRIDC array).
 */
export const spreadUind = (fu: Float64Array, fup: Float64Array, n: number, fill: PmeFill, g: PmeGrid): Float64Array => {
  const [nfft1, nfft2, nfft3] = g.nfft;
  const order = g.order;
  const nlpts = Math.floor((order - 1) / 2);
  const wrap = (v: number, nf: number): number => ((v % nf) + nf) % nf;
  const th = fill.theta, ig = fill.igrid;
  const c = new Float64Array(2 * nfft1 * nfft2 * nfft3);
  for (let m = 0; m < n; m++) {
    let kz = ig[3 * m + 2] - nlpts;
    for (let kb = 0; kb < order; kb++) {
      const v0 = th[2][m * order * 4 + kb * 4], v1 = th[2][m * order * 4 + kb * 4 + 1];
      let jy = ig[3 * m + 1] - nlpts;
      for (let jb = 0; jb < order; jb++) {
        const u0 = th[1][m * order * 4 + jb * 4], u1 = th[1][m * order * 4 + jb * 4 + 1];
        const term01 = fu[3 * m + 1] * u1 * v0 + fu[3 * m + 2] * u0 * v1;
        const term11 = fu[3 * m] * u0 * v0;
        const term02 = fup[3 * m + 1] * u1 * v0 + fup[3 * m + 2] * u0 * v1;
        const term12 = fup[3 * m] * u0 * v0;
        let ix = ig[3 * m] - nlpts;
        for (let ib = 0; ib < order; ib++) {
          const t0 = th[0][m * order * 4 + ib * 4], t1 = th[0][m * order * 4 + ib * 4 + 1];
          const idx = (wrap(kz, nfft3) * nfft2 + wrap(jy, nfft2)) * nfft1 + wrap(ix, nfft1);
          c[2 * idx] += term01 * t0 + term11 * t1;
          c[2 * idx + 1] += term02 * t0 + term12 * t1;
          ix++;
        }
        jy++;
      }
      kz++;
    }
  }
  return c;
};

/** Forward FFT, multiplication by the influence function, backward FFT, on the complex grid c (unscaled). */
export const convolveComplex = (c: Float64Array, g: PmeGrid, w: Float64Array): Float64Array => {
  const [nfft1, nfft2, nfft3] = g.nfft;
  const N = nfft1 * nfft2 * nfft3;
  const out = Float64Array.from(c);
  fft3d(out, nfft1, nfft2, nfft3, -1);
  for (let i = 0; i < N; i++) {
    out[2 * i] *= w[i];
    out[2 * i + 1] *= w[i];
  }
  fft3d(out, nfft1, nfft2, nfft3, 1);
  return out;
};

/** fphi_uind (amoeba_kspace.cpp): the induced-dipole potentials from the two channels of the convolved complex grid. */
export const gatherUind = (c: Float64Array, n: number, fill: PmeFill, g: PmeGrid): { phi1: Float64Array; phi2: Float64Array; sum: Float64Array } => {
  const [nfft1, nfft2, nfft3] = g.nfft;
  const order = g.order;
  const nlpts = Math.floor((order - 1) / 2);
  const wrap = (v: number, nf: number): number => ((v % nf) + nf) % nf;
  const th = fill.theta, ig = fill.igrid;
  const phi1 = new Float64Array(10 * n), phi2 = new Float64Array(10 * n), sum = new Float64Array(20 * n);
  // scratch reused for every atom (zeroed where native starts its sums): the sums and the wrapped grid columns
  const tuv = new Float64Array(20), tuv1 = new Float64Array(10), tuv2 = new Float64Array(10);
  const a1 = new Float64Array(6), a2 = new Float64Array(6), tt = new Float64Array(10);
  const wx = new Int32Array(order);
  for (let m = 0; m < n; m++) {
    tuv.fill(0); tuv1.fill(0); tuv2.fill(0);
    for (let ib = 0; ib < order; ib++) wx[ib] = wrap(ig[3 * m] - nlpts + ib, nfft1);
    let kz = ig[3 * m + 2] - nlpts;
    for (let kb = 0; kb < order; kb++) {
      const v0 = th[2][m * order * 4 + kb * 4], v1 = th[2][m * order * 4 + kb * 4 + 1];
      const v2 = th[2][m * order * 4 + kb * 4 + 2], v3 = th[2][m * order * 4 + kb * 4 + 3];
      // channel 1 (uind): tu00_1, tu01_1, tu10_1, tu20_1, tu11_1, tu02_1; channel 2 (uinp): the same with _2; total: all ten
      a1.fill(0); a2.fill(0); tt.fill(0);
      const wz = wrap(kz, nfft3);
      let jy = ig[3 * m + 1] - nlpts;
      for (let jb = 0; jb < order; jb++) {
        const u0 = th[1][m * order * 4 + jb * 4], u1 = th[1][m * order * 4 + jb * 4 + 1];
        const u2 = th[1][m * order * 4 + jb * 4 + 2], u3 = th[1][m * order * 4 + jb * 4 + 3];
        let t0_1 = 0.0, t1_1 = 0.0, t2_1 = 0.0, t0_2 = 0.0, t1_2 = 0.0, t2_2 = 0.0, t3 = 0.0;
        const row = (wz * nfft2 + wrap(jy, nfft2)) * nfft1;
        for (let ib = 0; ib < order; ib++) {
          const idx = row + wx[ib];
          const tq1 = c[2 * idx], tq2 = c[2 * idx + 1];
          const b0 = th[0][m * order * 4 + ib * 4], b1 = th[0][m * order * 4 + ib * 4 + 1];
          const b2 = th[0][m * order * 4 + ib * 4 + 2], b3 = th[0][m * order * 4 + ib * 4 + 3];
          t0_1 += tq1 * b0;
          t1_1 += tq1 * b1;
          t2_1 += tq1 * b2;
          t0_2 += tq2 * b0;
          t1_2 += tq2 * b1;
          t2_2 += tq2 * b2;
          t3 += (tq1 + tq2) * b3;
        }
        // channel 1: tu00_1 tu10_1 tu01_1 tu20_1 tu11_1 tu02_1 = a1[0..5]; channel 2 likewise in a2
        a1[0] += t0_1 * u0;
        a1[1] += t1_1 * u0;
        a1[2] += t0_1 * u1;
        a1[3] += t2_1 * u0;
        a1[4] += t1_1 * u1;
        a1[5] += t0_1 * u2;
        a2[0] += t0_2 * u0;
        a2[1] += t1_2 * u0;
        a2[2] += t0_2 * u1;
        a2[3] += t2_2 * u0;
        a2[4] += t1_2 * u1;
        a2[5] += t0_2 * u2;
        const t0 = t0_1 + t0_2, t1 = t1_1 + t1_2, t2 = t2_1 + t2_2;
        tt[0] += t0 * u0;
        tt[1] += t1 * u0;
        tt[2] += t0 * u1;
        tt[3] += t2 * u0;
        tt[4] += t1 * u1;
        tt[5] += t0 * u2;
        tt[6] += t3 * u0;
        tt[7] += t2 * u1;
        tt[8] += t1 * u2;
        tt[9] += t0 * u3;
        jy++;
      }
      // [1] tuv100_1 tuv010_1 tuv001_1 tuv200_1 tuv020_1 tuv002_1 tuv110_1 tuv101_1 tuv011_1 (channel 1 and 2 as native)
      tuv1[1] += a1[1] * v0;
      tuv1[2] += a1[2] * v0;
      tuv1[3] += a1[0] * v1;
      tuv1[4] += a1[3] * v0;
      tuv1[5] += a1[5] * v0;
      tuv1[6] += a1[0] * v2;
      tuv1[7] += a1[4] * v0;
      tuv1[8] += a1[1] * v1;
      tuv1[9] += a1[2] * v1;
      tuv2[1] += a2[1] * v0;
      tuv2[2] += a2[2] * v0;
      tuv2[3] += a2[0] * v1;
      tuv2[4] += a2[3] * v0;
      tuv2[5] += a2[5] * v0;
      tuv2[6] += a2[0] * v2;
      tuv2[7] += a2[4] * v0;
      tuv2[8] += a2[1] * v1;
      tuv2[9] += a2[2] * v1;
      tuv[0] += tt[0] * v0;
      tuv[1] += tt[1] * v0;
      tuv[2] += tt[2] * v0;
      tuv[3] += tt[0] * v1;
      tuv[4] += tt[3] * v0;
      tuv[5] += tt[5] * v0;
      tuv[6] += tt[0] * v2;
      tuv[7] += tt[4] * v0;
      tuv[8] += tt[1] * v1;
      tuv[9] += tt[2] * v1;
      tuv[10] += tt[6] * v0;
      tuv[11] += tt[9] * v0;
      tuv[12] += tt[0] * v3;
      tuv[13] += tt[7] * v0;
      tuv[14] += tt[3] * v1;
      tuv[15] += tt[8] * v0;
      tuv[16] += tt[5] * v1;
      tuv[17] += tt[1] * v2;
      tuv[18] += tt[2] * v2;
      tuv[19] += tt[4] * v1;
      kz++;
    }
    for (let k = 1; k < 10; k++) {
      phi1[10 * m + k] = tuv1[k];
      phi2[10 * m + k] = tuv2[k];
    }
    for (let k = 0; k < 20; k++) sum[20 * m + k] = tuv[k];
  }
  return { phi1, phi2, sum };
};

/**
 * multipole_kspace (amoeba_multipole.cpp): the reciprocal-space energy (0.5 sum fmp.fphi, with felec), the forces on the
 * owned sites (f[i] -= recip . grad), the torques converted to forces on the frames, and the virial. `x`, `f` are over the
 * owned and image copies; the sites are the owned atoms (rpole, zl/xl/yl frames as copy indices).
 */
export const multipoleKspace = (
  sites: MpoleSites, n: number, x: Float64Array, f: Float64Array, g: PmeGrid,
  aewald: number, dielectric: number,
): MpoleKspaceResult => {
  if (aewald < 1.0e-6) return { e: 0.0, virial: new Float64Array(6) };
  const [nfft1, nfft2, nfft3] = g.nfft;
  const felec = ELECTRIC / dielectric;
  const cmp = cmpFromRpole(sites.rpole, n);
  const fill = pmeFill(x, n, g);
  const fmp = cmpToFmp(cmp, n, g);
  const w = influence(g, aewald);
  const { fwd, pot } = convolveReal(spreadMpole(fmp, n, fill, g), g, w);
  // virial of the reciprocal sum from the transform before the convolution (amoeba_multipole.cpp)
  const pterm = (MY_PI / aewald) * (MY_PI / aewald);
  const nhalf1 = Math.floor((nfft1 + 1) / 2), nhalf2 = Math.floor((nfft2 + 1) / 2), nhalf3 = Math.floor((nfft3 + 1) / 2);
  let vxx = 0.0, vyy = 0.0, vzz = 0.0, vxy = 0.0, vxz = 0.0, vyz = 0.0;
  let nn = 0;
  for (let k = 0; k < nfft3; k++) {
    for (let j = 0; j < nfft2; j++) {
      for (let i = 0; i < nfft1; i++) {
        const r1 = i >= nhalf1 ? i - nfft1 : i;
        const r2 = j >= nhalf2 ? j - nfft2 : j;
        const r3 = k >= nhalf3 ? k - nfft3 : k;
        const h1 = g.recip[0] * r1 + g.recip[1] * r2 + g.recip[2] * r3;
        const h2 = g.recip[3] * r1 + g.recip[4] * r2 + g.recip[5] * r3;
        const h3 = g.recip[6] * r1 + g.recip[7] * r2 + g.recip[8] * r3;
        const hsq = h1 * h1 + h2 * h2 + h3 * h3;
        const term = -pterm * hsq;
        if (term > -50.0 && hsq !== 0.0) {
          const struc2 = fwd[2 * nn] * fwd[2 * nn] + fwd[2 * nn + 1] * fwd[2 * nn + 1];
          const eterm = 0.5 * felec * w[nn] * struc2;
          const vterm = (2.0 / hsq) * (1.0 - term) * eterm;
          vxx += h1 * h1 * vterm - eterm;
          vyy += h2 * h2 * vterm - eterm;
          vzz += h3 * h3 * vterm - eterm;
          vxy += h1 * h2 * vterm;
          vxz += h1 * h3 * vterm;
          vyz += h2 * h3 * vterm;
        }
        nn++;
      }
    }
  }
  const fphi = gatherMpole(pot, n, fill, g);
  for (let i = 0; i < 20 * n; i++) fphi[i] *= felec;
  const cphi = fphiToCphi(fphi, n, g);
  // energy and forces on the sites (f -= recip . gradient), native deriv1/2/3 index maps
  const deriv1 = [1, 4, 7, 8, 10, 15, 17, 13, 14, 19];
  const deriv2 = [2, 7, 5, 9, 13, 11, 18, 15, 19, 16];
  const deriv3 = [3, 8, 9, 6, 14, 16, 12, 19, 17, 18];
  let e = 0.0;
  for (let i = 0; i < n; i++) {
    let f1 = 0.0, f2 = 0.0, f3 = 0.0;
    for (let k = 0; k < 10; k++) {
      e += fmp[10 * i + k] * fphi[20 * i + k];
      f1 += fmp[10 * i + k] * fphi[20 * i + deriv1[k]];
      f2 += fmp[10 * i + k] * fphi[20 * i + deriv2[k]];
      f3 += fmp[10 * i + k] * fphi[20 * i + deriv3[k]];
    }
    f1 *= nfft1;
    f2 *= nfft2;
    f3 *= nfft3;
    const h1 = g.recip[0] * f1 + g.recip[1] * f2 + g.recip[2] * f3;
    const h2 = g.recip[3] * f1 + g.recip[4] * f2 + g.recip[5] * f3;
    const h3 = g.recip[6] * f1 + g.recip[7] * f2 + g.recip[8] * f3;
    f[3 * i] -= h1;
    f[3 * i + 1] -= h2;
    f[3 * i + 2] -= h3;
  }
  const energy = 0.5 * e;
  // the per-site virial terms and the torques (native cmp/cphi forms)
  const c = (i: number, k: number): number => cmp[10 * i + k];
  const q = (i: number, k: number): number => cphi[10 * i + k];
  for (let i = 0; i < n; i++) {
    vxx = vxx - c(i, 1) * q(i, 1) - 2.0 * c(i, 4) * q(i, 4) - c(i, 7) * q(i, 7) - c(i, 8) * q(i, 8);
    vxy = vxy - 0.5 * (c(i, 2) * q(i, 1) + c(i, 1) * q(i, 2)) - (c(i, 4) + c(i, 5)) * q(i, 7) -
      0.5 * c(i, 7) * (q(i, 4) + q(i, 5)) - 0.5 * (c(i, 8) * q(i, 9) + c(i, 9) * q(i, 8));
    vxz = vxz - 0.5 * (c(i, 3) * q(i, 1) + c(i, 1) * q(i, 3)) - (c(i, 4) + c(i, 6)) * q(i, 8) -
      0.5 * c(i, 8) * (q(i, 4) + q(i, 6)) - 0.5 * (c(i, 7) * q(i, 9) + c(i, 9) * q(i, 7));
    vyy = vyy - c(i, 2) * q(i, 2) - 2.0 * c(i, 5) * q(i, 5) - c(i, 7) * q(i, 7) - c(i, 9) * q(i, 9);
    vyz = vyz - 0.5 * (c(i, 3) * q(i, 2) + c(i, 2) * q(i, 3)) - (c(i, 5) + c(i, 6)) * q(i, 9) -
      0.5 * c(i, 9) * (q(i, 5) + q(i, 6)) - 0.5 * (c(i, 7) * q(i, 8) + c(i, 8) * q(i, 7));
    vzz = vzz - c(i, 3) * q(i, 3) - 2.0 * c(i, 6) * q(i, 6) - c(i, 8) * q(i, 8) - c(i, 9) * q(i, 9);
  }
  for (let i = 0; i < n; i++) {
    const tem = new Float64Array(3);
    tem[0] = c(i, 3) * q(i, 2) - c(i, 2) * q(i, 3) + 2.0 * (c(i, 6) - c(i, 5)) * q(i, 9) +
      c(i, 8) * q(i, 7) + c(i, 9) * q(i, 5) - c(i, 7) * q(i, 8) - c(i, 9) * q(i, 6);
    tem[1] = c(i, 1) * q(i, 3) - c(i, 3) * q(i, 1) + 2.0 * (c(i, 4) - c(i, 6)) * q(i, 8) +
      c(i, 7) * q(i, 9) + c(i, 8) * q(i, 6) - c(i, 8) * q(i, 4) - c(i, 9) * q(i, 7);
    tem[2] = c(i, 2) * q(i, 1) - c(i, 1) * q(i, 2) + 2.0 * (c(i, 5) - c(i, 4)) * q(i, 7) +
      c(i, 7) * q(i, 4) + c(i, 9) * q(i, 8) - c(i, 7) * q(i, 5) - c(i, 8) * q(i, 9);
    const { fx, fy, fz } = torque2force(sites, i, tem, x, f);
    if (sites.axis[i] === 'NOFRAME') continue;
    const iz = sites.zl[i], ix = sites.xl[i], iy = sites.yl[i];
    const xiz = x[3 * iz] - x[3 * i], yiz = x[3 * iz + 1] - x[3 * i + 1], ziz = x[3 * iz + 2] - x[3 * i + 2];
    const xix = x[3 * ix] - x[3 * i], yix = x[3 * ix + 1] - x[3 * i + 1], zix = x[3 * ix + 2] - x[3 * i + 2];
    const xiy = x[3 * iy] - x[3 * i], yiy = x[3 * iy + 1] - x[3 * i + 1], ziy = x[3 * iy + 2] - x[3 * i + 2];
    vxx += xix * fx[0] + xiy * fy[0] + xiz * fz[0];
    vxy += 0.5 * (yix * fx[0] + yiy * fy[0] + yiz * fz[0] + xix * fx[1] + xiy * fy[1] + xiz * fz[1]);
    vxz += 0.5 * (zix * fx[0] + ziy * fy[0] + ziz * fz[0] + xix * fx[2] + xiy * fy[2] + xiz * fz[2]);
    vyy += yix * fx[1] + yiy * fy[1] + yiz * fz[1];
    vyz += 0.5 * (zix * fx[1] + ziy * fy[1] + ziz * fz[1] + yix * fx[2] + yiy * fy[2] + yiz * fz[2]);
    vzz += zix * fx[2] + ziy * fy[2] + ziz * fz[2];
  }
  const virial = new Float64Array([-vxx, -vyy, -vzz, -vxy, -vxz, -vyz]);
  return { e: energy, virial };
};

/**
 * udirect1 (amoeba_induce.cpp): the reciprocal-space permanent field on each owned site, 3 per site, without the electric
 * factor (the field is in native units; polar_energy and the forces carry felec). The grid g is the polarization grid
 * (bsporder), w its influence function.
 */
export const permanentRecipField = (cmp: Float64Array, n: number, fill: PmeFill, g: PmeGrid, w: Float64Array): Float64Array => {
  const fmp = cmpToFmp(cmp, n, g);
  const { pot } = convolveReal(spreadMpole(fmp, n, fill, g), g, w);
  const cphi = fphiToCphi(gatherMpole(pot, n, fill, g), n, g);
  const field = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) field[3 * i + j] = -cphi[10 * i + 1 + j];
  return field;
};

/**
 * umutual1 (amoeba_induce.cpp): the reciprocal-space mutual field of the induced dipoles, returned as the amount to add
 * to the fields (native subtracts dipfield): field and fieldp for uind and uinp.
 */
export const mutualRecipField = (
  uind: Float64Array, uinp: Float64Array, n: number, fill: PmeFill, g: PmeGrid, w: Float64Array,
): { field: Float64Array; fieldp: Float64Array } => {
  const fu = new Float64Array(3 * n), fup = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 3; j++) {
      const a = g.nfft[j] * g.recip[4 * j];
      fu[3 * i + j] = a * uind[3 * i + j];
      fup[3 * i + j] = a * uinp[3 * i + j];
    }
  }
  // gather from the convolved grid with the two channels (real = uind, imaginary = uinp)
  const cc = convolveComplex(spreadUind(fu, fup, n, fill, g), g, w);
  const { phi1, phi2 } = gatherUind(cc, n, fill, g);
  const field = new Float64Array(3 * n), fieldp = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 3; j++) {
      const a = g.nfft[j] * g.recip[4 * j];
      field[3 * i + j] = -a * phi1[10 * i + 1 + j];
      fieldp[3 * i + j] = -a * phi2[10 * i + 1 + j];
    }
  }
  return { field, fieldp };
};

/** Forward FFT (unscaled) of a periodic real grid, the complex transform as interleaved values. */
const forwardFFT = (grid: Float64Array, g: PmeGrid): Float64Array => {
  const [nfft1, nfft2, nfft3] = g.nfft;
  const N = nfft1 * nfft2 * nfft3;
  const c = new Float64Array(2 * N);
  for (let i = 0; i < N; i++) c[2 * i] = grid[i];
  fft3d(c, nfft1, nfft2, nfft3, -1);
  return c;
};

/**
 * polar_kspace (amoeba_polar.cpp, MUTUAL): the reciprocal-space forces and torques of the polarization, added to f, and
 * its virial (returned as the amount to add, virpolar). The permanent potential (felec), the induced-dipole potentials of
 * the two channels and the sum of their gradients give the forces; the torques come from the half-scaled sum potential;
 * the virial has the permanent part, the per-site terms, the frame terms and the cross term of the permanent and induced
 * transforms (perm + uinp against perm + uind).
 */
export const polarKspaceForces = (
  sites: MpoleSites, n: number, x: Float64Array, f: Float64Array, g: PmeGrid, w: Float64Array, fill: PmeFill,
  cmp: Float64Array, uind: Float64Array, uinp: Float64Array, dielectric: number, aewald: number,
): Float64Array => {
  const [nfft1, nfft2, nfft3] = g.nfft;
  const felec = ELECTRIC / dielectric;
  const fmp = cmpToFmp(cmp, n, g);
  const fwdPerm = forwardFFT(spreadMpole(fmp, n, fill, g), g);
  const pot = new Float64Array(nfft1 * nfft2 * nfft3);
  {
    const cc = Float64Array.from(fwdPerm);
    for (let i = 0; i < pot.length; i++) {
      cc[2 * i] *= w[i];
      cc[2 * i + 1] *= w[i];
    }
    fft3d(cc, nfft1, nfft2, nfft3, 1);
    for (let i = 0; i < pot.length; i++) pot[i] = cc[2 * i];
  }
  const fphi = gatherMpole(pot, n, fill, g);
  for (let i = 0; i < 20 * n; i++) fphi[i] *= felec;
  const cphi = fphiToCphi(fphi, n, g);
  const fu = new Float64Array(3 * n), fup = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 3; j++) {
      const a = g.nfft[j] * g.recip[4 * j];
      fu[3 * i + j] = a * uind[3 * i + j];
      fup[3 * i + j] = a * uinp[3 * i + j];
    }
  }
  const cc = convolveComplex(spreadUind(fu, fup, n, fill, g), g, w);
  const { phi1, phi2, sum } = gatherUind(cc, n, fill, g);
  for (let i = 0; i < n; i++) {
    for (let k = 1; k < 10; k++) {
      phi1[10 * i + k] *= felec;
      phi2[10 * i + k] *= felec;
    }
    for (let k = 0; k < 20; k++) sum[20 * i + k] *= felec;
  }
  const deriv1 = [1, 4, 7, 8, 10, 15, 17, 13, 14, 19];
  const deriv2 = [2, 7, 5, 9, 13, 11, 18, 15, 19, 16];
  const deriv3 = [3, 8, 9, 6, 14, 16, 12, 19, 17, 18];
  for (let i = 0; i < n; i++) {
    let f1 = 0.0, f2 = 0.0, f3 = 0.0;
    for (let k = 0; k < 3; k++) {
      const j1 = deriv1[k + 1], j2 = deriv2[k + 1], j3 = deriv3[k + 1];
      const su = fu[3 * i + k] + fup[3 * i + k];
      f1 += su * fphi[20 * i + j1];
      f2 += su * fphi[20 * i + j2];
      f3 += su * fphi[20 * i + j3];
      f1 += fu[3 * i + k] * phi2[10 * i + j1] + fup[3 * i + k] * phi1[10 * i + j1];
      f2 += fu[3 * i + k] * phi2[10 * i + j2] + fup[3 * i + k] * phi1[10 * i + j2];
      f3 += fu[3 * i + k] * phi2[10 * i + j3] + fup[3 * i + k] * phi1[10 * i + j3];
    }
    for (let k = 0; k < 10; k++) {
      f1 += fmp[10 * i + k] * sum[20 * i + deriv1[k]];
      f2 += fmp[10 * i + k] * sum[20 * i + deriv2[k]];
      f3 += fmp[10 * i + k] * sum[20 * i + deriv3[k]];
    }
    f1 *= 0.5 * nfft1;
    f2 *= 0.5 * nfft2;
    f3 *= 0.5 * nfft3;
    const h1 = g.recip[0] * f1 + g.recip[1] * f2 + g.recip[2] * f3;
    const h2 = g.recip[3] * f1 + g.recip[4] * f2 + g.recip[5] * f3;
    const h3 = g.recip[6] * f1 + g.recip[7] * f2 + g.recip[8] * f3;
    f[3 * i] -= h1;
    f[3 * i + 1] -= h2;
    f[3 * i + 2] -= h3;
  }
  // the half-scaled sum potential gives the site torques (native fphidp *= 0.5, cphidp)
  const half = new Float64Array(20 * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < 10; j++) half[20 * i + j] = 0.5 * sum[20 * i + j];
  const cphidp = fphiToCphi(half, n, g);
  const c = (i: number, k: number): number => cmp[10 * i + k];
  const q = (i: number, k: number): number => cphidp[10 * i + k];
  // virial, native order: (A) the permanent transform (vxx -= ...), (B) the per-site terms, (C) the frames, (D) the cross term
  let vxx = 0.0, vyy = 0.0, vzz = 0.0, vxy = 0.0, vxz = 0.0, vyz = 0.0;
  const pterm = (MY_PI / aewald) * (MY_PI / aewald);
  const nhalf1 = Math.floor((nfft1 + 1) / 2), nhalf2 = Math.floor((nfft2 + 1) / 2), nhalf3 = Math.floor((nfft3 + 1) / 2);
  // the cross transform of perm + uinp and perm + uind (native gridfft1 and gridfft2 in the last virial block)
  const cmpU = Float64Array.from(cmp), cmpD = Float64Array.from(cmp);
  for (let i = 0; i < n; i++) {
    for (let j = 1; j < 4; j++) {
      cmpU[10 * i + j] = cmp[10 * i + j] + uinp[3 * i + j - 1];
      cmpD[10 * i + j] = cmp[10 * i + j] + uind[3 * i + j - 1];
    }
  }
  const fwdU = forwardFFT(spreadMpole(cmpToFmp(cmpU, n, g), n, fill, g), g);
  const fwdD = forwardFFT(spreadMpole(cmpToFmp(cmpD, n, g), n, fill, g), g);
  let nn = 0;
  for (let k = 0; k < nfft3; k++) {
    for (let j = 0; j < nfft2; j++) {
      for (let i = 0; i < nfft1; i++) {
        const r1 = i >= nhalf1 ? i - nfft1 : i;
        const r2 = j >= nhalf2 ? j - nfft2 : j;
        const r3 = k >= nhalf3 ? k - nfft3 : k;
        const h1 = g.recip[0] * r1 + g.recip[1] * r2 + g.recip[2] * r3;
        const h2 = g.recip[3] * r1 + g.recip[4] * r2 + g.recip[5] * r3;
        const h3 = g.recip[6] * r1 + g.recip[7] * r2 + g.recip[8] * r3;
        const hsq = h1 * h1 + h2 * h2 + h3 * h3;
        const term = -pterm * hsq;
        if (term > -50.0 && hsq !== 0.0) {
          const expterm = w[nn];
          const eterm = 0.5 * felec * expterm * (fwdPerm[2 * nn] * fwdPerm[2 * nn] + fwdPerm[2 * nn + 1] * fwdPerm[2 * nn + 1]);
          const vterm = (2.0 / hsq) * (1.0 - term) * eterm;
          vxx -= h1 * h1 * vterm - eterm;
          vyy -= h2 * h2 * vterm - eterm;
          vzz -= h3 * h3 * vterm - eterm;
          vxy -= h1 * h2 * vterm;
          vxz -= h1 * h3 * vterm;
          vyz -= h2 * h3 * vterm;
          const struc2 = fwdU[2 * nn] * fwdD[2 * nn] + fwdU[2 * nn + 1] * fwdD[2 * nn + 1];
          const eterm2 = 0.5 * felec * expterm * struc2;
          const vterm2 = (2.0 / hsq) * (1.0 - term) * eterm2;
          vxx += h1 * h1 * vterm2 - eterm2;
          vyy += h2 * h2 * vterm2 - eterm2;
          vzz += h3 * h3 * vterm2 - eterm2;
          vxy += h1 * h2 * vterm2;
          vyz += h2 * h3 * vterm2;
          vxz += h1 * h3 * vterm2;
        }
        nn++;
      }
    }
  }
  // (B) per-site virial terms with the induced dipoles (MUTUAL)
  const cphid = new Float64Array(4), cphip = new Float64Array(4);
  const ftc = fracToCart(g);
  for (let i = 0; i < n; i++) {
    const ux = uind[3 * i] + uinp[3 * i], uy = uind[3 * i + 1] + uinp[3 * i + 1], uz = uind[3 * i + 2] + uinp[3 * i + 2];
    const cp = (k: number): number => cphi[10 * i + k];
    vxx -= c(i, 1) * q(i, 1) + 0.5 * (ux * cp(1));
    vyy -= c(i, 2) * q(i, 2) + 0.5 * (uy * cp(2));
    vzz -= c(i, 3) * q(i, 3) + 0.5 * (uz * cp(3));
    vxy -= 0.5 * (q(i, 1) * c(i, 2) + q(i, 2) * c(i, 1)) + 0.25 * (uy * cp(1) + ux * cp(2));
    vyz -= 0.5 * (q(i, 2) * c(i, 3) + q(i, 3) * c(i, 2)) + 0.25 * (uz * cp(2) + uy * cp(3));
    vxz -= 0.5 * (q(i, 1) * c(i, 3) + q(i, 3) * c(i, 1)) + 0.25 * (uz * cp(1) + ux * cp(3));
    vxx -= 2.0 * c(i, 4) * q(i, 4) + c(i, 7) * q(i, 7) + c(i, 8) * q(i, 8);
    vyy -= 2.0 * c(i, 5) * q(i, 5) + c(i, 7) * q(i, 7) + c(i, 9) * q(i, 9);
    vzz -= 2.0 * c(i, 6) * q(i, 6) + c(i, 8) * q(i, 8) + c(i, 9) * q(i, 9);
    vxy -= (c(i, 4) + c(i, 5)) * q(i, 7) + 0.5 * (c(i, 7) * (q(i, 5) + q(i, 4)) + c(i, 8) * q(i, 9) + c(i, 9) * q(i, 8));
    vyz -= (c(i, 5) + c(i, 6)) * q(i, 9) + 0.5 * (c(i, 9) * (q(i, 5) + q(i, 6)) + c(i, 7) * q(i, 8) + c(i, 8) * q(i, 7));
    vxz -= (c(i, 4) + c(i, 6)) * q(i, 8) + 0.5 * (c(i, 8) * (q(i, 4) + q(i, 6)) + c(i, 7) * q(i, 9) + c(i, 9) * q(i, 7));
    // the field-gradient products of the induced dipoles (cphid from uind's potential, cphip from uinp's)
    for (let j = 1; j < 4; j++) {
      cphid[j] = 0.0;
      cphip[j] = 0.0;
      for (let k = 1; k < 4; k++) {
        cphid[j] += ftc[j][k] * phi1[10 * i + k];
        cphip[j] += ftc[j][k] * phi2[10 * i + k];
      }
    }
    const ui = (k: number): number => uinp[3 * i + k], ud = (k: number): number => uind[3 * i + k];
    vxx -= 0.5 * (cphid[1] * ui(0) + cphip[1] * ud(0));
    vyy -= 0.5 * (cphid[2] * ui(1) + cphip[2] * ud(1));
    vzz -= 0.5 * (cphid[3] * ui(2) + cphip[3] * ud(2));
    vxy -= 0.25 * (cphid[1] * ui(1) + cphip[1] * ud(1) + cphid[2] * ui(0) + cphip[2] * ud(0));
    vyz -= 0.25 * (cphid[2] * ui(2) + cphip[2] * ud(2) + cphid[3] * ui(1) + cphip[3] * ud(1));
    vxz -= 0.25 * (cphid[1] * ui(2) + cphip[1] * ud(2) + cphid[3] * ui(0) + cphip[3] * ud(0));
  }
  // site torques (native tep) and the frame virial of their forces (C)
  for (let i = 0; i < n; i++) {
    const tep = new Float64Array(3);
    tep[0] = c(i, 3) * q(i, 2) - c(i, 2) * q(i, 3) + 2.0 * (c(i, 6) - c(i, 5)) * q(i, 9) + c(i, 8) * q(i, 7) +
      c(i, 9) * q(i, 5) - c(i, 7) * q(i, 8) - c(i, 9) * q(i, 6);
    tep[1] = c(i, 1) * q(i, 3) - c(i, 3) * q(i, 1) + 2.0 * (c(i, 4) - c(i, 6)) * q(i, 8) + c(i, 7) * q(i, 9) +
      c(i, 8) * q(i, 6) - c(i, 8) * q(i, 4) - c(i, 9) * q(i, 7);
    tep[2] = c(i, 2) * q(i, 1) - c(i, 1) * q(i, 2) + 2.0 * (c(i, 5) - c(i, 4)) * q(i, 7) + c(i, 7) * q(i, 4) +
      c(i, 9) * q(i, 8) - c(i, 7) * q(i, 5) - c(i, 8) * q(i, 9);
    const { fx, fy, fz } = torque2force(sites, i, tep, x, f);
    const iz = sites.zl[i], ix = sites.xl[i], iy = sites.yl[i];
    const xiz = x[3 * iz] - x[3 * i], yiz = x[3 * iz + 1] - x[3 * i + 1], ziz = x[3 * iz + 2] - x[3 * i + 2];
    const xix = x[3 * ix] - x[3 * i], yix = x[3 * ix + 1] - x[3 * i + 1], zix = x[3 * ix + 2] - x[3 * i + 2];
    const xiy = x[3 * iy] - x[3 * i], yiy = x[3 * iy + 1] - x[3 * i + 1], ziy = x[3 * iy + 2] - x[3 * i + 2];
    vxx += xix * fx[0] + xiy * fy[0] + xiz * fz[0];
    vyy += yix * fx[1] + yiy * fy[1] + yiz * fz[1];
    vzz += zix * fx[2] + ziy * fy[2] + ziz * fz[2];
    vxy += 0.5 * (yix * fx[0] + yiy * fy[0] + yiz * fz[0] + xix * fx[1] + xiy * fy[1] + xiz * fz[1]);
    vyz += 0.5 * (zix * fx[1] + ziy * fy[1] + ziz * fz[1] + yix * fx[2] + yiy * fy[2] + yiz * fz[2]);
    vxz += 0.5 * (zix * fx[0] + ziy * fy[0] + ziz * fz[0] + xix * fx[2] + xiy * fy[2] + xiz * fz[2]);
  }
  // virpolar -= v (amoeba_polar.cpp polar_kspace tail)
  return new Float64Array([-vxx, -vyy, -vzz, -vxy, -vxz, -vyz]);
};

/** Guard for callers that pass an order the B-spline cannot hold (bsorder < 2 or above the grid). */
export const checkPmeOrder = (order: number, nfft: number): void => {
  if (order < 2 || order > 8) throw new StyleError(`pair amoeba: pme-order ${order} is not supported (2..8)`);
  if (order > nfft) throw new StyleError('pair amoeba: the PME grid is smaller than the B-spline order');
};
