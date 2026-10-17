/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/amoeba_dispersion.cpp (dispersion_kspace()) and src/AMOEBA/amoeba_kspace.cpp (grid_disp(), the
 * moduli and bspline fill of the dispersion grid) at commit b332f28, adapted from Tinker edrecip1d(). AMOEBA package
 * authors: the LAMMPS developers (Sandia Corporation, GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The reciprocal-space part of the damped dispersion (key "dewald", pair hippo): the C6 coefficients spread on a
 * B-spline grid, the convolution with the dispersion influence function, the forces gathered from the potential grid,
 * and the constant csixpr term of the self interaction. The grid, the B-spline fill and the FFT are those of the
 * multipole PME (kspace.ts) with the dispersion order and grid of the key file.
 */
import { checkPmeOrder, convolveReal, factorable, makePmeGrid, pmeFill, type PmeGrid } from './kspace';
import { erfcExact } from '../../engine/force/erfc';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

const MY_PI = Math.PI;

/** Grid of the dispersion sum: dpme-grid (or the box default) and dpme-order, with the box origin and lengths. */
export const makeDispersionGrid = (
  nfft: [number, number, number], order: number, lo: [number, number, number], prd: [number, number, number],
): PmeGrid => {
  for (let d = 0; d < 3; d++) if (nfft[d] < 1) throw new Error('pair hippo: dpme-grid must be positive');
  checkPmeOrder(order, Math.min(nfft[0], nfft[1], nfft[2]));
  return makePmeGrid(nfft, order, lo, prd);
};

/**
 * dispersion_kspace: the reciprocal energy (added to the returned e, which the caller adds to edisp) and the forces (added
 * to f) of the dispersion sum over the n owned atoms. `cls` gives the class of each atom and `csix` the C6 per class.
 * `csixpr` is the constant of the self term (sum over class pairs csix_i csix_j n_i n_j).
 */
export const dispersionKspace = (
  x: Float64Array, f: Float64Array, n: number, cls: Int32Array, csix: Float64Array,
  g: PmeGrid, aewald: number, csixpr: number,
): number => {
  if (aewald < 1.0e-6) return 0.0;
  const [nfft1, nfft2, nfft3] = g.nfft;
  const order = g.order;
  const nlpts = Math.floor((order - 1) / 2);
  const volbox = g.prd[0] * g.prd[1] * g.prd[2];
  const fill = pmeFill(x, n, g);
  const th = fill.theta, ig = fill.igrid;
  const wrap = (v: number, nf: number): number => ((v % nf) + nf) % nf;
  // grid_disp: the C6 of each site spread with the three B-spline sets (periodic wrap onto the grid)
  const N = nfft1 * nfft2 * nfft3;
  const grid = new Float64Array(N);
  for (let m = 0; m < n; m++) {
    const fi = csix[cls[m]];
    let kz = ig[3 * m + 2] - nlpts;
    for (let kb = 0; kb < order; kb++) {
      const v0 = th[2][m * order * 4 + kb * 4] * fi;
      let jy = ig[3 * m + 1] - nlpts;
      for (let jb = 0; jb < order; jb++) {
        const u0 = th[1][m * order * 4 + jb * 4];
        const term = v0 * u0;
        let ix = ig[3 * m] - nlpts;
        for (let ib = 0; ib < order; ib++) {
          const t0 = th[0][m * order * 4 + ib * 4];
          grid[(wrap(kz, nfft3) * nfft2 + wrap(jy, nfft2)) * nfft1 + wrap(ix, nfft1)] += term * t0;
          ix++;
        }
        jy++;
      }
      kz++;
    }
  }
  // the influence weights of the dispersion sum: -term1/denom per wavevector (zero at h = 0 and where exp underflows)
  const denom0 = (6.0 * volbox) / libmPow(MY_PI, 1.5);
  const fac1 = 2.0 * libmPow(MY_PI, 3.5);
  const fac2 = aewald * aewald * aewald;
  const fac3 = -2.0 * aewald * MY_PI * MY_PI;
  const bfac = MY_PI / aewald;
  const nhalf1 = Math.floor((nfft1 + 1) / 2), nhalf2 = Math.floor((nfft2 + 1) / 2), nhalf3 = Math.floor((nfft3 + 1) / 2);
  const bs1 = g.bsmod[0], bs2 = g.bsmod[1], bs3 = g.bsmod[2];
  const w = new Float64Array(N);
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
        const h = Math.sqrt(hsq);
        const b = h * bfac;
        const hhh = h * hsq;
        const term = -b * b;
        if (term > -50.0 && hsq !== 0.0) {
          const expterm = libmExp(term);
          const erfcterm = erfcExact(b);
          const term1 = fac1 * erfcterm * hhh + expterm * (fac2 + fac3 * hsq);
          const denom = denom0 * bs1[i] * bs2[j] * bs3[k];
          w[nn] = -(term1 / denom);
        }
        nn++;
      }
    }
  }
  // the energy sum e = sum w |F|^2 over the forward transform, and the potential grid of the gather
  const { fwd, pot } = convolveReal(grid, g, w);
  let e = 0.0;
  for (let q = 0; q < N; q++) e += w[q] * (fwd[2 * q] * fwd[2 * q] + fwd[2 * q + 1] * fwd[2 * q + 1]);
  // gather (dispersion_kspace): forces from the potential grid and the B-spline derivatives
  const recipX = g.recip[0], recipY = g.recip[4], recipZ = g.recip[8];
  for (let m = 0; m < n; m++) {
    let de1 = 0.0, de2 = 0.0, de3 = 0.0;
    let kz = ig[3 * m + 2] - nlpts;
    for (let kb = 0; kb < order; kb++) {
      const t3 = th[2][m * order * 4 + kb * 4];
      const dt3 = nfft3 * th[2][m * order * 4 + kb * 4 + 1];
      let jy = ig[3 * m + 1] - nlpts;
      for (let jb = 0; jb < order; jb++) {
        const t2 = th[1][m * order * 4 + jb * 4];
        const dt2 = nfft2 * th[1][m * order * 4 + jb * 4 + 1];
        let ix = ig[3 * m] - nlpts;
        for (let ib = 0; ib < order; ib++) {
          const t1 = th[0][m * order * 4 + ib * 4];
          const dt1 = nfft1 * th[0][m * order * 4 + ib * 4 + 1];
          const term = pot[(wrap(kz, nfft3) * nfft2 + wrap(jy, nfft2)) * nfft1 + wrap(ix, nfft1)];
          de1 += 2.0 * term * dt1 * t2 * t3;
          de2 += 2.0 * term * dt2 * t1 * t3;
          de3 += 2.0 * term * dt3 * t1 * t2;
          ix++;
        }
        jy++;
      }
      kz++;
    }
    const fi = csix[cls[m]];
    f[3 * m] -= fi * recipX * de1;
    f[3 * m + 1] -= fi * recipY * de2;
    f[3 * m + 2] -= fi * recipZ * de3;
  }
  // the constant self term of the reciprocal sum
  const term = (csixpr * aewald * aewald * aewald) / denom0;
  return e - term;
};

/** The default dispersion grid of one length (amoeba_kspace.cpp: ddens = 0.8, rounded up to a factorable size). */
export const dispersionGridSize = (prd: number): number => {
  let n = Math.trunc(prd * 0.8 - 1.0e-8) + 1;
  while (!factorable(n)) n++;
  return n;
};
