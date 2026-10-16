/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/pppm_disp.cpp at b332f28: the stencil and charge-assignment helpers (set_grid_local,
 * particle_map, compute_rho_coeff, compute_rho1d, compute_gf_denom, gf_denom), the Coulomb and geometric-dispersion
 * Green's functions (compute_gf, compute_gf_6), the k-vector arrays of setup() (fkx, fkx2), the vg virial
 * coefficients, the dispersion G-ewald and mesh estimate (f_6, derivf_6, set_init_g6, set_n_pppm_6, adjust_gewald_6,
 * lj_rspace_error, compute_qopt_6_ik) and factorable(); and the constant OFFSET of particle_map.
 * Contributing authors of pppm_disp.cpp: Rolf Isele-Holder (Aachen University), Paul Crozier (SNL).
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { erfcExact as erfcLocal } from '../../engine/force/erfc';
import { sin as libmSin } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';

/** Offset added before int() in particle_map (pppm_disp.cpp: "add/subtract OFFSET to avoid int(-0.75) = 0"). */
export const OFFSET = 16384;
/** pppm_disp.cpp: static constexpr int MAXORDER = 7. */
export const MAXORDER = 7;
/** pppm_disp.cpp: static constexpr double SMALL = 0.00001 and LARGE = 10000.0. */
export const SMALL = 0.00001;
export const LARGE = 10000;

/** factorable(): the mesh is a product of 2, 3 and 5 (factors[] of PPPMDisp::PPPMDisp). */
export const factorable = (n: number): boolean => {
  let m = n;
  while (m > 1) {
    if (m % 2 === 0) m /= 2;
    else if (m % 3 === 0) m /= 3;
    else if (m % 5 === 0) m /= 5;
    else return false;
  }
  return true;
};

/** Stencil of order P: set_grid_local's shift, shiftone, nlower, nupper and the coefficients of compute_rho_coeff. */
export interface Stencil {
  order: number;
  shift: number;
  shiftone: number;
  nlower: number;
  nupper: number;
  /** rc[l * order + (k - nlower)] = a(l, k): the polynomial of the weight, l = power of the offset. */
  rc: Float64Array;
}

/** compute_rho_coeff (pppm_disp.cpp): the coefficients of the order-n B-spline-like weight function. */
export const makeStencil = (ord: number): Stencil => {
  // set_grid_local: shift and shiftone depend on the parity of the order; nlower = -(order-1)/2, nupper = order/2
  const shift = ord % 2 ? OFFSET + 0.5 : OFFSET;
  const shiftone = ord % 2 ? 0.0 : 0.5;
  const nlower = -Math.floor((ord - 1) / 2);
  const nupper = Math.floor(ord / 2);
  const W = 2 * ord + 1;
  const a = new Float64Array(ord * W);
  const A = (l: number, k: number) => l * W + k + ord;
  a[A(0, 0)] = 1.0;
  for (let j = 1; j < ord; j++) {
    for (let k = -j; k <= j; k += 2) {
      let s = 0.0;
      for (let l = 0; l < j; l++) {
        a[A(l + 1, k)] = (a[A(l, k + 1)] - a[A(l, k - 1)]) / (l + 1);
        s += Math.pow(0.5, l + 1) * (a[A(l, k - 1)] + Math.pow(-1.0, l) * a[A(l, k + 1)]) / (l + 1);
      }
      a[A(0, k)] = s;
    }
  }
  const rc = new Float64Array(ord * ord);
  let m = 0;
  for (let k = -(ord - 1); k < ord; k += 2) {
    for (let l = 0; l < ord; l++) rc[l * ord + m] = a[A(l, k)];
    m++;
  }
  return { order: ord, shift, shiftone, nlower, nupper, rc };
};

/** compute_rho1d: the P weights along one dimension at offset dx (index k - nlower). */
export const rho1d = (st: Stencil, dx: number, out: Float64Array): void => {
  const P = st.order;
  for (let idx = 0; idx < P; idx++) {
    let r = 0.0;
    for (let l = P - 1; l >= 0; l--) r = st.rc[l * P + idx] + r * dx;
    out[idx] = r;
  }
};

/** compute_gf_denom: the coefficients of the Hockney-Eastwood denominator expansion (pre-computed once per order). */
export const gfDenomCoeffs = (ord: number): Float64Array => {
  const gf = new Float64Array(Math.max(ord, 1));
  for (let l = 1; l < ord; l++) gf[l] = 0.0;
  gf[0] = 1.0;
  let l = 0;
  for (let m = 1; m < ord; m++) {
    for (l = m; l > 0; l--) gf[l] = 4.0 * (gf[l] * (l - m) * (l - m - 0.5) - gf[l - 1] * (l - m - 1) * (l - m - 1));
    gf[0] = 4.0 * (gf[0] * (l - m) * (l - m - 0.5));
  }
  let ifact = 1;
  for (let k = 1; k < 2 * ord; k++) ifact *= k;
  const gaminv = 1.0 / ifact;
  for (let i = 0; i < ord; i++) gf[i] *= gaminv;
  return gf;
};

/** gf_denom: (product over the three dimensions of the Horner sums)^2, with x = sin^2(k dx / 2). */
export const gfDenom = (x: number, y: number, z: number, gb: Float64Array, ord: number): number => {
  let sx = 0.0, sy = 0.0, sz = 0.0;
  for (let l = ord - 1; l >= 0; l--) {
    sx = gb[l] + sx * x;
    sy = gb[l] + sy * y;
    sz = gb[l] + sz * z;
  }
  const s = sx * sy * sz;
  return s * s;
};

/** Signed mesh index of grid point i (setup(): per = i - N*(2i/N)). */
export const signedIndex = (i: number, N: number): number => i - N * Math.floor((2 * i) / N);

/** Box and mesh of one solve: the grid, the box lengths and the k-vector arrays of setup() (fkx, fkx2 per dimension). */
export interface MeshGeom {
  N: [number, number, number];
  prd: [number, number, number];
  unit: [number, number, number];
  /** fk[d][i] = unit_d * signed(i), fk2[d][i] = unit_d * signed((N_d - i) % N_d). */
  fk: [Float64Array, Float64Array, Float64Array];
  fk2: [Float64Array, Float64Array, Float64Array];
}

export const meshGeom = (N: [number, number, number], prd: [number, number, number]): MeshGeom => {
  const unit: [number, number, number] = [2 * Math.PI / prd[0], 2 * Math.PI / prd[1], 2 * Math.PI / prd[2]];
  const fk = [0, 1, 2].map((d) => new Float64Array(N[d])) as MeshGeom['fk'];
  const fk2 = [0, 1, 2].map((d) => new Float64Array(N[d])) as MeshGeom['fk2'];
  for (let d = 0; d < 3; d++) {
    for (let i = 0; i < N[d]; i++) {
      fk[d][i] = unit[d] * signedIndex(i, N[d]);
      const j = (N[d] - i) % N[d];
      fk2[d][i] = unit[d] * signedIndex(j, N[d]);
    }
  }
  return { N, prd, unit, fk, fk2 };
};

/**
 * Coulomb Green's function of compute_gf (numerator 4 pi / k^2, no alias sums), over the whole mesh in FFT order
 * (index (iz * ny + iy) * nx + ix). The volume-independent part; the caller multiplies nothing else.
 */
export const greenCoulomb = (mg: MeshGeom, g: number, ord: number): Float64Array => {
  const [nx, ny, nz] = mg.N;
  const [xprd, yprd, zprd] = mg.prd;
  const [ux, uy, uz] = mg.unit;
  const gb = gfDenomCoeffs(ord);
  const out = new Float64Array(nx * ny * nz);
  let n = 0;
  for (let m = 0; m < nz; m++) {
    const mper = m - nz * Math.floor((2 * m) / nz);
    const qz = uz * mper;
    const snz = libmSin(0.5 * qz * zprd / nz);
    const snz2 = snz * snz;
    const sz = libmExp(-0.25 * Math.pow(qz / g, 2.0));
    let wz = 1.0;
    const argz = 0.5 * qz * zprd / nz;
    if (argz !== 0.0) wz = Math.pow(libmSin(argz) / argz, ord);
    wz *= wz;
    for (let l = 0; l < ny; l++) {
      const lper = l - ny * Math.floor((2 * l) / ny);
      const qy = uy * lper;
      const sny = libmSin(0.5 * qy * yprd / ny);
      const sny2 = sny * sny;
      const sy = libmExp(-0.25 * Math.pow(qy / g, 2.0));
      let wy = 1.0;
      const argy = 0.5 * qy * yprd / ny;
      if (argy !== 0.0) wy = Math.pow(libmSin(argy) / argy, ord);
      wy *= wy;
      for (let k = 0; k < nx; k++) {
        const kper = k - nx * Math.floor((2 * k) / nx);
        const qx = ux * kper;
        const snx = libmSin(0.5 * qx * xprd / nx);
        const snx2 = snx * snx;
        const sx = libmExp(-0.25 * Math.pow(qx / g, 2.0));
        let wx = 1.0;
        const argx = 0.5 * qx * xprd / nx;
        if (argx !== 0.0) wx = Math.pow(libmSin(argx) / argx, ord);
        wx *= wx;
        const sqk = Math.pow(qx, 2.0) + Math.pow(qy, 2.0) + Math.pow(qz, 2.0);
        if (sqk !== 0.0) {
          const numerator = 4.0 * Math.PI / sqk;
          const denominator = gfDenom(snx2, sny2, snz2, gb, ord);
          out[n++] = numerator * sx * sy * sz * wx * wy * wz / denominator;
        } else out[n++] = 0.0;
      }
    }
  }
  return out;
};

/** The Coulomb virial coefficients vg (setup(): vterm = -2 (1/k^2 + 1/(4 g^2))), 6 per grid point. */
export const virialCoulomb = (mg: MeshGeom, g: number): Float64Array => {
  const [nx, ny, nz] = mg.N;
  const vg = new Float64Array(6 * nx * ny * nz);
  const gew2inv = 1 / (g * g);
  let n = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const kx = mg.fk[0][i], ky = mg.fk[1][j], kz = mg.fk[2][k];
    const sqk = kx * kx + ky * ky + kz * kz;
    if (sqk !== 0.0) {
      const vterm = -2.0 * (1.0 / sqk + 0.25 * gew2inv);
      vg[6 * n] = 1.0 + vterm * kx * kx;
      vg[6 * n + 1] = 1.0 + vterm * ky * ky;
      vg[6 * n + 2] = 1.0 + vterm * kz * kz;
      vg[6 * n + 3] = vterm * kx * ky;
      vg[6 * n + 4] = vterm * kx * kz;
      vg[6 * n + 5] = vterm * ky * kz;
    }
    n++;
  }
  return vg;
};

/** The geometric dispersion Green's function of compute_gf_6 (numerator -pi^(3/2) g^3 / 3). */
export const greenDispersion = (mg: MeshGeom, g: number, ord: number): Float64Array => {
  const [nx, ny, nz] = mg.N;
  const [xprd, yprd, zprd] = mg.prd;
  const [ux, uy, uz] = mg.unit;
  const gb = gfDenomCoeffs(ord);
  const out = new Float64Array(nx * ny * nz);
  const inv2ew = 1 / (2 * g);
  const rtpi = Math.sqrt(Math.PI);
  const numerator = -Math.PI * rtpi * g * g * g / 3.0;
  let n = 0;
  for (let m = 0; m < nz; m++) {
    const mper = m - nz * Math.floor((2 * m) / nz);
    const qz = uz * mper;
    const snz = libmSin(0.5 * uz * mper * zprd / nz);
    const snz2 = snz * snz;
    const sz = libmExp(-qz * qz * inv2ew * inv2ew);
    let wz = 1.0;
    const argz = 0.5 * qz * zprd / nz;
    if (argz !== 0.0) wz = Math.pow(libmSin(argz) / argz, ord);
    wz *= wz;
    for (let l = 0; l < ny; l++) {
      const lper = l - ny * Math.floor((2 * l) / ny);
      const qy = uy * lper;
      const sny = libmSin(0.5 * uy * lper * yprd / ny);
      const sny2 = sny * sny;
      const sy = libmExp(-qy * qy * inv2ew * inv2ew);
      let wy = 1.0;
      const argy = 0.5 * qy * yprd / ny;
      if (argy !== 0.0) wy = Math.pow(libmSin(argy) / argy, ord);
      wy *= wy;
      for (let k = 0; k < nx; k++) {
        const kper = k - nx * Math.floor((2 * k) / nx);
        const qx = ux * kper;
        const snx = libmSin(0.5 * ux * kper * xprd / nx);
        const snx2 = snx * snx;
        const sx = libmExp(-qx * qx * inv2ew * inv2ew);
        let wx = 1.0;
        const argx = 0.5 * qx * xprd / nx;
        if (argx !== 0.0) wx = Math.pow(libmSin(argx) / argx, ord);
        wx *= wx;
        const sqk = qx * qx + qy * qy + qz * qz;
        if (sqk !== 0.0) {
          const denominator = gfDenom(snx2, sny2, snz2, gb, ord);
          const rtsqk = Math.sqrt(sqk);
          const term = (1 - 2 * sqk * inv2ew * inv2ew) * sx * sy * sz
            + 2 * sqk * rtsqk * inv2ew * inv2ew * inv2ew * rtpi * erfcLocal(rtsqk * inv2ew);
          out[n++] = numerator * term * wx * wy * wz / denominator;
        } else out[n++] = 0.0;
      }
    }
  }
  return out;
};

/** The dispersion virial coefficients vg_6 of setup() (vterm from b = |k|/(2 g), erfc form), 6 per grid point. */
export const virialDispersion = (mg: MeshGeom, g: number): Float64Array => {
  const [nx, ny, nz] = mg.N;
  const vg = new Float64Array(6 * nx * ny * nz);
  const gewinv = 1 / g;
  const rtpi = Math.sqrt(Math.PI);
  let n = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const kx = mg.fk[0][i], ky = mg.fk[1][j], kz = mg.fk[2][k];
    const sqk = kx * kx + ky * ky + kz * kz;
    if (sqk !== 0.0) {
      const b = 0.5 * Math.sqrt(sqk) * gewinv;
      const bs = b * b;
      const bt = bs * b;
      const erft = 2 * bt * rtpi * erfcLocal(b);
      const expt = libmExp(-bs);
      const nom = erft - 2 * bs * expt;
      const denom = nom + expt;
      const vterm = denom === 0 ? 3.0 / sqk : 3.0 * nom / (sqk * denom);
      vg[6 * n] = 1.0 + vterm * kx * kx;
      vg[6 * n + 1] = 1.0 + vterm * ky * ky;
      vg[6 * n + 2] = 1.0 + vterm * kz * kz;
      vg[6 * n + 3] = vterm * kx * ky;
      vg[6 * n + 4] = vterm * kx * kz;
      vg[6 * n + 5] = vterm * ky * kz;
    }
    n++;
  }
  return vg;
};

/** The lj_rspace_error of pppm_disp.cpp: the real-space force error of the dispersion term. */
export const ljRspaceError = (
  cutoffLj: number, g: number, csum: number, natoms: number, xprd: number, yprd: number, zprd: number,
): number => {
  let rgs = cutoffLj * g;
  rgs *= rgs;
  const rgsInv = 1.0 / rgs;
  return csum / Math.sqrt(natoms * xprd * yprd * zprd * cutoffLj)
    * Math.sqrt(Math.PI) * Math.pow(g, 5) * Math.exp(-rgs) * (1 + rgsInv * (3 + rgsInv * (6 + rgsInv * 6)));
};

/**
 * compute_qopt_6_ik: the optimal-influence-function error sum of the ik dispersion solver on mesh N with G g.
 * Loops over all grid points, with the alias images -2..2 per dimension, as the C++ does.
 */
export const qoptDispersionIk = (N: [number, number, number], prd: [number, number, number], g: number, ord: number): number => {
  const [nx, ny, nz] = N;
  const [xprd, yprd, zprd] = prd;
  const unitkx = 2.0 * Math.PI / xprd;
  const unitky = 2.0 * Math.PI / yprd;
  const unitkz = 2.0 * Math.PI / zprd;
  const inv2ew = 1.0 / (2 * g);
  const rtpi = Math.sqrt(Math.PI);
  const nb = 2;
  let qopt = 0.0;
  for (let m = 0; m < nz; m++) {
    const mper = m - nz * Math.floor((2 * m) / nz);
    for (let l = 0; l < ny; l++) {
      const lper = l - ny * Math.floor((2 * l) / ny);
      for (let k = 0; k < nx; k++) {
        const kper = k - nx * Math.floor((2 * k) / nx);
        const sqk = Math.pow(unitkx * kper, 2.0) + Math.pow(unitky * lper, 2.0) + Math.pow(unitkz * mper, 2.0);
        if (sqk === 0.0) continue;
        let sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
        for (let ax = -nb; ax <= nb; ax++) {
          const qx = unitkx * (kper + nx * ax);
          const sx = libmExp(-qx * qx * inv2ew * inv2ew);
          let wx = 1.0;
          const argx = 0.5 * qx * xprd / nx;
          if (argx !== 0.0) wx = Math.pow(libmSin(argx) / argx, ord);
          for (let ay = -nb; ay <= nb; ay++) {
            const qy = unitky * (lper + ny * ay);
            const sy = libmExp(-qy * qy * inv2ew * inv2ew);
            let wy = 1.0;
            const argy = 0.5 * qy * yprd / ny;
            if (argy !== 0.0) wy = Math.pow(libmSin(argy) / argy, ord);
            for (let az = -nb; az <= nb; az++) {
              const qz = unitkz * (mper + nz * az);
              const sz = libmExp(-qz * qz * inv2ew * inv2ew);
              let wz = 1.0;
              const argz = 0.5 * qz * zprd / nz;
              if (argz !== 0.0) wz = Math.pow(libmSin(argz) / argz, ord);
              const dot1 = unitkx * kper * qx + unitky * lper * qy + unitkz * mper * qz;
              const dot2 = qx * qx + qy * qy + qz * qz;
              const rtdot2 = Math.sqrt(dot2);
              let term = (1 - 2 * dot2 * inv2ew * inv2ew) * sx * sy * sz
                + 2 * dot2 * rtdot2 * inv2ew * inv2ew * inv2ew * rtpi * erfcLocal(rtdot2 * inv2ew);
              term *= g * g * g;
              const u2 = Math.pow(wx * wy * wz, 2.0);
              sum1 += term * term * Math.PI * Math.PI * Math.PI / 9.0 * dot2;
              sum2 += -u2 * term * Math.PI * rtpi / 3.0 * dot1;
              sum3 += u2;
            }
          }
        }
        sum2 *= sum2;
        sum3 *= sum3 * sqk;
        qopt += sum1 - sum2 / sum3;
      }
    }
  }
  return qopt;
};

/**
 * Eigenvalues of a real symmetric n x n matrix (row-major), by cyclic Jacobi rotations, sorted by decreasing |value|.
 * Used for the nsplit count of PPPMDisp::init_coeffs (the QR eigen-decomposition of the dispersion matrix C_ij; only the
 * eigenvalues are needed here, and the count is the same).
 */
export const symEigenvalues = (M: Float64Array, n: number): number[] => {
  const a = Float64Array.from(M.subarray(0, n * n));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
    if (off < 1e-300) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (apq === 0) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p], akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k], aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
      }
    }
  }
  const ev: number[] = [];
  for (let i = 0; i < n; i++) ev.push(a[i * n + i]);
  return ev.sort((x, y) => Math.abs(y) - Math.abs(x));
};
