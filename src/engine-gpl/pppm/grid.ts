/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/pppm.cpp (init, set_grid_global, factorable, compute_df_kspace, compute_qopt, estimate_ik_error,
 * adjust_gewald, newton_raphson_f, derivf, compute_gf_denom, the triclinic grid scaling) and src/kspace.cpp
 * (KSpace::x2lamdaT, KSpace::lamda2xT) at commit b332f28. PPPM contributing authors: Steve Plimpton (Sandia), Stan Moore
 * (SNL, triclinic), Paul Crozier (Sandia), and the other LAMMPS developers. Copyright Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The automatic G-ewald and mesh of kspace_style pppm in triclinic boxes, and of kspace_modify diff ad (orthogonal
 * boxes), as native LAMMPS chooses them. The core PPPM (src/engine/force/kspace/pppm.ts) calls nativePppmGrid() through
 * the lazy HOOKS entry 'pppm' only when the box is triclinic or diff ad is set and gewald or mesh is not given.
 *
 * Triclinic path: the initial mesh estimate uses the orthogonal edge lengths, the grid is then mapped to the skewed
 * box with the transposed cell matrix (KSpace::lamda2xT of the grid counts divided by the edge lengths, +1 after
 * truncation), and the mesh spacings used by the G-ewald Newton solver come from KSpace::x2lamdaT of the final grid.
 * Ad path (orthogonal only; native refuses ad with a triclinic box): the mesh is found by shrinking the spacing from
 * 4/G by 0.95 until the analytic-differentiation force estimate (compute_qopt) meets the accuracy.
 */
import { StyleError } from '../../engine/force/types';
import { pppmKspaceError, type PppmGridRequest, type PppmGridResult } from '../../engine/force/kspace/pppm';
import { isFFTSize } from '../../engine/force/fft';
import { log as libmLog } from '../libm/log';
import { sinFma } from '../libm/sincos_fma';
import { exp as libmExp } from '../libm/exp';

const LARGE = 10000;
const SMALL = 0.00001;
const MY_2PI = 6.28318530717958647692;
const MY_4PI = 12.56637061435917295384;

/** math_special.h powsinxx: (sin x / x)^n by binary powering, 1 at x = 0. */
const powsinxx = (x: number, n: number): number => {
  if (x === 0) return 1;
  let ww = sinFma(x) / x;
  let yy = 1;
  for (; n !== 0; n >>= 1, ww *= ww) if (n & 1) yy *= ww;
  return yy;
};

/** PPPM::factorable: every prime factor is 2, 3 or 5. */
const factorable = (n: number): boolean => isFFTSize(n);

/** Triclinic cell matrix of the core (domain h = lx, ly, lz, yz, xz, xy): inverse upper triangular (invertUpper). */
const invert = (r: PppmGridRequest): number[][] => {
  const { lx, ly, lz, xy, xz, yz } = r;
  return [
    [1 / lx, -xy / (lx * ly), (xy * yz - ly * xz) / (lx * ly * lz)],
    [0, 1 / ly, -yz / (ly * lz)],
    [0, 0, 1 / lz],
  ];
};

/** KSpace::x2lamdaT: lamda = (H^-1)^T v, the z component divided by the slab volfactor (slabflag 1). */
const x2lamdaT = (r: PppmGridRequest, v: number[]): number[] => {
  const inv = invert(r);
  const lam = [
    inv[0][0] * v[0],
    inv[0][1] * v[0] + inv[1][1] * v[1],
    inv[0][2] * v[0] + inv[1][2] * v[1] + inv[2][2] * v[2],
  ];
  if (r.slab > 1) lam[2] /= r.slab;
  return lam;
};

/** KSpace::lamda2xT: v = H^T lamda with the absolute tilts (as the native code takes fabs of the tilts). */
const lamda2xT = (r: PppmGridRequest, lam: number[]): number[] => [
  r.lx * lam[0],
  Math.abs(r.xy) * lam[0] + r.ly * lam[1],
  Math.abs(r.xz) * lam[0] + Math.abs(r.yz) * lam[1] + r.lz * lam[2],
];

/** PPPM::estimate_ik_error (through pppmKspaceError, the same expression) for one edge of length prd. */
const estimateIk = (r: PppmGridRequest, g: number, h: number, prd: number, natoms: number): number =>
  pppmKspaceError(g, h, r.order, r.q2, natoms, prd);

/** PPPM::compute_gf_denom: the coefficients of the denominator polynomial (gf_b). */
export const gfDenomCoefficients = (order: number): number[] => {
  const b = new Array<number>(order).fill(0);
  b[0] = 1;
  for (let m = 1; m < order; m++) {
    let l = m;
    for (; l > 0; l--) b[l] = 4 * (b[l] * (l - m) * (l - m - 0.5) - b[l - 1] * (l - m - 1) * (l - m - 1));
    b[0] = 4 * (b[0] * (l - m) * (l - m - 0.5));
  }
  let ifact = 1;
  for (let k = 1; k < 2 * order; k++) ifact *= k;
  const gaminv = 1 / ifact;
  return b.map((v) => v * gaminv);
};

/**
 * PPPM::compute_qopt: the estimate sum that gives the analytic-differentiation force error for an orthogonal box
 * (edge lengths lx, ly, lzs with lzs = lz times the slab volfactor), mesh nx ny nz and G.
 */
export const computeQopt = (nx: number, ny: number, nz: number, g: number, lx: number, ly: number, lzs: number, order: number): number => {
  const unitkx = MY_2PI / lx, unitky = MY_2PI / ly, unitkz = MY_2PI / lzs;
  const twoorder = 2 * order;
  let qopt = 0;
  for (let m = 0; m < nz; m++) {
    const mper = m - nz * Math.trunc((2 * m) / nz);
    for (let l = 0; l < ny; l++) {
      const lper = l - ny * Math.trunc((2 * l) / ny);
      for (let k = 0; k < nx; k++) {
        const kper = k - nx * Math.trunc((2 * k) / nx);
        const sqk = (unitkx * kper) * (unitkx * kper) + (unitky * lper) * (unitky * lper) + (unitkz * mper) * (unitkz * mper);
        if (sqk === 0) continue;
        let sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
        for (let ix = -2; ix <= 2; ix++) {
          let qx = unitkx * (kper + nx * ix);
          const sx = libmExp(-0.25 * ((qx / g) * (qx / g)));
          const wx = powsinxx((0.5 * qx * lx) / nx, twoorder);
          qx *= qx;
          for (let iy = -2; iy <= 2; iy++) {
            let qy = unitky * (lper + ny * iy);
            const sy = libmExp(-0.25 * ((qy / g) * (qy / g)));
            const wy = powsinxx((0.5 * qy * ly) / ny, twoorder);
            qy *= qy;
            for (let iz = -2; iz <= 2; iz++) {
              let qz = unitkz * (mper + nz * iz);
              const sz = libmExp(-0.25 * ((qz / g) * (qz / g)));
              const wz = powsinxx((0.5 * qz * lzs) / nz, twoorder);
              qz *= qz;
              const dot2 = qx + qy + qz;
              const u1 = sx * sy * sz;
              const u2 = wx * wy * wz;
              sum1 += (u1 * u1) / dot2 * MY_4PI * MY_4PI;
              sum2 += u1 * u2 * MY_4PI;
              sum3 += u2;
              sum4 += dot2 * u2;
            }
          }
        }
        sum2 *= sum2;
        qopt += sum1 - sum2 / (sum3 * sum4);
      }
    }
  }
  return qopt;
};

/** Newton solver for the G-ewald root (adjust_gewald), with the native derivative step and stop. */
const adjustGewald = (g0: number, f: (g: number) => number): number => {
  let g = g0;
  for (let i = 0; i < LARGE; i++) {
    const f1 = f(g);
    const dfx = (f(g + 0.000001) - f1) / 0.000001;
    if (dfx === 0 || Number.isNaN(dfx)) break;
    let dx = f1 / dfx;
    while (g - dx <= 0) dx *= 0.5;
    g -= dx;
    if (Math.abs(f(g)) < SMALL) return g;
  }
  throw new StyleError('Could not compute g_ewald');
};

/** The initial G-ewald of PPPM::init (before the grid is chosen), from the accuracy and the orthogonal volume. */
const initialG = (r: PppmGridRequest, natoms: number): number => {
  const pre = (r.accuracy * Math.sqrt(natoms * r.cutoff * r.lx * r.ly * r.lz)) / (2 * r.q2);
  if (pre >= 1) return (1.35 - 0.15 * libmLog(r.accuracy)) / r.cutoff;
  return Math.sqrt(-libmLog(pre)) / r.cutoff;
};

/** The native automatic (or partly user-set) G-ewald and mesh for a triclinic box, or for diff ad on an orthogonal box. */
export const nativePppmGrid = (r: PppmGridRequest): PppmGridResult => {
  if (r.ad) return adGrid(r);
  return triclinicGrid(r);
};

const triclinicGrid = (r: PppmGridRequest): PppmGridResult => {
  const natoms = Math.max(1, r.natoms);
  const lzs = r.lz * r.slab;
  const gInit = r.gUser > 0 ? r.gUser : initialG(r, natoms);
  let nx: number, ny: number, nz: number;
  if (r.meshUser) [nx, ny, nz] = r.meshUser;
  else {
    // the orthogonal estimate: h = 1/G, n = int(prd/h) + 1, then grow n until the edge error meets the accuracy
    const axis = (prd: number): number => {
      let h = 1 / gInit;
      let n = Math.trunc(prd / h) + 1;
      let err = estimateIk(r, gInit, h, prd, natoms);
      let guard = 0;
      while (err > r.accuracy) {
        err = estimateIk(r, gInit, h, prd, natoms);
        n++;
        h = prd / n;
        if (++guard > 1000000) throw new StyleError('kspace_style pppm: the mesh search did not end');
      }
      return n;
    };
    nx = axis(r.lx);
    ny = axis(r.ly);
    nz = axis(lzs);
    // scale the grid for the triclinic skew (set_grid_global)
    const t = lamda2xT(r, [nx / r.lx, ny / r.ly, nz / r.lz]);
    nx = Math.trunc(t[0]) + 1;
    ny = Math.trunc(t[1]) + 1;
    nz = Math.trunc(t[2]) + 1;
  }
  while (!factorable(nx)) nx++;
  while (!factorable(ny)) ny++;
  while (!factorable(nz)) nz++;
  if (r.gUser > 0) return { g: r.gUser, mesh: [nx, ny, nz] };
  // the spacings of the final grid (x2lamdaT of the counts): fixed while G is solved
  const lam = x2lamdaT(r, [nx, ny, nz]);
  const hx = 1 / lam[0], hy = 1 / lam[1], hz = 1 / lam[2];
  const dfRspace = (g: number) => (2 * r.q2 * libmExp(-g * g * r.cutoff * r.cutoff)) / Math.sqrt(natoms * r.cutoff * r.lx * r.ly * r.lz);
  const dfKspace = (g: number) => {
    const a = estimateIk(r, g, hx, r.lx, natoms);
    const b = estimateIk(r, g, hy, r.ly, natoms);
    const c = estimateIk(r, g, hz, lzs, natoms);
    return Math.sqrt(a * a + b * b + c * c) / Math.sqrt(3);
  };
  const g = adjustGewald(gInit, (x) => dfRspace(x) - dfKspace(x));
  return { g, mesh: [nx, ny, nz] };
};

const adGrid = (r: PppmGridRequest): PppmGridResult => {
  const natoms = Math.max(1, r.natoms);
  const lzs = r.lz * r.slab;
  const V = r.lx * r.ly * lzs;
  const gInit = r.gUser > 0 ? r.gUser : initialG(r, natoms);
  const dfKspace = (g: number, nx: number, ny: number, nz: number) =>
    Math.sqrt(computeQopt(nx, ny, nz, g, r.lx, r.ly, lzs, r.order) / natoms) * r.q2 / V;
  let nx: number, ny: number, nz: number;
  if (r.meshUser) [nx, ny, nz] = r.meshUser;
  else {
    let h = (4 / gInit);
    let count = 0;
    for (;;) {
      nx = Math.trunc(r.lx / h);
      ny = Math.trunc(r.ly / h);
      nz = Math.trunc(lzs / h);
      if (nx <= 1) nx = 2;
      if (ny <= 1) ny = 2;
      if (nz <= 1) nz = 2;
      const df = dfKspace(gInit, nx, ny, nz);
      count++;
      if (df <= r.accuracy) break;
      if (count > 500) throw new StyleError('Could not compute grid size');
      h *= 0.95;
    }
  }
  while (!factorable(nx)) nx++;
  while (!factorable(ny)) ny++;
  while (!factorable(nz)) nz++;
  if (r.gUser > 0) return { g: r.gUser, mesh: [nx, ny, nz] };
  const dfRspace = (g: number) => (2 * r.q2 * libmExp(-g * g * r.cutoff * r.cutoff)) / Math.sqrt(natoms * r.cutoff * r.lx * r.ly * r.lz);
  const g = adjustGewald(gInit, (x) => dfRspace(x) - dfKspace(x, nx, ny, nz));
  return { g, mesh: [nx, ny, nz] };
};
