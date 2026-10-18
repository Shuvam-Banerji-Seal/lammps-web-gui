/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/pppm.cpp (compute_gf_ad, compute_sf_precoeff, poisson_ad, fieldforce_ad, the ad branch of
 * set_grid_global and adjust_gewald, and the ad branch of setup) at commit b332f28, on top of the core PPPM it extends
 * (src/engine/force/kspace/pppm.ts). PPPM contributing authors: Steve Plimpton (Sandia), Stan Moore (SNL), Paul Crozier
 * (Sandia), and the other LAMMPS developers. Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * kspace_modify diff ad: the PPPM force by analytic differentiation of the charge assignment (4 FFTs per step in the
 * native code: one forward transform of the charge mesh, one inverse transform of the potential, then the field is
 * the derivative of the interpolation weights). The self-force correction (sf_coeff) and the optimal influence
 * function for ad are the native ones. Orthogonal boxes only: the native code refuses ad with a triclinic box. The
 * mesh and G-ewald are chosen by nativePppmGrid (grid.ts).
 *
 * kspace_modify diff ad on a kspace_style pppm instance replaces it with this class (promoteAd). Everything else
 * (charges, neighbour, slab correction, energy constants) is the core's.
 */
import { KSpacePPPM, type PppmGplHook } from '../../engine/force/kspace/pppm';
import { StyleError, type KSpaceCompute, type StyleContext } from '../../engine/force/types';
import { fft3d } from '../../engine/force/fft';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { gfDenomCoefficients, nativePppmGrid } from './grid';
import { sin as libmSin } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';

const MY_2PI = 6.28318530717958647692;
const MY_4PI = 12.56637061435917295384;
const MY_PI = Math.PI;

/** math_special.h powsinxx (see grid.ts). */
const powsinxx = (x: number, n: number): number => {
  if (x === 0) return 1;
  let ww = libmSin(x) / x;
  let yy = 1;
  for (; n !== 0; n >>= 1, ww *= ww) if (n & 1) yy *= ww;
  return yy;
};

/** Cardinal B-spline M_p(x), support [0, p]. */
const bspline = (p: number, x: number): number => {
  if (p === 1) return x >= 0 && x < 1 ? 1 : 0;
  if (x <= 0 || x >= p) return 0;
  return (x * bspline(p - 1, x) + (p - x) * bspline(p - 1, x - 1)) / (p - 1);
};

/** d/dx of the cardinal B-spline: M_p'(x) = M_{p-1}(x) - M_{p-1}(x-1). */
const bsplineDeriv = (p: number, x: number): number => bspline(p - 1, x) - bspline(p - 1, x - 1);

/**
 * PPPM::compute_sf_precoeff: for every grid point (native order: m, l, k), the six sums of products of the
 * stencil weights of the order-P assignment with the shifted weights (one set per dimension and direction).
 */
const sfPrecoeff = (nx: number, ny: number, nz: number, order: number): Float64Array[] => {
  const ng = nx * ny * nz;
  const out = [0, 1, 2, 3, 4, 5].map(() => new Float64Array(ng));
  const wx0 = new Float64Array(5), wy0 = new Float64Array(5), wz0 = new Float64Array(5);
  const wx1 = new Float64Array(5), wy1 = new Float64Array(5), wz1 = new Float64Array(5);
  const wx2 = new Float64Array(5), wy2 = new Float64Array(5), wz2 = new Float64Array(5);
  let n = 0;
  for (let m = 0; m < nz; m++) {
    const mper = m - nz * Math.trunc((2 * m) / nz);
    for (let l = 0; l < ny; l++) {
      const lper = l - ny * Math.trunc((2 * l) / ny);
      for (let k = 0; k < nx; k++) {
        const kper = k - nx * Math.trunc((2 * k) / nx);
        for (let i = 0; i < 5; i++) {
          wx0[i] = powsinxx((0.5 * MY_2PI * (kper + nx * (i - 2))) / nx, order);
          wx1[i] = powsinxx((0.5 * MY_2PI * (kper + nx * (i - 1))) / nx, order);
          wx2[i] = powsinxx((0.5 * MY_2PI * (kper + nx * i)) / nx, order);
          wy0[i] = powsinxx((0.5 * MY_2PI * (lper + ny * (i - 2))) / ny, order);
          wy1[i] = powsinxx((0.5 * MY_2PI * (lper + ny * (i - 1))) / ny, order);
          wy2[i] = powsinxx((0.5 * MY_2PI * (lper + ny * i)) / ny, order);
          wz0[i] = powsinxx((0.5 * MY_2PI * (mper + nz * (i - 2))) / nz, order);
          wz1[i] = powsinxx((0.5 * MY_2PI * (mper + nz * (i - 1))) / nz, order);
          wz2[i] = powsinxx((0.5 * MY_2PI * (mper + nz * i)) / nz, order);
        }
        let s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
        for (let a = 0; a < 5; a++) {
          for (let b = 0; b < 5; b++) {
            for (let c = 0; c < 5; c++) {
              const u0 = wx0[a] * wy0[b] * wz0[c];
              s1 += u0 * (wx1[a] * wy0[b] * wz0[c]);
              s2 += u0 * (wx2[a] * wy0[b] * wz0[c]);
              s3 += u0 * (wx0[a] * wy1[b] * wz0[c]);
              s4 += u0 * (wx0[a] * wy2[b] * wz0[c]);
              s5 += u0 * (wx0[a] * wy0[b] * wz1[c]);
              s6 += u0 * (wx0[a] * wy0[b] * wz2[c]);
            }
          }
        }
        out[0][n] = s1; out[1][n] = s2; out[2][n] = s3; out[3][n] = s4; out[4][n] = s5; out[5][n] = s6;
        n++;
      }
    }
  }
  return out;
};

export class KSpacePPPMAd extends KSpacePPPM {
  private adGreen = new Float64Array(0);
  private adSf = new Float64Array(6);
  private adKey = '';

  override modify(key: string, values: string[]): number {
    if (key === 'diff') {
      if (values[0] === 'ad') return 1;
      throw new StyleError('kspace_modify diff ik after diff ad is not supported by the browser engine');
    }
    return super.modify(key, values);
  }

  override init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    // native: "Cannot (yet) use PPPM with triclinic box and kspace_modify diff ad" (pppm.cpp init)
    if (s.box.triclinic || geom.xy !== 0 || geom.xz !== 0 || geom.yz !== 0) {
      throw new StyleError('Cannot (yet) use PPPM with triclinic box and kspace_modify diff ad');
    }
    if (this.minorder > this.order) throw new StyleError(`PPPM order < minimum allowed order (kspace_modify minorder ${this.minorder} > order ${this.order})`);
    this.cutCoul = cutCoul;
    this.setupCharges(s, geom, ctx);
    if (this.qsqsum === 0 && this.gewaldUser <= 0) throw new StyleError("Must use 'kspace_modify gewald' for uncharged system");
    const r = nativePppmGrid({
      lx: geom.lx, ly: geom.ly, lz: geom.lz, xy: 0, xz: 0, yz: 0, natoms: this.natoms,
      q2: qqrd2e * this.qsqsum, accuracy: this.absAccuracy(s, qqrd2e), cutoff: cutCoul, order: this.order,
      gUser: this.gewaldUser, meshUser: this.meshUser, slab: this.slab, ad: true,
    });
    this.gEwald = r.g;
    this.mesh = r.mesh;
    this.adKey = '';
    ctx.log(`  G vector (1/distance) = ${this.gEwald}\n  grid = ${this.mesh.join(' ')}\n  stencil order = ${this.order}`);
  }

  /** Charges enter the mesh as in the core (pppm: every charge). */
  override compute(kc: KSpaceCompute): void {
    const s = kc.s;
    const geom = kc.geom;
    const n = s.n;
    const C = kc.qqrd2e;
    const [nx, ny, nz] = this.mesh;
    const P = this.order;
    const g = this.gEwald;
    this.refreshChargeSums(s);
    const lzs = geom.lz * this.slab;
    const V = geom.lx * geom.ly * lzs;
    this.volume = V;
    this.buildAd(geom, lzs);
    const ng = nx * ny * nz;
    const N = [nx, ny, nz];
    const lam = [0, 0, 0];
    // charge assignment: the first stencil index and, per dimension, the weights and their derivatives in u = N lambda
    const first = new Int32Array(3 * n);
    const wts = new Float64Array(3 * n * P);
    const dwts = new Float64Array(3 * n * P);
    for (let i = 0; i < n; i++) {
      geom.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
      lam[2] /= this.slab;
      for (let d = 0; d < 3; d++) {
        const u = lam[d] * N[d];
        const i0 = P % 2 === 1 ? Math.round(u) - (P - 1) / 2 : Math.floor(u) - P / 2 + 1;
        first[3 * i + d] = i0;
        for (let k = 0; k < P; k++) {
          const x = u - (i0 + k) + P / 2;
          wts[(3 * i + d) * P + k] = bspline(P, x);
          dwts[(3 * i + d) * P + k] = bsplineDeriv(P, x);
        }
      }
    }
    const mod = (a: number, m: number) => ((a % m) + m) % m;
    const qg = this.gridCharges(s);
    const rho = new Float64Array(2 * ng);
    for (let i = 0; i < n; i++) {
      const q = qg[i];
      if (q === 0) continue;
      const fx = first[3 * i], fy = first[3 * i + 1], fz = first[3 * i + 2];
      for (let c = 0; c < P; c++) {
        const iz = mod(fz + c, nz);
        const wz = q * wts[(3 * i + 2) * P + c];
        for (let b = 0; b < P; b++) {
          const iy = mod(fy + b, ny);
          const wyz = wz * wts[(3 * i + 1) * P + b];
          for (let a = 0; a < P; a++) {
            const ix = mod(fx + a, nx);
            rho[2 * ((iz * ny + iy) * nx + ix)] += wyz * wts[(3 * i) * P + a];
          }
        }
      }
    }
    fft3d(rho, nx, ny, nz, -1);
    // energy and virial from |rho(k)|^2 with the ad Green's function; the potential is G rho(k) back-transformed
    const G = this.adGreen;
    const pot = new Float64Array(2 * ng);
    // per-atom virial (stress/atom): as poisson_peratom does in the ad case, the six bricks G rho vg_j; the per-atom
    // energy uses the potential brick of the forces (poisson_ad builds u_brick, poisson_peratom skips it for ad)
    const vb = kc.vatom ? Array.from({ length: 6 }, () => new Float64Array(2 * ng)) : null;
    let e = 0;
    const w = [0, 0, 0, 0, 0, 0];
    for (let iz = 0; iz < nz; iz++) {
      const mz = iz < nz / 2 ? iz : iz - nz;
      for (let iy = 0; iy < ny; iy++) {
        const my = iy < ny / 2 ? iy : iy - ny;
        for (let ix = 0; ix < nx; ix++) {
          const mx = ix < nx / 2 ? ix : ix - nx;
          const idx = (iz * ny + iy) * nx + ix;
          const Gk = G[idx];
          if (Gk === 0) continue;
          const re = rho[2 * idx], im = rho[2 * idx + 1];
          const r2 = re * re + im * im;
          const kx = (MY_2PI * mx) / geom.lx, ky = (MY_2PI * my) / geom.ly, kz = (MY_2PI * mz) / lzs;
          const k2 = kx * kx + ky * ky + kz * kz;
          e += Gk * r2;
          const b = 2 * (1 / k2 + 1 / (4 * g * g));
          const t = Gk * r2;
          w[0] += t * (1 - b * kx * kx); w[1] += t * (1 - b * ky * ky); w[2] += t * (1 - b * kz * kz);
          w[3] += t * (-b * kx * ky); w[4] += t * (-b * kx * kz); w[5] += t * (-b * ky * kz);
          pot[2 * idx] = Gk * re;
          pot[2 * idx + 1] = Gk * im;
          if (vb) {
            const vg = [1 - b * kx * kx, 1 - b * ky * ky, 1 - b * kz * kz, -b * kx * ky, -b * kx * kz, -b * ky * kz];
            for (let c = 0; c < 6; c++) { vb[c][2 * idx] = Gk * re * vg[c]; vb[c][2 * idx + 1] = Gk * im * vg[c]; }
          }
        }
      }
    }
    fft3d(pot, nx, ny, nz, 1);
    if (vb) for (const v of vb) fft3d(v, nx, ny, nz, 1);
    // field: E = -grad phi with phi = (1/V) (unnormalised inverse transform); the grid derivative of the weights gives
    // the gradient. The z field uses nz / zprd (not zprd times the slab volfactor), as in the native fieldforce_ad.
    const hx = nx / geom.lx, hy = ny / geom.ly, hz = nz / geom.lz;
    const qfactor = C;
    const sfc = this.adSf;
    for (let i = 0; i < n; i++) {
      const q = qg[i];
      if (q === 0) continue;
      const fx0 = first[3 * i], fy0 = first[3 * i + 1], fz0 = first[3 * i + 2];
      let Ex = 0, Ey = 0, Ez = 0;
      for (let c = 0; c < P; c++) {
        const iz = mod(fz0 + c, nz);
        for (let b = 0; b < P; b++) {
          const iy = mod(fy0 + b, ny);
          for (let a = 0; a < P; a++) {
            const ix = mod(fx0 + a, nx);
            const idx = 2 * ((iz * ny + iy) * nx + ix);
            const phi = pot[idx] / V;
            const wx = wts[(3 * i) * P + a], wy = wts[(3 * i + 1) * P + b], wz = wts[(3 * i + 2) * P + c];
            const dx = dwts[(3 * i) * P + a], dy = dwts[(3 * i + 1) * P + b], dz = dwts[(3 * i + 2) * P + c];
            Ex += dx * wy * wz * phi;
            Ey += wx * dy * wz * phi;
            Ez += wx * wy * dz * phi;
          }
        }
      }
      // the weights are functions of u = N lambda, so d/dx = (N / L) d/du; E = -grad phi
      Ex *= -hx; Ey *= -hy; Ez *= -hz;
      // self-force correction (fieldforce_ad): positions enter unshifted, as in the native code
      const x = s.x[3 * i], y = s.x[3 * i + 1], z = s.x[3 * i + 2];
      const sx = this.selfForce(sfc, 0, x * hx, q);
      const sy = this.selfForce(sfc, 2, y * hy, q);
      const sz = this.selfForce(sfc, 4, z * hz, q);
      kc.f[3 * i] += qfactor * (Ex * q - sx);
      kc.f[3 * i + 1] += qfactor * (Ey * q - sy);
      kc.f[3 * i + 2] += qfactor * (Ez * q - sz);
    }
    const pref = C / (2 * V);
    if (kc.eatom || kc.vatom) this.peratomMesh(kc, qg, first, wts, kc.eatom ? pot : null, vb, pref);
    kc.acc.elong += pref * e + this.constantEnergy(C);
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += pref * w[c];
    this.slabCorrection(kc);
  }

  /** sf_coeff[c] sin(2 pi s) + sf_coeff[c+1] sin(4 pi s), times 2 q^2 (fieldforce_ad). */
  private selfForce(sfc: Float64Array, c: number, s: number, q: number): number {
    let sf = sfc[c] * libmSin(2 * MY_PI * s);
    sf += sfc[c + 1] * libmSin(4 * MY_PI * s);
    return sf * 2 * q * q;
  }

  /** The ad Green's function and the self-force coefficients for the current box, mesh and G (cached). */
  private buildAd(geom: Geometry, lzs: number): void {
    const [nx, ny, nz] = this.mesh;
    const P = this.order;
    const g = this.gEwald;
    const key = [geom.lx, geom.ly, lzs, geom.lz, nx, ny, nz, P, g].join(',');
    if (key === this.adKey) return;
    this.adKey = key;
    const V = geom.lx * geom.ly * lzs;
    const unitkx = MY_2PI / geom.lx, unitky = MY_2PI / geom.ly, unitkz = MY_2PI / lzs;
    const gfb = gfDenomCoefficients(P);
    const denom = (x: number, y: number, z: number): number => {
      let sx = 0, sy = 0, sz = 0;
      for (let l = P - 1; l >= 0; l--) {
        sx = gfb[l] + sx * x;
        sy = gfb[l] + sy * y;
        sz = gfb[l] + sz * z;
      }
      const s = sx * sy * sz;
      return s * s;
    };
    const ng = nx * ny * nz;
    const green = new Float64Array(ng);
    const twoorder = 2 * P;
    let n = 0;
    const sfPre = sfPrecoeff(nx, ny, nz, P);
    for (let m = 0; m < nz; m++) {
      const mper = m - nz * Math.trunc((2 * m) / nz);
      const qz = unitkz * mper;
      const snz = libmSin((0.5 * qz * lzs) / nz) ** 2;
      const sz = libmExp(-0.25 * ((qz / g) * (qz / g)));
      const wz = powsinxx((0.5 * qz * lzs) / nz, twoorder);
      for (let l = 0; l < ny; l++) {
        const lper = l - ny * Math.trunc((2 * l) / ny);
        const qy = unitky * lper;
        const sny = libmSin((0.5 * qy * geom.ly) / ny) ** 2;
        const sy = libmExp(-0.25 * ((qy / g) * (qy / g)));
        const wy = powsinxx((0.5 * qy * geom.ly) / ny, twoorder);
        for (let k = 0; k < nx; k++) {
          const kper = k - nx * Math.trunc((2 * k) / nx);
          const qx = unitkx * kper;
          const snx = libmSin((0.5 * qx * geom.lx) / nx) ** 2;
          const sx = libmExp(-0.25 * ((qx / g) * (qx / g)));
          const wx = powsinxx((0.5 * qx * geom.lx) / nx, twoorder);
          const sqk = qx * qx + qy * qy + qz * qz;
          if (sqk !== 0) {
            const numerator = MY_4PI / sqk;
            green[n] = (numerator * sx * sy * sz * wx * wy * wz) / denom(snx, sny, snz);
          } else green[n] = 0;
          n++;
        }
      }
    }
    this.adGreen = green;
    // self-force coefficients: sums of the precoefficients with the Green's function, scaled per direction
    const sf = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < ng; i++) for (let c = 0; c < 6; c++) sf[c] += sfPre[c][i] * green[i];
    const prex = (MY_PI / V) * (nx / geom.lx), prey = (MY_PI / V) * (ny / geom.ly), prez = (MY_PI / V) * (nz / lzs);
    sf[0] *= prex; sf[1] *= prex * 2;
    sf[2] *= prey; sf[3] *= prey * 2;
    sf[4] *= prez; sf[5] *= prez * 2;
    this.adSf = Float64Array.from(sf);
  }
}

/**
 * kspace_modify diff ad on a kspace_style pppm instance: the analytic-differentiation style with the same settings
 * (the instance's fields are copied). Returns null for other styles (pppm/cg, the electrode and dielectric variants),
 * and the instance itself when it is already ad.
 */
export const promoteAd = (base: KSpacePPPM): KSpacePPPM | null => {
  if (base instanceof KSpacePPPMAd) return base;
  if (base.name !== 'pppm' || base.constructor !== KSpacePPPM) return null;
  return Object.assign(new KSpacePPPMAd(), base);
};

/** The hook the core reads (src/engine/force/kspace/pppmHook.ts). */
export const PPPM_HOOK: PppmGplHook = { grid: nativePppmGrid, promoteAd };
