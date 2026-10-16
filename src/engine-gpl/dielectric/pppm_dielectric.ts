/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/pppm_dielectric.cpp and pppm_dielectric.h at commit b332f28 (Trung Nguyen and Monica Olvera de la Cruz, Northwestern
 * University, and the DIELECTRIC package authors), with the grid choice of src/KSPACE/pppm.cpp (set_grid_global,
 * adjust_gewald, estimate_ik_error, compute_df_kspace) at the same commit (Roy Pollock, LLNL, and Paul Crozier, SNL, and the LAMMPS developers). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * kspace_style pppm/dielectric: PPPM for the dielectric atoms. The mesh charges are the scaled charges q_scaled (the
 * induced surface charges of the interface patches included); the forces and the field are scaled by the local
 * dielectric constant epsilon_i; the energy is that of the unscaled charges times C / <epsilon>, with <epsilon> the mean
 * over all atoms; the slab correction uses the scaled charges weighted by epsilon. The grid and g_ewald follow the
 * LAMMPS choice with the charge sum q2 = qqrd2e * sum(q_i^2 / epsilon_i) (qsum_qsq) and the slab-extended z edge for the
 * mesh (pppm.cpp set_grid_global: the real-space error uses the box volume, the mesh the extended edge).
 *
 * The mesh solve is the engine's spectral PPPM (Hockney and Eastwood assignment, optimal influence function for ik
 * differentiation, the same formulas as core kspace pppm), written here so that the grid, the charges and the field
 * can be taken from the scaled and unscaled charges separately. The per-atom field (efield) is written as LAMMPS does
 * (scale * epsilon_i * E / V) and read by the polarization fixes. Not ported: pppm/dielectric's per-atom energy and
 * virial (compute pe/atom, stress/atom with this kspace style are refused), triclinic boxes, and kspace_modify diff ad.
 */
import { KSpacePPPM } from '../../engine/force/kspace/pppm';
import { pppmKspaceError, pppmRealError } from '../../engine/force/kspace/pppm';
import { invertUpper } from '../../engine/force/kspace/ewald';
import { fft3d, isFFTSize } from '../../engine/force/fft';
import { StyleError, type KSpaceCompute, type StyleContext } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { dielArrays } from './common';
import { log as libmLog } from '../libm/log';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';

/** LAMMPS pppm.cpp SMALL: the convergence of adjust_gewald. */
const SMALL = 0.00001;
const LARGE = 10000;
/** Largest number of mesh points tried per edge before the grid choice is refused. */
const MAX_MESH = 100000;
/** Alias images in the optimal influence function (pppm.ts ALIAS_C, measured against native LAMMPS). */
const ALIAS_C = Math.pow(7 * Math.LN10, 0.25);

/** Cardinal B-spline M_p(x), support [0, p] (the charge assignment function of PPPM). */
const bspline = (p: number, x: number): number => {
  if (p === 1) return x >= 0 && x < 1 ? 1 : 0;
  if (x <= 0 || x >= p) return 0;
  return (x * bspline(p - 1, x) + (p - x) * bspline(p - 1, x - 1)) / (p - 1);
};

export class KSpacePPPMDielectric extends KSpacePPPM {
  override readonly name: string = 'pppm/dielectric';
  /** Electric field of the mesh at each owned atom, LAMMPS efield: scale * epsilon_i * E / V (3 per atom). */
  efield = new Float64Array(0);
  private dieMesh: [number, number, number] = [0, 0, 0];
  private dieGreen = new Float64Array(0);
  private dieGreenKey = '';

  /** Grid and g_ewald (pppm.cpp set_grid_global and adjust_gewald) with q2 = qqrd2e sum q^2/epsilon. */
  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    if (this.slabAuto) throw new StyleError('kspace_modify slab auto is not supported by kspace_style pppm/dielectric');
    if (s.box.triclinic) throw new StyleError('kspace_style pppm/dielectric is not supported for triclinic boxes by the browser engine');
    if (this.minorder > this.order) throw new StyleError(`PPPM order < minimum allowed order (kspace_modify minorder ${this.minorder} > order ${this.order})`);
    this.cutCoul = cutCoul;
    this.setupCharges(s, geom, ctx);
    const d = dielArrays(s, 'kspace pppm/dielectric');
    let qsqe = 0;
    for (let i = 0; i < s.n; i++) qsqe += (s.q[i] * s.q[i]) / d.epsilon[i];
    const q2 = qqrd2e * qsqe;
    const natoms = Math.max(1, s.n);
    const lx = geom.lx, ly = geom.ly, lz = geom.lz;
    const zSlab = lz * this.slab;
    const rc = cutCoul;
    const accE = this.absAccuracy(s, qqrd2e);
    if (!(this.gewaldUser > 0) && q2 === 0) throw new StyleError("Must use 'kspace_modify gewald' for uncharged system");
    // initial estimate with the box volume (no slab extension): g = acc / (2 q2 / sqrt(N rc V)), pppm.cpp set_grid_global
    const Vbox = lx * ly * lz;
    const ratio = accE / ((2 * q2) / Math.sqrt(natoms * rc * Vbox));
    const g0 = ratio >= 1 ? (1.35 - 0.15 * libmLog(accE)) / rc : Math.sqrt(-libmLog(ratio)) / rc;
    let g = this.gewaldUser > 0 ? this.gewaldUser : g0;
    // mesh: per edge the smallest size meeting the error of pppm.cpp estimate_ik_error, then the next factorable size
    const L = [lx, ly, zSlab];
    const mesh: [number, number, number] = [0, 0, 0];
    for (let dd = 0; dd < 3; dd++) {
      if (this.meshUser) { mesh[dd] = this.meshUser[dd]; continue; }
      // set_grid_global (ik branch): start at int(L / h) + 1 with h = 1/g, then grow while the estimate is above the accuracy
      let h = 1 / g;
      let nn = Math.trunc(L[dd] / h) + 1;
      let err = pppmKspaceError(g, h, this.order, q2, natoms, L[dd]);
      let guard = 0;
      while (err > accE) {
        err = pppmKspaceError(g, h, this.order, q2, natoms, L[dd]);
        nn++;
        h = L[dd] / nn;
        if (++guard > MAX_MESH) throw new StyleError('kspace_style pppm/dielectric: no mesh meets the accuracy');
      }
      while (!isFFTSize(nn)) nn++;
      mesh[dd] = nn;
    }
    this.dieMesh = mesh;
    const kspaceRms = (gg: number): number => {
      let sum = 0;
      for (let dd = 0; dd < 3; dd++) {
        const e = pppmKspaceError(gg, L[dd] / mesh[dd], this.order, q2, natoms, L[dd]);
        sum += e * e;
      }
      return Math.sqrt(sum / 3);
    };
    // adjust_gewald: Newton-Raphson on the real-space minus reciprocal-space error (not with kspace_modify gewald)
    if (!(this.gewaldUser > 0)) {
      const f = (gg: number) => pppmRealError(gg, rc, q2, natoms, Vbox) - kspaceRms(gg);
      let converged = false;
      for (let it = 0; it < LARGE; it++) {
        const f1 = f(g);
        const h = 0.000001;
        const dfx = (f(g + h) - f1) / h;
        if (dfx === 0 || Number.isNaN(dfx)) break;
        let dx = f1 / dfx;
        while (g - dx <= 0) dx *= 0.5;
        g -= dx;
        if (Math.abs(f(g)) < SMALL) { converged = true; break; }
      }
      if (!converged) throw new StyleError('Could not compute g_ewald');
    }
    this.gEwald = g;
    this.dieGreenKey = '';
    this.volume = geom.volume(3) * this.slab;
    ctx.log(`  G vector (1/distance) = ${g}\n  grid = ${mesh.join(' ')}\n  stencil order = ${this.order}`);
  }

  /** Charges, and the sums of the unscaled charges (qsum, qsqsum are those of q, as in LAMMPS qsum_qsq). */
  compute(kc: KSpaceCompute): void {
    // the per-atom energy and virial of this style are not tallied (LAMMPS PPPMDielectric::fieldforce_peratom is not ported)
    if (kc.eatom || kc.vatom) throw new StyleError('compute pe/atom and stress/atom with kspace_style pppm/dielectric are not supported by the browser engine');
    const s = kc.s, geom = kc.geom, n = s.n;
    const d = dielArrays(s, 'kspace pppm/dielectric');
    const eps = d.epsilon, qs = d.qs, q = s.q;
    if (this.efield.length < 3 * n) this.efield = new Float64Array(3 * n);
    if (this.qsqsum === 0) { this.efield.fill(0); return; }
    let epsSum = 0;
    for (let i = 0; i < n; i++) epsSum += eps[i];
    const epsAve = epsSum / Math.max(1, n);
    const C = kc.qqrd2e;
    const P = this.order;
    const g = this.gEwald;
    const [nx, ny, nz] = this.dieMesh;
    const V = geom.volume(3) * this.slab;
    this.volume = V;
    const ng = nx * ny * nz;
    this.buildDieGreen(geom);
    // weights of the charge assignment (independent of the charges)
    const first = new Int32Array(3 * n);
    const wts = new Float64Array(3 * n * P);
    const lam = [0, 0, 0];
    const N = [nx, ny, nz];
    for (let i = 0; i < n; i++) {
      geom.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
      lam[2] /= this.slab;
      for (let dd = 0; dd < 3; dd++) {
        const u = lam[dd] * N[dd];
        const i0 = P % 2 === 1 ? Math.round(u) - (P - 1) / 2 : Math.floor(u) - P / 2 + 1;
        first[3 * i + dd] = i0;
        for (let k = 0; k < P; k++) wts[(3 * i + dd) * P + k] = bspline(P, u - (i0 + k) + P / 2);
      }
    }
    const mod = (a: number, m: number) => ((a % m) + m) % m;
    const grid = (charges: Float64Array): Float64Array => {
      const rho = new Float64Array(2 * ng);
      for (let i = 0; i < n; i++) {
        const qi = charges[i];
        if (qi === 0) continue;
        const fx = first[3 * i], fy = first[3 * i + 1], fz = first[3 * i + 2];
        for (let c = 0; c < P; c++) {
          const iz = mod(fz + c, nz);
          const wz = qi * wts[(3 * i + 2) * P + c];
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
      return rho;
    };
    const inv = invertUpper(geom.lx, geom.xy, geom.xz, geom.ly, geom.yz, geom.lz * this.slab);
    const kvec = (mx: number, my: number, mz: number): [number, number, number] => [
      2 * Math.PI * (mx * inv[0][0] + my * inv[1][0] + mz * inv[2][0]),
      2 * Math.PI * (mx * inv[0][1] + my * inv[1][1] + mz * inv[2][1]),
      2 * Math.PI * (mx * inv[0][2] + my * inv[1][2] + mz * inv[2][2]),
    ];
    // forces and field from the scaled charges (ik differentiation)
    const rhoS = grid(qs);
    const ex = new Float64Array(2 * ng), ey = new Float64Array(2 * ng), ez = new Float64Array(2 * ng);
    for (let iz = 0; iz < nz; iz++) {
      const mz = iz < nz / 2 ? iz : iz - nz;
      for (let iy = 0; iy < ny; iy++) {
        const my = iy < ny / 2 ? iy : iy - ny;
        for (let ix = 0; ix < nx; ix++) {
          const mx = ix < nx / 2 ? ix : ix - nx;
          const idx = (iz * ny + iy) * nx + ix;
          const G = this.dieGreen[idx];
          if (G === 0) continue;
          const re = rhoS[2 * idx], im = rhoS[2 * idx + 1];
          const [kx, ky, kz] = kvec(mx, my, mz);
          ex[2 * idx] = kx * G * im; ex[2 * idx + 1] = -kx * G * re;
          ey[2 * idx] = ky * G * im; ey[2 * idx + 1] = -ky * G * re;
          ez[2 * idx] = kz * G * im; ez[2 * idx + 1] = -kz * G * re;
        }
      }
    }
    fft3d(ex, nx, ny, nz, 1);
    fft3d(ey, nx, ny, nz, 1);
    fft3d(ez, nx, ny, nz, 1);
    const pre0 = 1 / V;
    for (let i = 0; i < n; i++) {
      const fx0 = first[3 * i], fy0 = first[3 * i + 1], fz0 = first[3 * i + 2];
      let Ex = 0, Ey = 0, Ez = 0;
      for (let c = 0; c < P; c++) {
        const iz = mod(fz0 + c, nz);
        const wz = wts[(3 * i + 2) * P + c];
        for (let b = 0; b < P; b++) {
          const iy = mod(fy0 + b, ny);
          const wyz = wz * wts[(3 * i + 1) * P + b];
          for (let a = 0; a < P; a++) {
            const ix = mod(fx0 + a, nx);
            const ww = wyz * wts[(3 * i) * P + a];
            const idx = 2 * ((iz * ny + iy) * nx + ix);
            Ex += ww * ex[idx]; Ey += ww * ey[idx]; Ez += ww * ez[idx];
          }
        }
      }
      // LAMMPS efield = scale * eps * ek with ek = E / V; force = qqrd2e * eps * q_scaled * ek
      const ef = eps[i] * pre0;
      this.efield[3 * i] = ef * Ex; this.efield[3 * i + 1] = ef * Ey; this.efield[3 * i + 2] = ef * Ez;
      const qfac = C * eps[i] * qs[i] * pre0;
      if (qs[i] !== 0) {
        kc.f[3 * i] += qfac * Ex; kc.f[3 * i + 1] += qfac * Ey; kc.f[3 * i + 2] += qfac * Ez;
      }
    }
    // energy: the unscaled charges, times C / <epsilon> (LAMMPS: energy *= qscale / epsilon_ave)
    const rhoQ = grid(q);
    const Cn = C / epsAve;
    let e = 0;
    const w = [0, 0, 0, 0, 0, 0];
    for (let iz = 0; iz < nz; iz++) {
      const mz = iz < nz / 2 ? iz : iz - nz;
      for (let iy = 0; iy < ny; iy++) {
        const my = iy < ny / 2 ? iy : iy - ny;
        for (let ix = 0; ix < nx; ix++) {
          const mx = ix < nx / 2 ? ix : ix - nx;
          const idx = (iz * ny + iy) * nx + ix;
          const G = this.dieGreen[idx];
          if (G === 0) continue;
          const re = rhoQ[2 * idx], im = rhoQ[2 * idx + 1];
          const r2 = re * re + im * im;
          const [kx, ky, kz] = kvec(mx, my, mz);
          const k2 = kx * kx + ky * ky + kz * kz;
          e += G * r2;
          const b = 2 * (1 / k2 + 1 / (4 * g * g));
          const t = G * r2;
          w[0] += t * (1 - b * kx * kx); w[1] += t * (1 - b * ky * ky); w[2] += t * (1 - b * kz * kz);
          w[3] += t * (-b * kx * ky); w[4] += t * (-b * kx * kz); w[5] += t * (-b * ky * kz);
        }
      }
    }
    const pref = Cn / (2 * V);
    kc.acc.elong += pref * e + this.constantEnergy(Cn);
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += pref * w[c];
    this.dielectricSlabCorrection(kc, eps, qs, C, V);
  }

  /**
   * LAMMPS pppm_dielectric.cpp slabcorr: dipole sums over epsilon_i q_scaled,i z_i; the energy is C times the slab term
   * (not divided by the mean epsilon); the force is scaled by epsilon_i; the field z-component is not (LAMMPS).
   */
  private dielectricSlabCorrection(kc: KSpaceCompute, eps: Float64Array, qs: Float64Array, C: number, V: number): void {
    if (this.slab <= 1) return;
    const s = kc.s, n = s.n;
    const zprdSlab = (s.box.hi[2] - s.box.lo[2]) * this.slab;
    let dipole = 0, dipoleR2 = 0;
    for (let i = 0; i < n; i++) dipole += eps[i] * qs[i] * s.x[3 * i + 2];
    for (let i = 0; i < n; i++) dipoleR2 += eps[i] * qs[i] * s.x[3 * i + 2] * s.x[3 * i + 2];
    const qsum = this.qsum;
    const eSlab = (2 * Math.PI) * (dipole * dipole - qsum * dipoleR2 - qsum * qsum * zprdSlab * zprdSlab / 12.0) / V;
    kc.acc.elong += C * eSlab;
    const ffact = C * (-4.0 * Math.PI / V);
    for (let i = 0; i < n; i++) {
      kc.f[3 * i + 2] += ffact * eps[i] * qs[i] * (dipole - qsum * s.x[3 * i + 2]);
      this.efield[3 * i + 2] += ffact * (dipole - qsum * s.x[3 * i + 2]);
    }
  }

  /** Optimal influence function of the dielectric mesh (the core pppm formulas, with this grid). */
  private buildDieGreen(geom: Geometry): void {
    const [nx, ny, nz] = this.dieMesh;
    const lz = geom.lz * this.slab;
    const key = [geom.lx, geom.ly, lz, geom.xy, geom.xz, geom.yz, nx, ny, nz, this.order, this.gEwald].join(',');
    if (key === this.dieGreenKey) return;
    this.dieGreenKey = key;
    const inv = invertUpper(geom.lx, geom.xy, geom.xz, geom.ly, geom.yz, lz);
    const g = this.gEwald;
    const P = this.order;
    const N = [nx, ny, nz];
    const green = new Float64Array(nx * ny * nz);
    const kvec = (a: number, b: number, c: number, out: number[]) => {
      out[0] = 2 * Math.PI * (a * inv[0][0] + b * inv[1][0] + c * inv[2][0]);
      out[1] = 2 * Math.PI * (a * inv[0][1] + b * inv[1][1] + c * inv[2][1]);
      out[2] = 2 * Math.PI * (a * inv[0][2] + b * inv[1][2] + c * inv[2][2]);
    };
    const sinc = (x: number) => (x === 0 ? 1 : libmSin(x) / x);
    const u2 = (m: number, nn: number) => Math.pow(sinc((Math.PI * m) / nn), 2 * P);
    const edge = [Math.abs(geom.lx), Math.abs(geom.xy) + Math.abs(geom.ly), Math.abs(geom.xz) + Math.abs(geom.yz) + Math.abs(lz)];
    const nbImg = [0, 1, 2].map((dd) => Math.floor(((g * edge[dd]) / (Math.PI * N[dd])) * ALIAS_C));
    const den1 = N.map((nn) => {
      const out = new Float64Array(nn);
      for (let i = 0; i < nn; i++) {
        for (let j = 1 - P; j < P; j++) out[i] += bspline(2 * P, P + j) * libmCos((2 * Math.PI * j * i) / nn);
      }
      return out;
    });
    const k = [0, 0, 0], km = [0, 0, 0];
    for (let iz = 0; iz < nz; iz++) {
      const mz = iz < nz / 2 ? iz : iz - nz;
      for (let iy = 0; iy < ny; iy++) {
        const my = iy < ny / 2 ? iy : iy - ny;
        for (let ix = 0; ix < nx; ix++) {
          const mx = ix < nx / 2 ? ix : ix - nx;
          const idx = (iz * ny + iy) * nx + ix;
          if (mx === 0 && my === 0 && mz === 0) { green[idx] = 0; continue; }
          kvec(mx, my, mz, k);
          const k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
          let num = 0;
          for (let ax = -nbImg[0]; ax <= nbImg[0]; ax++) {
            const nxm = mx + ax * nx;
            const ux = u2(nxm, nx);
            for (let ay = -nbImg[1]; ay <= nbImg[1]; ay++) {
              const nym = my + ay * ny;
              const uy = u2(nym, ny);
              for (let az = -nbImg[2]; az <= nbImg[2]; az++) {
                const nzm = mz + az * nz;
                kvec(nxm, nym, nzm, km);
                const km2 = km[0] * km[0] + km[1] * km[1] + km[2] * km[2];
                num += ux * uy * u2(nzm, nz) * ((k[0] * km[0] + k[1] * km[1] + k[2] * km[2]) / km2) * 4 * Math.PI * libmExp(-km2 / (4 * g * g));
              }
            }
          }
          const den = den1[0][ix] * den1[1][iy] * den1[2][iz];
          green[idx] = num / (k2 * den * den);
        }
      }
    }
    this.dieGreen = green;
  }
}
