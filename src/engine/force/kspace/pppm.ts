import { StyleError, type KSpaceCompute, type StyleContext } from '../types';
import type { SimState } from '../../types';
import type { Geometry } from '../../domain';
import { parseNum } from '../util';
import { KSpaceBase, invertUpper } from './ewald';
import { fft3d, isFFTSize, nextFFTSize } from '../fft';

/*
 * kspace_style pppm accuracy — docs.lammps.org/kspace_style.html: "The pppm
 * style invokes a particle-particle particle-mesh solver (Hockney) which maps
 * atom charge to a 3d mesh, uses 3d FFTs to solve Poisson's equation on the
 * mesh, then interpolates electric fields on the mesh points back to the
 * atoms." kspace_modify.html: "The ik approach is the default for PPPM and is
 * the original formulation used in (Hockney). It performs differentiation in
 * Kspace, and uses 3 FFTs to transfer each component of the computed fields
 * back to real space"; "The mesh keyword sets the grid size ... each
 * dimension must be factorizable into powers of 2, 3, and 5"; "The order
 * keyword determines how many grid spacings an atom's charge extends when it
 * is mapped to the grid ... The default for this parameter is 5 for PPPM ...
 * For PPPM, the minimum allowed setting is 2 and the maximum allowed setting
 * is 7."
 *
 * Written from Hockney & Eastwood (1988) and Deserno & Holm, J Chem Phys 109,
 * 7678 (1998): charges are assigned to the mesh in fractional coordinates
 * with the cardinal B-spline of order P (the Hockney-Eastwood assignment
 * function); the mesh charge is Fourier transformed; with k = 2 pi H^-T n
 * and aliases k_m = 2 pi H^-T (n + m N), U(k) the Fourier transform of the
 * assignment function and the reference kernel 4 pi / k^2 exp(-k^2/(4 g^2)),
 * the optimal influence function for ik differentiation is
 *   G(k) = sum_m U^2(k_m) (k . k_m) / k_m^2 4 pi exp(-k_m^2/(4 g^2))
 *          / (k^2 (sum_m U^2(k_m))^2),
 *   E    = C / (2 V) sum_k G(k) |rho(k)|^2 + self + background (as Ewald),
 *   E_a  = inverse FFT of -i k_a G rho / V, interpolated back with the same
 *          weights; F_j = C q_j E(r_j),
 *   W_ab = C / (2 V) sum_k G |rho|^2 (delta_ab - 2 k_a k_b (1/k^2 + 1/(4g^2))).
 * Without kspace_modify mesh, the mesh is chosen so that (h g)^P times the
 * smooth-part estimate stays below the requested accuracy; the result can
 * differ from LAMMPS's own choice — set mesh, order and gewald to reproduce a
 * LAMMPS run.
 */

/*
 * Alias images in the numerator of G(k): per mesh dimension d,
 *   n_d = floor(g h_d / pi * ALIAS_C),  h_d = (L1 norm of box edge d) / N_d,
 * with ALIAS_C = (ln 1e7)^(1/4), i.e. images m with (pi m / (g h))^4 <= ln 1e7.
 * Not documented; measured against native LAMMPS (2 Sep 2026) by scanning
 * g_ewald on a coarse mesh: the switch points bracket ALIAS_C in
 * (2.0030, 2.0044] (orthogonal, both the 0->1 and 1->2 switch), and in
 * triclinic boxes the switch points of all three dimensions follow the L1
 * norms lx, |xy| + ly, |xz| + |yz| + lz of the edge vectors (two tilt sets,
 * predicted switches confirmed to 0.005 in g). Oracle case pppm_slab is the
 * one that depends on it (it uses no images; using one changes pzz by 5e-7).
 */
const ALIAS_C = Math.pow(7 * Math.LN10, 0.25);

/** Cardinal B-spline M_p(x), support [0, p]. */
const bspline = (p: number, x: number): number => {
  if (p === 1) return x >= 0 && x < 1 ? 1 : 0;
  if (x <= 0 || x >= p) return 0;
  return (x * bspline(p - 1, x) + (p - x) * bspline(p - 1, x - 1)) / (p - 1);
};

export class KSpacePPPM extends KSpaceBase {
  readonly name = 'pppm';
  meshUser: [number, number, number] | null = null;
  order = 5;
  private mesh: [number, number, number] = [0, 0, 0];
  private green = new Float64Array(0);
  private greenBox = '';

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'mesh': {
        const m = [0, 1, 2].map((d) => parseNum(values[d], 'mesh'));
        if (m.every((v) => v === 0)) { this.meshUser = null; return 3; }
        if (m.some((v) => !isFFTSize(v))) throw new StyleError('kspace_modify mesh: each dimension must factor into 2, 3 and 5');
        this.meshUser = m as [number, number, number];
        return 3;
      }
      case 'order': {
        const o = parseNum(values[0], 'order');
        if (!Number.isInteger(o) || o < 2 || o > 7) throw new StyleError('kspace_modify order for pppm must be an integer from 2 to 7');
        this.order = o;
        return 1;
      }
      case 'diff':
        if (values[0] !== 'ik') throw new StyleError('kspace_modify diff ad is not supported by the browser engine (ik is)');
        return 1;
      case 'kmax/ewald': throw new StyleError('kspace_modify kmax/ewald applies to ewald, not pppm');
      default: return super.modify(key, values);
    }
  }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    this.cutCoul = cutCoul;
    this.setupCharges(s, geom, ctx);
    this.gEwald = this.chooseGEwald(s, qqrd2e);
    if (this.meshUser) this.mesh = [...this.meshUser];
    else {
      // mesh spacing h with (h g)^P ~ relative accuracy, scaled for the smooth part
      const L = [geom.lx, geom.ly, geom.lz * this.slab];
      const P = this.order;
      const hg = Math.pow(Math.max(1e-12, this.accuracy), 1 / P) * 2.0;
      const h = hg / this.gEwald;
      this.mesh = L.map((len) => nextFFTSize(Math.max(2 * P, Math.ceil(len / h)))) as [number, number, number];
    }
    this.greenBox = '';
    ctx.log(`  G vector (1/distance) = ${this.gEwald}\n  grid = ${this.mesh.join(' ')}\n  stencil order = ${this.order}`);
  }

  /** Optimal influence function for the current box. */
  private buildGreen(geom: Geometry): void {
    const [nx, ny, nz] = this.mesh;
    const lz = geom.lz * this.slab;
    const key = [geom.lx, geom.ly, lz, geom.xy, geom.xz, geom.yz, nx, ny, nz, this.order, this.gEwald].join(',');
    if (key === this.greenBox) return;
    this.greenBox = key;
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
    const sinc = (x: number) => (x === 0 ? 1 : Math.sin(x) / x);
    const u2 = (n: number, nn: number) => Math.pow(sinc((Math.PI * n) / nn), 2 * P);
    // the numerator's alias images per dimension (measured, see ALIAS_C)
    const edge = [Math.abs(geom.lx), Math.abs(geom.xy) + Math.abs(geom.ly), Math.abs(geom.xz) + Math.abs(geom.yz) + Math.abs(lz)];
    const nb = [0, 1, 2].map((d) => Math.floor(((g * edge[d]) / (Math.PI * N[d])) * ALIAS_C));
    // the denominator sum_m U^2(k_m) factorizes per dimension; by Poisson
    // summation each factor is the finite cosine series of the autocorrelation
    // of the order-P B-spline (the order-2P B-spline at the integers):
    //   sum_m U^2(x + m) = sum_{|j| < P} M_2P(P + j) cos(2 pi j x),  x = n / N
    const den1 = N.map((nn) => {
      const out = new Float64Array(nn);
      for (let i = 0; i < nn; i++) {
        for (let j = 1 - P; j < P; j++) out[i] += bspline(2 * P, P + j) * Math.cos((2 * Math.PI * j * i) / nn);
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
          for (let ax = -nb[0]; ax <= nb[0]; ax++) {
            const nxm = mx + ax * nx;
            const ux = u2(nxm, nx);
            for (let ay = -nb[1]; ay <= nb[1]; ay++) {
              const nym = my + ay * ny;
              const uy = u2(nym, ny);
              for (let az = -nb[2]; az <= nb[2]; az++) {
                const nzm = mz + az * nz;
                kvec(nxm, nym, nzm, km);
                const km2 = km[0] * km[0] + km[1] * km[1] + km[2] * km[2];
                num += ux * uy * u2(nzm, nz) * ((k[0] * km[0] + k[1] * km[1] + k[2] * km[2]) / km2) * 4 * Math.PI * Math.exp(-km2 / (4 * g * g));
              }
            }
          }
          const den = den1[0][ix] * den1[1][iy] * den1[2][iz];
          green[idx] = num / (k2 * den * den);
        }
      }
    }
    this.green = green;
  }

  compute(kc: KSpaceCompute): void {
    const s = kc.s;
    const geom = kc.geom;
    const n = s.n;
    const C = kc.qqrd2e;
    const [nx, ny, nz] = this.mesh;
    const P = this.order;
    const g = this.gEwald;
    this.volume = geom.volume(3) * this.slab;
    const V = this.volume;
    this.buildGreen(geom);
    const ng = nx * ny * nz;
    // charge assignment: grid index and P weights per dimension per atom
    const first = new Int32Array(3 * n);
    const wts = new Float64Array(3 * n * P);
    const lam = [0, 0, 0];
    const N = [nx, ny, nz];
    for (let i = 0; i < n; i++) {
      geom.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
      lam[2] /= this.slab;
      for (let d = 0; d < 3; d++) {
        const u = lam[d] * N[d];
        const i0 = P % 2 === 1 ? Math.round(u) - (P - 1) / 2 : Math.floor(u) - P / 2 + 1;
        first[3 * i + d] = i0;
        for (let k = 0; k < P; k++) wts[(3 * i + d) * P + k] = bspline(P, u - (i0 + k) + P / 2);
      }
    }
    const mod = (a: number, m: number) => ((a % m) + m) % m;
    const rho = new Float64Array(2 * ng);
    for (let i = 0; i < n; i++) {
      const q = s.q[i];
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
    // energy, virial and the three field components in k-space
    const inv = invertUpper(geom.lx, geom.xy, geom.xz, geom.ly, geom.yz, geom.lz * this.slab);
    const ex = new Float64Array(2 * ng), ey = new Float64Array(2 * ng), ez = new Float64Array(2 * ng);
    let e = 0;
    const w = [0, 0, 0, 0, 0, 0];
    for (let iz = 0; iz < nz; iz++) {
      const mz = iz < nz / 2 ? iz : iz - nz;
      for (let iy = 0; iy < ny; iy++) {
        const my = iy < ny / 2 ? iy : iy - ny;
        for (let ix = 0; ix < nx; ix++) {
          const mx = ix < nx / 2 ? ix : ix - nx;
          const idx = (iz * ny + iy) * nx + ix;
          const G = this.green[idx];
          if (G === 0) continue;
          const re = rho[2 * idx], im = rho[2 * idx + 1];
          const r2 = re * re + im * im;
          const kx = 2 * Math.PI * (mx * inv[0][0] + my * inv[1][0] + mz * inv[2][0]);
          const ky = 2 * Math.PI * (mx * inv[0][1] + my * inv[1][1] + mz * inv[2][1]);
          const kz = 2 * Math.PI * (mx * inv[0][2] + my * inv[1][2] + mz * inv[2][2]);
          const k2 = kx * kx + ky * ky + kz * kz;
          e += G * r2;
          const b = 2 * (1 / k2 + 1 / (4 * g * g));
          const t = G * r2;
          w[0] += t * (1 - b * kx * kx); w[1] += t * (1 - b * ky * ky); w[2] += t * (1 - b * kz * kz);
          w[3] += t * (-b * kx * ky); w[4] += t * (-b * kx * kz); w[5] += t * (-b * ky * kz);
          // E_a(k) = -i k_a G rho  ->  (re, im) * (-i k_a G) = (k_a G im, -k_a G re)
          ex[2 * idx] = kx * G * im; ex[2 * idx + 1] = -kx * G * re;
          ey[2 * idx] = ky * G * im; ey[2 * idx + 1] = -ky * G * re;
          ez[2 * idx] = kz * G * im; ez[2 * idx + 1] = -kz * G * re;
        }
      }
    }
    fft3d(ex, nx, ny, nz, 1);
    fft3d(ey, nx, ny, nz, 1);
    fft3d(ez, nx, ny, nz, 1);
    // interpolate the field back and apply F = C q E / V
    for (let i = 0; i < n; i++) {
      const q = s.q[i];
      if (q === 0) continue;
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
      const pre = (C * q) / V;
      kc.f[3 * i] += pre * Ex; kc.f[3 * i + 1] += pre * Ey; kc.f[3 * i + 2] += pre * Ez;
    }
    const pref = C / (2 * V);
    kc.acc.elong += pref * e + this.constantEnergy(C);
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += pref * w[c];
    this.slabCorrection(kc);
  }
}
