import { KSpace, StyleError, type KSpaceCompute, type StyleContext } from '../types';
import type { SimState } from '../../types';
import type { Geometry } from '../../domain';
import { parseNum } from '../util';
import { modifyNoEffect } from './ewald';
import { libm } from '../../libm';

/*
 * kspace_style ewald/dipole — docs.lammps.org/kspace_style.html: "The *ewald/dipole* style adds
 * long-range standard Ewald summations for dipole-dipole interactions, see (Toukmaji)." and
 * "RMS force errors for dipoles when using *ewald/disp* or *ewald/dipole* are estimated using
 * equations 33 and 46 of (Wang3)." The Wang3 reference is Wang and Holm, Estimate of the Cutoff
 * Errors in the Ewald Summation for Dipolar Systems (arXiv cond-mat/0107064). Its equation 33 is the
 * rms real-space force error, Delta F(r) = M^2 (L^3 alpha^4 rc^9 N)^(-1/2) sqrt(13/6 Cc^2 + 2/15 Dc^2
 * - 13/15 Cc Dc) exp(-alpha^2 rc^2) with Cc = 4 x^4 + 6 x^2 + 3, Dc = 8 x^6 + 20 x^4 + 30 x^2 + 15,
 * x = alpha rc, and M^2 = sum_j mu_j^2; its equation 46 is the rms k-space error,
 * Delta F(k) = 8 pi M^2 L^-3 alpha (2 pi kc^3 / (15 N))^(1/2) exp(-(pi kc / (alpha L))^2).
 *
 * Measured with native LAMMPS (black box, 2026-10, real, lj and metal units, 2 to 8 dipoles):
 * g_ewald is the root of Delta F(r) = accuracy, with no unit factor (the same G in lj, metal and
 * real units for one accuracy). With the printed G matched to 8 digits in 14 cases.
 * kmax/ewald per dimension d is the smallest integer m with Delta F(k) <= accuracy, where
 * L_d is the box length of dimension d and the volume V is the full box volume (the k-space
 * error uses L^-3 = 1/V); this matched 14 cubic cases and the non-cubic boxes 10x12x14 and 8x10x20.
 * The printed estimated absolute RMS force accuracy is 0 for a dipole-only system (Ewald charges
 * are zero), which is consistent with the dipole estimate above being the one used.
 *
 * Energy (Wang and Holm eqs 2 to 5; tin-foil, i.e. no surface term; prefactor C = qqr2e / dielectric):
 *   E_k    = C (4 pi / V) sum_{K != 0, half space} exp(-K^2/(4 g^2)) / K^2 |M(K)|^2
 *   M(K)   = sum_j (mu_j . K) exp(i K . r_j),  K = 2 pi n / L (integer n, half space)
 *   E_self = -C (2 g^3 / (3 sqrt(pi))) sum_j mu_j^2
 * Measured with native LAMMPS (black box): elong equals E_k + E_self to 15 digits at a fixed G
 * (kspace_modify gewald 0.45, kmax/ewald 5 5 5, four dipoles in a 12 box). The self term belongs to elong.
 * Forces and torques (derived from the energy; checked by finite differences in the unit test):
 *   F_i   = C (8 pi / V) sum_half G(K) K (mu_i . K) Im(M* exp(i K . r_i))
 *   E_i   = -C (8 pi / V) sum_half G(K) K Re(M* exp(i K . r_i)),  tau_i = mu_i x E_i
 * Virial (mu held fixed under a box deformation; the box-scaling finite difference agrees):
 *   W_ab  = C (4 pi / V) sum_half G(K) [ |M|^2 (delta_ab - b K_a K_b) + Re((K_b P_a + K_a P_b) M*) ]
 *   with b = 2 (1/K^2 + 1/(4 g^2)) and P_a = sum_j mu_ja exp(i K . r_j).
 * The k-vectors are |n_d| <= kmax_d and K^2 <= max_d (2 pi kmax_d / L_d)^2, the sphere used by
 * the ewald style; the boundary vectors are included (measured: excluding them changes elong by 3e-4).
 *
 * Refusals (with native behaviour where measured): point charges. Measured with native LAMMPS (black
 * box): ewald/dipole refuses any system with charges (a message says charges cannot be used with
 * EwaldDipole), while a zero charge on every atom is accepted. The engine refuses the same case.
 * Triclinic and non-periodic boxes, kspace_modify slab, force, mesh and order are refused (not measured).
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** Real-space dipole-dipole error of equation 33 of Wang and Holm (the automatic G-ewald target). */
export const dipoleRealForceError = (g: number, rc: number, V: number, N: number, musq: number): number => {
  const x = g * rc, x2 = x * x, x4 = x2 * x2, x6 = x4 * x2;
  const C = 4 * x4 + 6 * x2 + 3;
  const D = 8 * x6 + 20 * x4 + 30 * x2 + 15;
  const inner = (13 / 6) * C * C + (2 / 15) * D * D - (13 / 15) * C * D;
  return musq / Math.sqrt(V * g ** 4 * rc ** 9 * N) * Math.sqrt(inner) * libm.exp(-x2);
};

/** Reciprocal-space rms force error of equation 46 for a cutoff kc along one box length L_d. */
export const dipoleKSpaceForceError = (kc: number, g: number, Ld: number, V: number, N: number, musq: number): number =>
  8 * Math.PI * musq / V * g * Math.sqrt((2 * Math.PI * kc ** 3) / (15 * N)) * libm.exp(-(((Math.PI * kc) / (g * Ld)) ** 2));

export class KSpaceEwaldDipole extends KSpace {
  readonly name = 'ewald/dipole';
  /** kspace_modify gewald: a user value, or 0 for the automatic choice. */
  gewaldUser = 0;
  /** kspace_modify kmax/ewald: a user value per dimension, or null for the automatic choice. */
  kmaxUser: [number, number, number] | null = null;
  private kmax: [number, number, number] = [0, 0, 0];
  private natoms = 0;
  private musq = 0;
  private volume = 0;
  /** Half-space wave vectors: K (3) and the integer n (3) per vector. */
  private kvec = new Float64Array(0);
  private nvec = new Int32Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('Illegal kspace_style ewald/dipole command');
    this.accuracy = parseNum(args[0], 'accuracy');
    if (!(this.accuracy > 0)) throw new StyleError('KSpace accuracy must be > 0');
  }

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'gewald': {
        this.gewaldUser = parseNum(values[0], 'gewald');
        return 1;
      }
      case 'kmax/ewald': {
        const k = [0, 1, 2].map((d) => parseNum(values[d], 'kmax/ewald'));
        if (k.some((v) => !Number.isInteger(v) || v < 0)) throw new StyleError('kmax/ewald values must be integers >= 0');
        this.kmaxUser = k.every((v) => v === 0) ? null : (k as [number, number, number]);
        return 3;
      }
      case 'force': case 'slab': case 'mesh': case 'order': case 'diff':
        throw new StyleError(`kspace_modify ${key} is not supported by kspace_style ewald/dipole`);
      default:
        return modifyNoEffect(key, values);
    }
  }

  init(s: SimState, geom: Geometry, cutCoul: number, _qqrd2e: number, ctx: StyleContext): void {
    if (!s.mu) throw new StyleError('kspace_style ewald/dipole requires atom_style dipole (or hybrid with dipole)');
    // Measured with native LAMMPS (black box): ewald/dipole refuses charged systems (see the header).
    if (s.q) for (let i = 0; i < s.n; i++) {
      if (s.q[i] !== 0) throw new StyleError('kspace_style ewald/dipole cannot be used with point charges (dipoles only; use ewald or ewald/disp for charges)');
    }
    const p = s.box.periodic;
    if (!p[0] || !p[1] || !p[2]) throw new StyleError('kspace_style ewald/dipole needs a fully periodic box');
    if (s.box.triclinic || geom.xy !== 0 || geom.xz !== 0 || geom.yz !== 0) {
      throw new StyleError('kspace_style ewald/dipole is implemented for orthogonal boxes only');
    }
    const mu = s.mu;
    let musq = 0;
    for (let i = 0; i < s.n; i++) musq += mu[4 * i] ** 2 + mu[4 * i + 1] ** 2 + mu[4 * i + 2] ** 2;
    if (!(musq > 0)) throw new StyleError('kspace_style ewald/dipole needs at least one dipole with a nonzero moment');
    this.musq = musq;
    this.natoms = s.n;
    this.volume = geom.volume(3);
    const L = [geom.lx, geom.ly, geom.lz];
    this.gEwald = this.gewaldUser > 0 ? this.gewaldUser : this.chooseGEwald(musq, cutCoul);
    const g = this.gEwald;
    if (this.kmaxUser) this.kmax = [...this.kmaxUser];
    else this.kmax = [0, 1, 2].map((d) => this.kmaxFor(g, L[d])) as [number, number, number];
    ctx.log(`  G vector (1/distance) = ${g}\n  kmax/ewald = ${this.kmax.join(' ')}`);
    this.buildVectors(L);
  }

  /**
   * Root of equation 33 in g, with the real-space cutoff rc: Delta F(r)(g) = accuracy. The
   * function decreases monotonically in g, so bisection on ln g is used.
   */
  private chooseGEwald(musq: number, rc: number): number {
    const target = libm.log(this.accuracy);
    const f = (g: number) => libm.log(dipoleRealForceError(g, rc, this.volume, this.natoms, musq)) - target;
    let lo = 1e-6 / rc, hi = 100 / rc;
    if (!(f(lo) > 0 && f(hi) < 0)) throw new StyleError('kspace_style ewald/dipole cannot choose G-ewald for this accuracy and cutoff');
    for (let it = 0; it < 200; it++) {
      const mid = Math.sqrt(lo * hi);
      if (f(mid) > 0) lo = mid; else hi = mid;
      if (hi / lo - 1 < 1e-15) break;
    }
    return Math.sqrt(lo * hi);
  }

  /** Smallest m >= 1 with the equation 46 error at or below the accuracy, for box length Ld. */
  private kmaxFor(g: number, Ld: number): number {
    for (let m = 1; m < 1000; m++) {
      if (dipoleKSpaceForceError(m, g, Ld, this.volume, this.natoms, this.musq) <= this.accuracy) return m;
    }
    return 999;
  }

  private buildVectors(L: number[]): void {
    const [kx, ky, kz] = this.kmax;
    const km = [kx, ky, kz];
    let gsqmx = 0;
    for (let d = 0; d < 3; d++) gsqmx = Math.max(gsqmx, (2 * Math.PI * km[d] / L[d]) ** 2);
    const list: number[] = [], ns: number[] = [];
    for (let nx = 0; nx <= kx; nx++) {
      for (let ny = -ky; ny <= ky; ny++) {
        for (let nz = -kz; nz <= kz; nz++) {
          if (nx === 0 && (ny < 0 || (ny === 0 && nz <= 0))) continue;
          const Kx = 2 * Math.PI * nx / L[0], Ky = 2 * Math.PI * ny / L[1], Kz = 2 * Math.PI * nz / L[2];
          const K2 = Kx * Kx + Ky * Ky + Kz * Kz;
          if (K2 > gsqmx * (1 + 1e-12)) continue;
          list.push(Kx, Ky, Kz);
          ns.push(nx, ny, nz);
        }
      }
    }
    this.kvec = Float64Array.from(list);
    this.nvec = Int32Array.from(ns);
  }

  compute(kc: KSpaceCompute): void {
    const s = kc.s, g = this.gEwald;
    // the box may change under a barostat: refresh the box-dependent data
    const geom = kc.geom;
    this.volume = geom.volume(3);
    this.buildVectors([geom.lx, geom.ly, geom.lz]);
    const n = s.n, mu = s.mu!, x = s.x, f = kc.f, C = kc.qqrd2e, V = this.volume;
    const torque = kc.torque ?? null;
    const nk = this.nvec.length / 3;
    const cosv = new Float64Array(n), sinv = new Float64Array(n);
    const Ef = new Float64Array(3 * n);
    let esum = 0;
    const W = [0, 0, 0, 0, 0, 0];
    const g2 = 4 * g * g;
    for (let v = 0; v < nk; v++) {
      const Kx = this.kvec[3 * v], Ky = this.kvec[3 * v + 1], Kz = this.kvec[3 * v + 2];
      const K2 = Kx * Kx + Ky * Ky + Kz * Kz;
      const Gk = libm.exp(-K2 / g2) / K2;
      let Mr = 0, Mi = 0;
      const P = [0, 0, 0], Pi = [0, 0, 0];
      for (let i = 0; i < n; i++) {
        const ph = Kx * x[3 * i] + Ky * x[3 * i + 1] + Kz * x[3 * i + 2];
        const c = libm.cos(ph), sn = libm.sin(ph);
        cosv[i] = c; sinv[i] = sn;
        const mk = mu[4 * i] * Kx + mu[4 * i + 1] * Ky + mu[4 * i + 2] * Kz;
        Mr += mk * c; Mi += mk * sn;
        for (let a = 0; a < 3; a++) { P[a] += mu[4 * i + a] * c; Pi[a] += mu[4 * i + a] * sn; }
      }
      const M2 = Mr * Mr + Mi * Mi;
      esum += Gk * M2;
      // virial (half space, times 4 pi / V below)
      const b = 2 * (1 / K2 + 1 / (4 * g * g));
      const Kv = [Kx, Ky, Kz];
      const pairs: [number, number, number][] = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [0, 1, 3], [0, 2, 4], [1, 2, 5]];
      for (const [a, c, idx] of pairs) {
        const dl = a === c ? 1 : 0;
        // Pc_a = P_a + i Pi_a; Re((K_c Pc_a + K_a Pc_c) conj(M))  with conj(M) = Mr - i Mi
        const reA = Kv[c] * P[a] + Kv[a] * P[c];
        const imA = Kv[c] * Pi[a] + Kv[a] * Pi[c];
        const reProd = reA * Mr + imA * Mi;
        W[idx] += Gk * (M2 * (dl - b * Kv[a] * Kv[c]) + reProd);
      }
      // forces and the field on each atom
      const cf = C * (8 * Math.PI / V) * Gk;
      for (let i = 0; i < n; i++) {
        const c = cosv[i], sn = sinv[i];
        const mk = mu[4 * i] * Kx + mu[4 * i + 1] * Ky + mu[4 * i + 2] * Kz;
        const im = Mr * sn - Mi * c;
        const re = Mr * c + Mi * sn;
        const t = cf * mk * im;
        f[3 * i] += t * Kx; f[3 * i + 1] += t * Ky; f[3 * i + 2] += t * Kz;
        Ef[3 * i] -= cf * re * Kx; Ef[3 * i + 1] -= cf * re * Ky; Ef[3 * i + 2] -= cf * re * Kz;
      }
    }
    if (torque) {
      for (let i = 0; i < n; i++) {
        const mx = mu[4 * i], my = mu[4 * i + 1], mz = mu[4 * i + 2];
        const ex = Ef[3 * i], ey = Ef[3 * i + 1], ez = Ef[3 * i + 2];
        torque[3 * i] += my * ez - mz * ey;
        torque[3 * i + 1] += mz * ex - mx * ez;
        torque[3 * i + 2] += mx * ey - my * ex;
      }
    }
    // the dipoles may rotate (fix nve/sphere update dipole): the self term uses the current moments
    let musq = 0;
    for (let i = 0; i < n; i++) musq += mu[4 * i] ** 2 + mu[4 * i + 1] ** 2 + mu[4 * i + 2] ** 2;
    this.musq = musq;
    const pref = C * 4 * Math.PI / V;
    const self = -C * (TWO_OVER_SQRTPI * g * g * g / 3) * musq;
    kc.acc.elong += pref * esum + self;
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += pref * W[c];
  }
}
