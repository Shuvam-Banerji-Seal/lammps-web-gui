import { KSpace, StyleError, type KSpaceCompute, type StyleContext } from '../types';
import type { SimState } from '../../types';
import type { Geometry } from '../../domain';
import { parseNum } from '../util';

/*
 * kspace_style ewald accuracy — docs.lammps.org/kspace_style.html: "The ewald
 * style performs a standard Ewald summation as described in any solid-state
 * physics text." "The specified accuracy determines the relative RMS error in
 * per-atom forces calculated by the long-range solver. It is set as a
 * dimensionless number, relative to the force that two unit point charges
 * (e.g. 2 monovalent ions) exert on each other at a distance of 1 Angstrom."
 * "RMS force errors in real space for ewald and pppm are estimated using
 * equation 18 of (Kolafa)". kspace_modify.html: "gewald ... sets the value of
 * the Ewald or PPPM G-ewald parameter"; "kmax/ewald ... sets the number of
 * kspace vectors in each dimension"; slab: "treating the system as if it were
 * periodic in z, but inserting empty volume between atom slabs and removing
 * dipole inter-slab interactions" (Yeh & Berkowitz); "force value = accuracy
 * (force units)" sets an absolute accuracy.
 *
 * With S(k) = sum_j q_j exp(i k.r_j), A(k) = exp(-k^2/(4 g^2)) / k^2 and
 * C = qqr2e/dielectric (textbook Ewald, e.g. Frenkel & Smit 12.1):
 *   E_k    = C 2 pi / V sum_{k != 0} A(k) |S(k)|^2
 *   E_self = -C g / sqrt(pi) sum_j q_j^2
 *   E_q    = -C pi Q^2 / (2 V g^2)  (neutralizing background when Q = sum q != 0)
 *   F_j    = C 4 pi q_j / V sum_k A(k) k [Re S sin(k.r_j) - Im S cos(k.r_j)]
 *   W_ab   = C 2 pi / V sum_k A(k) |S|^2 (delta_ab - 2 k_a k_b (1/k^2 + 1/(4 g^2)))
 * Wave vectors k = 2 pi H^-T n for integer n (any restricted triclinic box),
 * |n_d| <= kmax_d, and k^2 <= max_d (2 pi kmax_d / L_d)^2 — the spherical
 * cutoff was determined from native LAMMPS output (pressures of the
 * ewald_* oracle cases agree only with it). Default parameters, when gewald / kmax/ewald are not set:
 * g from the Kolafa-Perram real-space estimate 2 C Q2 exp(-g^2 rc^2) /
 * sqrt(N rc V) = accuracy, and kmax_d the smallest m whose Kolafa-Perram rms
 * k-space force error is below the accuracy (kmaxFor). Measured with native
 * LAMMPS (black box, 2026-10): g_ewald and kmax/ewald matched both rules on 8
 * automatic cases (accuracy 1e-3..1e-6, cutoffs 6..12, a 12x18x24 box).
 */

/** Distance of 1 Angstrom and the proton charge in each unit style (units.html). */
export const ANGSTROM: Record<string, number> = { lj: 1, real: 1, metal: 1, si: 1e-10, cgs: 1e-8, electron: 1.88972612, micro: 1e-4, nano: 0.1 };
export const QELECTRON: Record<string, number> = { lj: 1, real: 1, metal: 1, si: 1.6021765e-19, cgs: 4.8032044e-10, electron: 1, micro: 1.6021765e-7, nano: 1 };

export abstract class KSpaceBase extends KSpace {
  /** kspace_modify force: absolute accuracy (force units), or -1. */
  forceAbs = -1;
  gewaldUser = 0;
  slab = 1;
  qsum = 0;
  qsqsum = 0;
  protected natoms = 0;
  protected volume = 0;
  protected cutCoul = 0;

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError(`usage: kspace_style ${this.name} accuracy`);
    this.accuracy = parseNum(args[0], 'accuracy');
    if (!(this.accuracy > 0)) throw new StyleError('kspace accuracy must be > 0');
  }

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'gewald': this.gewaldUser = parseNum(values[0], 'gewald'); return 1;
      case 'force': this.forceAbs = parseNum(values[0], 'force'); return 1;
      case 'slab':
        if (values[0] === 'nozforce') throw new StyleError('kspace_modify slab nozforce is not supported by the browser engine');
        this.slab = parseNum(values[0], 'slab volfactor');
        if (this.slab < 1) throw new StyleError('kspace_modify slab volfactor must be >= 1.0');
        return 1;
      case 'pressure/scalar': case 'collective': case 'overlap': case 'fftbench': case 'cutoff/adjust': return 1;
      case 'minorder': return 1;
      case 'tol': return 1;
      default: return 0;
    }
  }

  /** Absolute force accuracy: user 'force', or accuracy times the force of two unit charges 1 Angstrom apart. */
  protected absAccuracy(s: SimState, qqrd2e: number): number {
    if (this.forceAbs > 0) return this.forceAbs;
    const a = ANGSTROM[s.units.style], q = QELECTRON[s.units.style];
    return (this.accuracy * qqrd2e * q * q) / (a * a);
  }

  /** Charges, neutrality and the box checks shared by Ewald and PPPM. */
  protected setupCharges(s: SimState, geom: Geometry, ctx: StyleContext): void {
    let qs = 0, q2 = 0;
    for (let i = 0; i < s.n; i++) { qs += s.q[i]; q2 += s.q[i] * s.q[i]; }
    this.qsum = qs;
    this.qsqsum = q2;
    this.natoms = s.n;
    if (s.atomStyle !== 'charge' && s.atomStyle !== 'full') throw new StyleError(`kspace_style ${this.name} needs charges (atom_style charge or full)`);
    if (q2 === 0) ctx.log(`WARNING: kspace_style ${this.name}: no atoms are charged`);
    if (Math.abs(qs) > 1e-5) ctx.log(`WARNING: kspace_style ${this.name}: the system is not charge neutral (net charge ${qs}); a uniform neutralizing background is assumed`);
    const p = s.box.periodic;
    if (this.slab > 1) {
      if (!p[0] || !p[1] || p[2]) throw new StyleError('kspace_modify slab needs boundary p p f (or p p s/m)');
      if (s.box.triclinic) throw new StyleError('kspace_modify slab is not supported with a triclinic box');
    } else if (!p[0] || !p[1] || !p[2]) {
      throw new StyleError(`kspace_style ${this.name} needs a fully periodic box (or kspace_modify slab for p p f)`);
    }
    this.volume = geom.volume(3) * this.slab;
  }

  /** g from the real-space estimate (Kolafa-Perram eq. 18) unless set by the user. */
  protected chooseGEwald(s: SimState, qqrd2e: number): number {
    if (this.gewaldUser > 0) return this.gewaldUser;
    const acc = this.absAccuracy(s, qqrd2e);
    const rc = this.cutCoul;
    const n = Math.max(1, this.natoms);
    const pre = (2 * qqrd2e * this.qsqsum) / Math.sqrt(n * rc * this.volume);
    if (!(pre > acc)) return 1 / rc;   // any splitting meets the target; a gentle default
    // 2 C Q2 exp(-g^2 rc^2) / sqrt(N rc V) = acc
    return Math.sqrt(Math.log(pre / acc)) / rc;
  }

  /** Self and background energies (added to elong). */
  protected constantEnergy(qqrd2e: number): number {
    const g = this.gEwald;
    let e = -(g / Math.sqrt(Math.PI)) * this.qsqsum;
    e += -(Math.PI * this.qsum * this.qsum) / (2 * this.volume * g * g);
    return e * qqrd2e;
  }

  /**
   * Slab correction (Yeh & Berkowitz; extended to non-neutral systems): a
   * dipole term 2 pi / V M_z^2 with its force -4 pi q_i M_z / V.
   */
  protected slabCorrection(kc: KSpaceCompute): void {
    if (this.slab <= 1) return;
    const s = kc.s;
    let mz = 0, qz2 = 0;
    for (let i = 0; i < s.n; i++) { mz += s.q[i] * s.x[3 * i + 2]; qz2 += s.q[i] * s.x[3 * i + 2] * s.x[3 * i + 2]; }
    const V = this.volume;
    const zprd = (s.box.hi[2] - s.box.lo[2]) * this.slab;
    const C = kc.qqrd2e;
    const e = (2 * Math.PI / V) * (mz * mz - this.qsum * qz2 - (this.qsum * this.qsum * zprd * zprd) / 12);
    kc.acc.elong += C * e;
    for (let i = 0; i < s.n; i++) {
      const fz = -C * (4 * Math.PI / V) * s.q[i] * (mz - this.qsum * s.x[3 * i + 2]);
      kc.f[3 * i + 2] += fz;
    }
  }
}

export class KSpaceEwald extends KSpaceBase {
  readonly name = 'ewald';
  kmaxUser: [number, number, number] | null = null;
  private kvecs = new Float64Array(0);   // kx ky kz A(k) per vector (half space)
  private nvec = new Int32Array(0);      // n_x n_y n_z per vector
  private kmax: [number, number, number] = [0, 0, 0];

  modify(key: string, values: string[]): number {
    if (key === 'kmax/ewald') {
      const k = [0, 1, 2].map((d) => parseNum(values[d], 'kmax/ewald'));
      if (k.some((v) => !Number.isInteger(v) || v < 0)) throw new StyleError('kmax/ewald values must be integers >= 0');
      this.kmaxUser = k.every((v) => v === 0) ? null : (k as [number, number, number]);
      return 3;
    }
    if (key === 'mesh' || key === 'order' || key === 'diff') throw new StyleError(`kspace_modify ${key} applies to pppm, not ewald`);
    return super.modify(key, values);
  }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    this.cutCoul = cutCoul;
    this.setupCharges(s, geom, ctx);
    this.gEwald = this.chooseGEwald(s, qqrd2e);
    const g = this.gEwald;
    if (this.kmaxUser) this.kmax = [...this.kmaxUser];
    else {
      // smallest kmax per dimension whose Kolafa-Perram rms force error is below the absolute accuracy
      const acc = this.absAccuracy(s, qqrd2e);
      const L = [geom.lx, geom.ly, geom.lz * this.slab];
      this.kmax = [0, 1, 2].map((d) => this.kmaxFor(acc, g, L[d], qqrd2e)) as [number, number, number];
    }
    ctx.log(`  G vector (1/distance) = ${g}\n  kmax/ewald = ${this.kmax.join(' ')}`);
    this.buildVectors(geom);
  }

  /**
   * Smallest m >= 1 with 2 C Q2 g / L sqrt(1/(pi m N)) exp(-(pi m / (g L))^2) <= acc
   * (Kolafa & Perram 1992, the real-space-equivalent rms force error of the k-space
   * sum truncated at m). kspace_style.html: "RMS force errors in real space for
   * ewald and pppm are estimated using equation 18 of (Kolafa)". Measured with
   * native LAMMPS (black box, 2026-10, real units, 8 cases incl. a 12x18x24 box):
   * the printed kmax/ewald matched this rule in every case.
   */
  private kmaxFor(acc: number, g: number, prd: number, qqrd2e: number): number {
    const q2 = qqrd2e * this.qsqsum;
    const N = Math.max(1, this.natoms);
    for (let m = 1; m < 1000; m++) {
      const rms = (2 * q2 * g / prd) * Math.sqrt(1 / (Math.PI * m * N)) * Math.exp(-(Math.PI * Math.PI * m * m) / (g * g * prd * prd));
      if (rms <= acc) return m;
    }
    return 999;
  }

  /** Half-space wave vectors for the current box. */
  private buildVectors(geom: Geometry): void {
    const [kx, ky, kz] = this.kmax;
    const g = this.gEwald;
    // reciprocal vectors: rows of 2 pi H^-1 (H columns are the edge vectors), slab-extended in z
    const lz = geom.lz * this.slab;
    const inv = invertUpper(geom.lx, geom.xy, geom.xz, geom.ly, geom.yz, lz);
    const list: number[] = [];
    const ns: number[] = [];
    // spherical cutoff at the largest axis wave vector (see the header)
    const L = [geom.lx, geom.ly, lz];
    let gsqmx = 0;
    for (let d = 0; d < 3; d++) gsqmx = Math.max(gsqmx, (2 * Math.PI * this.kmax[d] / L[d]) ** 2);
    for (let nx = 0; nx <= kx; nx++) {
      for (let ny = -ky; ny <= ky; ny++) {
        for (let nz = -kz; nz <= kz; nz++) {
          if (nx === 0 && (ny < 0 || (ny === 0 && nz <= 0))) continue;
          // k = 2 pi (n . H^-1)  (H^-1 rows), i.e. k_c = 2 pi sum_d n_d inv[d][c]
          const kxv = 2 * Math.PI * (nx * inv[0][0] + ny * inv[1][0] + nz * inv[2][0]);
          const kyv = 2 * Math.PI * (nx * inv[0][1] + ny * inv[1][1] + nz * inv[2][1]);
          const kzv = 2 * Math.PI * (nx * inv[0][2] + ny * inv[1][2] + nz * inv[2][2]);
          const k2 = kxv * kxv + kyv * kyv + kzv * kzv;
          // vectors on the sphere (n^2 = kmax^2 in a cubic box, e.g. (3,4,0) for kmax 5) are
          // included, as native LAMMPS does (oracle case ewald_sphere_boundary); the slack only
          // absorbs the last-bit rounding of k2 against gsqmx
          if (k2 > gsqmx * (1 + 1e-12)) continue;
          list.push(kxv, kyv, kzv, Math.exp(-k2 / (4 * g * g)) / k2);
          ns.push(nx, ny, nz);
        }
      }
    }
    this.kvecs = Float64Array.from(list);
    this.nvec = Int32Array.from(ns);
  }

  compute(kc: KSpaceCompute): void {
    const s = kc.s;
    const g = kc.geom;
    const n = s.n;
    const C = kc.qqrd2e;
    const g2 = this.gEwald;
    // box may have changed (barostat): rebuild wave vectors and volume
    this.volume = g.volume(3) * this.slab;
    this.buildVectors(g);
    const V = this.volume;
    const [kx, ky, kz] = this.kmax;
    // per-atom phase factors exp(2 pi i n_d u_d) for 0 <= n_d <= kmax_d via recurrence
    const lam = [0, 0, 0];
    const cs = [new Float64Array(n * (kx + 1)), new Float64Array(n * (ky + 1)), new Float64Array(n * (kz + 1))];
    const sn = [new Float64Array(n * (kx + 1)), new Float64Array(n * (ky + 1)), new Float64Array(n * (kz + 1))];
    const km = [kx, ky, kz];
    for (let i = 0; i < n; i++) {
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
      lam[2] /= this.slab;
      for (let d = 0; d < 3; d++) {
        const K = km[d] + 1;
        const c1 = Math.cos(2 * Math.PI * lam[d]), s1 = Math.sin(2 * Math.PI * lam[d]);
        cs[d][i * K] = 1; sn[d][i * K] = 0;
        for (let m = 1; m <= km[d]; m++) {
          const pc = cs[d][i * K + m - 1], ps = sn[d][i * K + m - 1];
          cs[d][i * K + m] = pc * c1 - ps * s1;
          sn[d][i * K + m] = ps * c1 + pc * s1;
        }
      }
    }
    const nk = this.nvec.length / 3;
    const cosk = new Float64Array(n), sink = new Float64Array(n);
    let e = 0;
    const w = [0, 0, 0, 0, 0, 0];
    const q = s.q;
    for (let v = 0; v < nk; v++) {
      const nx = this.nvec[3 * v], ny = this.nvec[3 * v + 1], nz = this.nvec[3 * v + 2];
      const kxv = this.kvecs[4 * v], kyv = this.kvecs[4 * v + 1], kzv = this.kvecs[4 * v + 2], A = this.kvecs[4 * v + 3];
      let sre = 0, sim = 0;
      for (let i = 0; i < n; i++) {
        // exp(i 2 pi (nx ux + ny uy + nz uz)) with negative n via conjugation
        const cxr = cs[0][i * (kx + 1) + nx], cxi = sn[0][i * (kx + 1) + nx];
        const ay = Math.abs(ny), az = Math.abs(nz);
        const cyr = cs[1][i * (ky + 1) + ay], cyi = (ny < 0 ? -1 : 1) * sn[1][i * (ky + 1) + ay];
        const czr = cs[2][i * (kz + 1) + az], czi = (nz < 0 ? -1 : 1) * sn[2][i * (kz + 1) + az];
        const xyr = cxr * cyr - cxi * cyi, xyi = cxr * cyi + cxi * cyr;
        const cr = xyr * czr - xyi * czi, ci = xyr * czi + xyi * czr;
        cosk[i] = cr; sink[i] = ci;
        sre += q[i] * cr; sim += q[i] * ci;
      }
      const s2 = sre * sre + sim * sim;
      // half space: each vector stands for +k and -k
      e += 2 * A * s2;
      const k2 = kxv * kxv + kyv * kyv + kzv * kzv;
      const vfac = 2 * A * s2;
      const b = 2 * (1 / k2 + 1 / (4 * g2 * g2));
      w[0] += vfac * (1 - b * kxv * kxv); w[1] += vfac * (1 - b * kyv * kyv); w[2] += vfac * (1 - b * kzv * kzv);
      w[3] += vfac * (-b * kxv * kyv); w[4] += vfac * (-b * kxv * kzv); w[5] += vfac * (-b * kyv * kzv);
      const pre = C * (8 * Math.PI / V) * A;
      for (let i = 0; i < n; i++) {
        if (q[i] === 0) continue;
        const t = pre * q[i] * (sre * sink[i] - sim * cosk[i]);
        kc.f[3 * i] += t * kxv; kc.f[3 * i + 1] += t * kyv; kc.f[3 * i + 2] += t * kzv;
      }
    }
    const pref = C * 2 * Math.PI / V;
    kc.acc.elong += pref * e + this.constantEnergy(C);
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += pref * w[c];
    // the neutralizing-background energy has no virial term here: native LAMMPS pressures
    // of a non-neutral system (oracle case ewald_tri_charged) leave it out
    this.slabCorrection(kc);
  }
}

/** Inverse of the upper-triangular box matrix [[lx, xy, xz], [0, ly, yz], [0, 0, lz]]. */
export const invertUpper = (lx: number, xy: number, xz: number, ly: number, yz: number, lz: number): number[][] => [
  [1 / lx, -xy / (lx * ly), (xy * yz - ly * xz) / (lx * ly * lz)],
  [0, 1 / ly, -yz / (ly * lz)],
  [0, 0, 1 / lz],
];
