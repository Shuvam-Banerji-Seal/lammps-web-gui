import { hasChargeStyle } from '../../atoms';
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

/** kspace_modify yes/no keywords: a value other than yes or no is refused (native message, measured). */
const yesNoValue = (w: string | undefined, key: string): void => {
  if (w !== 'yes' && w !== 'no') throw new StyleError(`Expected boolean parameter instead of '${w ?? ''}' for kspace_modify ${key}`);
};

/** kspace_modify minorder value: an integer of at least 2 (native message, measured: Kspace_modify minorder must be at least 2). */
export const minorderValue = (w: string | undefined): number => {
  if (w === undefined || !/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' for kspace_modify minorder`);
  const m = Number(w);
  if (m < 2) throw new StyleError('Kspace_modify minorder must be at least 2');
  return m;
};

/**
 * kspace_modify keywords that native LAMMPS accepts for the kspace styles of this engine and whose output
 * this engine reproduces with no effect (each one measured against native, see the comments below).
 * Returns the number of values used, or 0 when the keyword is not one of them. A keyword that changes
 * results, or that native rejects, is refused here with an EngineError naming it.
 */
export const modifyNoEffect = (key: string, values: string[]): number => {
  switch (key) {
    // kspace_modify.html: "to remap data for 3d-FFT operations instead of the default" (nonblocking) and
    // "instead of the default point-to-point communication" (collective). They choose the MPI communication
    // pattern of the 3d FFT remap, which this single-process engine does not use. Native LAMMPS accepts both
    // with ewald and pppm (measured) and its thermo output is identical with and without them (measured).
    case 'collective': yesNoValue(values[0], 'collective'); return 1;
    case 'nonblocking': yesNoValue(values[0], 'nonblocking'); return 1;
    // kspace_modify.html (4Jul2026): "It controls whether data remapped to the" same MPI rank is handled by a
    // direct copy; this engine has no MPI remap, so the value is checked and has no effect.
    case 'collective/self/copy':
      if (values[0] !== 'yes' && values[0] !== 'no' && values[0] !== 'onerank') {
        throw new StyleError(`Unknown kspace_modify collective/self/copy flag ${values[0] ?? ''}`);
      }
      return 1;
    // kspace_modify.html: "The *fftbench* keyword applies only to PPPM." It adds a timing benchmark to the log
    // and nothing to the forces. Measured with native LAMMPS (black box): with fftbench yes the thermo rows of
    // ewald, pppm, ewald/disp and ewald/dipole are identical to the run without it (pppm/tip4p: identical up to
    // the last bit, which two runs of the same unchanged input do not reproduce either).
    case 'fftbench': yesNoValue(values[0], 'fftbench'); return 1;
    // kspace_modify.html: "If the *overlap* keyword is set to *yes* then this communication is allowed to extend
    // beyond nearest-neighbor processors". With one process the ghost communication is always to nearest
    // neighbours, so overlap yes and no give the same result. Measured with native LAMMPS (black box): overlap yes
    // and overlap no leave the thermo rows of ewald and pppm unchanged.
    case 'overlap': yesNoValue(values[0], 'overlap'); return 1;
    // kspace_modify.html: "The *minorder* keyword is not currently supported in MSM." and "The minimum allowed
    // value for PPPM is 2". The order is reduced only when the ghost communication overlaps beyond the nearest
    // neighbours, which a single process never does. The pppm solver checks minorder <= order at init (measured:
    // order 5 with minorder 6 stops with PPPM order < minimum allowed order). Native ewald accepts minorder 2 to 8
    // with no effect (measured).
    case 'minorder': minorderValue(values[0]); return 1;
    // kspace_modify.html: "The *cutoff/adjust* keyword applies only to MSM." Measured with native LAMMPS (black
    // box): cutoff/adjust yes leaves the thermo rows (energies, pressure tensor) of ewald, pppm, ewald/disp,
    // ewald/dipole and pppm/tip4p unchanged.
    case 'cutoff/adjust': yesNoValue(values[0], 'cutoff/adjust'); return 1;
    // kspace_modify.html: "The *pressure/scalar* keyword applies only to MSM." Measured with native LAMMPS (black
    // box): pressure/scalar yes with only press in thermo_style gives the same numbers, but with the pxx..pyz
    // components in thermo_style it stops with (measured): Must use kspace_modify pressure/scalar no for tensor
    // components. The engine does not reproduce that path, so yes is refused rather than half done.
    case 'pressure/scalar':
      yesNoValue(values[0], 'pressure/scalar');
      if (values[0] === 'yes') {
        throw new StyleError('kspace_modify pressure/scalar yes is not supported by the browser engine (it applies only to MSM, which the engine does not have)');
      }
      return 1;
    // tol is not a kspace_modify keyword: the keyword list in kspace_modify.html does not contain it, and native
    // LAMMPS stops with an illegal kspace_modify command error for ewald, pppm, ewald/disp, pppm/tip4p and
    // ewald/dipole (measured).
    case 'tol':
      throw new StyleError('kspace_modify tol is not a kspace_modify keyword (native LAMMPS rejects it as an illegal kspace_modify command)');
    default: return 0;
  }
};

export abstract class KSpaceBase extends KSpace {
  /** kspace_modify force: absolute accuracy (force units), or -1. */
  forceAbs = -1;
  gewaldUser = 0;
  slab = 1;
  /** kspace_modify slab auto: the volfactor follows from the tolerance and the box (see autoSlabVolfactor). */
  slabAuto = false;
  qsum = 0;
  qsqsum = 0;
  protected natoms = 0;
  protected volume = 0;
  protected cutCoul = 0;
  /** kspace_modify minorder (PPPM: the smallest order the grid stencil may be reduced to; 2 by default). */
  minorder = 2;

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
        if (values[0] === 'auto') { this.slabAuto = true; this.slab = 1; return 1; }
        this.slabAuto = false;
        this.slab = parseNum(values[0], 'slab volfactor');
        if (this.slab < 1) throw new StyleError('kspace_modify slab volfactor must be >= 1.0');
        // native LAMMPS refuses a volfactor of exactly 1.0 with Bad kspace_modify slab parameter (measured)
        if (this.slab === 1) throw new StyleError('Bad kspace_modify slab parameter (volfactor 1.0 is refused; omit slab for no slab correction)');
        return 1;
      case 'minorder': this.minorder = minorderValue(values[0]); return 1;
      default: return modifyNoEffect(key, values);
    }
  }

  /** Absolute force accuracy: user 'force', or accuracy times the force of two unit charges 1 Angstrom apart. */
  protected absAccuracy(s: SimState, _qqrd2e: number): number {
    if (this.forceAbs > 0) return this.forceAbs;
    // Measured with native LAMMPS (black box): the dielectric command does not change this reference force. With
    // dielectric 0.4255, 0.5, 1 and 2 the G-ewald estimate gives (g rc)^2 = L0 - ln(dielectric) (L0 from dielectric 1),
    // i.e. only the pre-factor 2 qqrd2e Q2 carries 1/dielectric; the reference uses qqr2e.
    const a = ANGSTROM[s.units.style], q = QELECTRON[s.units.style];
    return (this.accuracy * s.units.qqr2e * q * q) / (a * a);
  }

  /** Charges, neutrality and the box checks shared by Ewald and PPPM. */
  protected setupCharges(s: SimState, geom: Geometry, ctx: StyleContext): void {
    let qs = 0, q2 = 0;
    for (let i = 0; i < s.n; i++) { qs += s.q[i]; q2 += s.q[i] * s.q[i]; }
    this.qsum = qs;
    this.qsqsum = q2;
    this.natoms = s.n;
    if (!hasChargeStyle(s.atomStyle)) throw new StyleError(`kspace_style ${this.name} needs charges (atom_style charge or full)`);
    if (q2 === 0) ctx.log(`WARNING: kspace_style ${this.name}: no atoms are charged`);
    if (Math.abs(qs) > 1e-5) ctx.log(`WARNING: kspace_style ${this.name}: the system is not charge neutral (net charge ${qs}); a uniform neutralizing background is assumed`);
    const p = s.box.periodic;
    if (this.slabAuto) {
      this.slab = this.autoSlabVolfactor(s, geom);
      ctx.log(`  auto slab volfactor = ${this.slab}\n  auto slab extended z = ${geom.lz * this.slab}`);
    }
    if (this.slab > 1) {
      if (!p[0] || !p[1] || p[2]) throw new StyleError('kspace_modify slab needs boundary p p f (or p p s/m)');
      // "Because the slab normal must be the Cartesian z axis, only an *xy* tilt is allowed" (kspace_modify.html)
      if (s.box.triclinic && (geom.xz !== 0 || geom.yz !== 0)) {
        throw new StyleError('Triclinic slab (EW3DC) correction requires xz = yz = 0 (the slab normal must be the z axis); xy tilt is allowed');
      }
    } else if (!p[0] || !p[1] || !p[2]) {
      throw new StyleError(`kspace_style ${this.name} needs a fully periodic box (or kspace_modify slab for p p f)`);
    }
    this.volume = geom.volume(3) * this.slab;
  }

  /**
   * kspace_modify slab auto. kspace_modify.html: "*auto* chooses the extended z dimension from the
   * current normalized force tolerance, lateral box dimensions, and G-ewald". Measured with native
   * LAMMPS (black box, real and metal units): the volfactor equals
   *   1 + ln(1/T) max(Lx, Ly) / (2 pi Lz)
   * with T the normalized tolerance (the accuracy, or force / (qqr2e q^2 / a^2) for a force value),
   * Lz the box length before the slab extension. G-ewald did not change it (0.2, 0.4 and 0.8 gave
   * the same volfactor), nor did the charges. The same formula was measured for ewald and pppm.
   * The formula is measured exact for T from 0.01 to 0.35 (and for T = 1e-5 .. 1e-20); above
   * about 0.4 the native value departs from it and is not reproduced, so such T are refused.
   * In lj units the native value did not follow the same T mapping, so only real and metal are
   * accepted. Native refusal: kspace_modify slab auto requires a normalized force tolerance between 0 and 1.
   */
  protected autoSlabVolfactor(s: SimState, geom: Geometry): number {
    const style = s.units.style;
    if (style !== 'real' && style !== 'metal') {
      throw new StyleError(`kspace_modify slab auto is measured for units real and metal only (units ${style} are not reproduced)`);
    }
    const unit = s.units.qqr2e * QELECTRON[style] * QELECTRON[style] / (ANGSTROM[style] * ANGSTROM[style]);
    const T = this.forceAbs > 0 ? this.forceAbs / unit : this.accuracy;
    if (!(T > 0 && T < 1)) throw new StyleError('kspace_modify slab auto requires a normalized force tolerance between 0 and 1');
    if (T > 0.35) throw new StyleError(`kspace_modify slab auto is reproduced only for a normalized force tolerance up to 0.35 (got ${T})`);
    const lateral = Math.max(geom.lx, geom.ly);
    return 1 + Math.log(1 / T) * lateral / (2 * Math.PI * geom.lz);
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

  /**
   * qsum and qsqsum follow the current charges at every compute (the self and background terms use them),
   * while G-ewald, the kmax and the mesh stay as chosen at init. Measured with native LAMMPS (black box): with
   * fix adapt changing the charges each step, the elong self and background terms follow the charges of that step.
   */
  protected refreshChargeSums(s: SimState): void {
    if (this.chargeSumsFrozen) return;
    let qs = 0, q2 = 0;
    for (let i = 0; i < s.n; i++) { qs += s.q[i]; q2 += s.q[i] * s.q[i]; }
    this.qsum = qs;
    this.qsqsum = q2;
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
    this.refreshChargeSums(s);
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

/**
 * kspace_style zero accuracy — kspace_style.html: "The *zero* style does not do any calculations, but is
 * compatible with all pair styles that require some version of a kspace style." The accuracy only sets
 * G-ewald (the real-space splitting of the pair style). Measured with native LAMMPS (black box, real
 * units, lj/cut/coul/long 6.0, 75 charged atoms in a 16 x 16 x 12.8 box, no slab): with kspace_style zero
 * 1.0e-5 and no gewald, E_coul was -110.249089294577 and E_long 0, which is the G-ewald of the Kolafa-Perram
 * estimate with the box volume (g = 0.4724443323721925); with kspace_modify gewald 0.4 the E_coul was
 * -223.471826518548, the same as kspace_style ewald. Measured: kspace_style zero 0.0 without gewald gives
 * KSpace accuracy must be > 0; with gewald 0.4 it is accepted. Measured: kspace_style zero with no
 * argument gives Illegal kspace_style zero command. The pressure with zero is the real-space virial only.
 */
export class KSpaceZero extends KSpaceBase {
  readonly name = 'zero';

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal kspace_style zero command');
    this.accuracy = parseNum(args[0], 'accuracy');
  }

  modify(key: string, values: string[]): number {
    // slab and force change the volume or the accuracy that G-ewald uses; their effect with zero was not measured
    if (key === 'slab' || key === 'force') throw new StyleError(`kspace_modify ${key} is not supported by kspace_style zero`);
    return super.modify(key, values);
  }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    this.cutCoul = cutCoul;
    let qs = 0, q2 = 0;
    for (let i = 0; i < s.n; i++) { qs += s.q[i]; q2 += s.q[i] * s.q[i]; }
    this.qsum = qs;
    this.qsqsum = q2;
    this.natoms = s.n;
    this.volume = geom.volume(3);
    if (!(this.gewaldUser > 0) && !(this.accuracy > 0)) throw new StyleError('KSpace accuracy must be > 0');
    this.gEwald = this.chooseGEwald(s, qqrd2e);
    ctx.log(`  G vector (1/distance) = ${this.gEwald}`);
  }

  /** No wave-space sum: the energy, forces and virial stay zero. */
  compute(_kc: KSpaceCompute): void {
    // "does not do any calculations"
  }
}

/** Inverse of the upper-triangular box matrix [[lx, xy, xz], [0, ly, yz], [0, 0, lz]]. */
export const invertUpper = (lx: number, xy: number, xz: number, ly: number, yz: number, lz: number): number[][] => [
  [1 / lx, -xy / (lx * ly), (xy * yz - ly * xz) / (lx * ly * lz)],
  [0, 1 / ly, -yz / (ly * lz)],
  [0, 0, 1 / lz],
];
