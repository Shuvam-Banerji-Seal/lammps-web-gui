/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ELECTRODE/ewald_electrode.cpp and ewald_electrode.h, src/ELECTRODE/electrode_kspace.h (the interface the
 * fixes call), and the Ewald base it extends (src/KSPACE/ewald.cpp, rms and the k-vector setup) at commit b332f28.
 * ELECTRODE contributing authors: Ludwig Ahrens-Iwers (TUHH), Shern Tee (UQ), Robert Meissner (TUHH), and the LAMMPS
 * developers. Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * kspace_style ewald/electrode: the Ewald sum of the core ewald style plus the matrix and vector routines the
 * constant-potential fix needs. Single process: every atom is local, so the MPI gathers of the C++ are not needed.
 * Not ported (refused with a StyleError): the wire correction, kspace_modify slab nozforce or auto, per-atom energy
 * and virial, and triclinic boxes (the C++ refuses those too).
 */
import { KSpace, StyleError, type KSpaceCompute, type StyleContext } from '../../engine/force/types';
import { ANGSTROM, QELECTRON, modifyNoEffect } from '../../engine/force/kspace/ewald';
import { parseNum } from '../../engine/force/util';
import { hasChargeStyle } from '../../engine/atoms';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { MY_PI, MY_PI2, MY_PIS } from './electrode_math';
import { BoundaryCorrection, NoBoundaryCorrection, SlabDipole, Slab2d, type BoundaryEnv, inGroup } from './boundary_correction';
import { log as libmLog } from '../libm/log';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';

/** electrode_kspace.h: what the constant-potential fixes call on a k-space style. */
export interface ElectrodeKSpace {
  readonly qqrd2e: number;
  readonly gEwald: number;
  computeVector(s: SimState, vec: Float64Array, sensorBit: number, sourceBit: number, invert: boolean): void;
  computeVectorCorr(s: SimState, vec: Float64Array, sensorBit: number, sourceBit: number, invert: boolean): void;
  computeMatrix(s: SimState, imat: Int32Array, matrix: Float64Array, ngroup: number): void;
  computeMatrixCorr(s: SimState, imat: Int32Array, matrix: Float64Array, ngroup: number): void;
}

/** ewald.cpp Ewald::rms: the rms force error of the k-space sum for km vectors along a box length prd. */
const rms = (km: number, prd: number, natoms: number, q2: number, g: number): number => {
  const n = natoms === 0 ? 1 : natoms;
  return 2.0 * q2 * g / prd * Math.sqrt(1.0 / (MY_PI * km * n)) * libmExp(-MY_PI * MY_PI * km * km / (g * g * prd * prd));
};

export class KSpaceEwaldElectrode extends KSpace implements ElectrodeKSpace {
  readonly name = 'ewald/electrode';
  /** kspace_modify force (absolute accuracy); -1 = relative accuracy times the two-charge force. */
  private accuracyAbs = -1;
  /** The accuracy the C++ stores as `accuracy` (relative times the two-charge force, or absolute). */
  private accuracyEff = 0;
  private gewaldFlag = false;
  private kewaldFlag = false;
  private kEwald: [number, number, number] = [1, 1, 1];
  /** 0 none, 1 EW3DC dipole correction (slab), 3 EW2D (kspace_modify slab ew2d). */
  private slabflag = 0;
  private slabVolfactor = 1;
  private twoChargeForce = 0;
  private cutoff = 0;
  qqrd2e = 0;
  private qsum = 0;
  private qsqsum = 0;
  private q2 = 0;
  private natoms = 0;
  private volume = 0;
  private area = 0;
  private unitk: [number, number, number] = [0, 0, 0];
  private kmax = 0;
  private kxmax = 0;
  private kymax = 0;
  private kzmax = 0;
  private gsqmx = 0;
  private kcount = 0;
  private kxvecs = new Int32Array(0);
  private kyvecs = new Int32Array(0);
  private kzvecs = new Int32Array(0);
  private ug = new Float64Array(0);
  private eg = new Float64Array(0);
  private vg = new Float64Array(0);
  private sfacrl = new Float64Array(0);
  private sfacim = new Float64Array(0);
  /** cs[d][(k + K) * n + i] = cos(k unitk[d] x_i), sn likewise, for k = -K..K (K = kmax). */
  private cs: Float64Array[] = [new Float64Array(0), new Float64Array(0), new Float64Array(0)];
  private sn: Float64Array[] = [new Float64Array(0), new Float64Array(0), new Float64Array(0)];
  private csK = -1;
  private csN = -1;
  private eikrStep = -1;
  private boundcorr: BoundaryCorrection = new NoBoundaryCorrection();
  private env: BoundaryEnv = { volume: 0, zprdSlab: 0, area: 0, gEwald: 0, qqrd2e: 0 };
  private initDone = false;

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal kspace_style ewald/electrode command (usage: kspace_style ewald/electrode accuracy)');
    this.accuracy = Math.abs(parseNum(args[0], 'accuracy'));
  }

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'slab':
        if (values[0] === 'ew2d') { this.slabflag = 3; return 1; }
        if (values[0] === 'nozforce' || values[0] === 'auto') {
          throw new StyleError(`kspace_modify slab ${values[0]} is not supported by kspace_style ewald/electrode in the browser engine`);
        }
        this.slabVolfactor = parseNum(values[0], 'slab volfactor');
        if (this.slabVolfactor <= 1.0) throw new StyleError('kspace_modify slab volfactor must be > 1.0');
        this.slabflag = 1;
        return 1;
      case 'gewald': {
        const v = parseNum(values[0], 'gewald');
        this.gewaldFlag = v !== 0;
        this.gEwald = v;
        return 1;
      }
      case 'kmax/ewald': {
        const k = [0, 1, 2].map((d) => parseNum(values[d], 'kmax/ewald'));
        if (k.some((v) => !Number.isInteger(v) || v < 0)) throw new StyleError('kspace_modify kmax/ewald values must be integers >= 0');
        this.kEwald = k as [number, number, number];
        this.kewaldFlag = k.some((v) => v > 0);
        return 3;
      }
      case 'force':
        this.accuracyAbs = parseNum(values[0], 'force');
        return 1;
      case 'mesh': case 'order': case 'diff':
        throw new StyleError(`kspace_modify ${key} applies to pppm, not ewald/electrode`);
      case 'wire':
        throw new StyleError('kspace_modify wire (the wire correction) is not supported by the browser engine');
      default:
        return modifyNoEffect(key, values);
    }
  }

  init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    if (geom.triclinic) throw new StyleError('Cannot (yet) use ewald/electrode with triclinic box');
    if (s.dimension === 2) throw new StyleError('Cannot use ewald/electrode with 2d simulation');
    if (!hasChargeStyle(s.atomStyle)) throw new StyleError('KSpace style ewald/electrode requires atom attribute q');
    const p = geom.periodic;
    const xyPeriodic = p[0] && p[1];
    const zPeriodic = p[2];
    if (this.slabflag === 0 && !(p[0] && p[1] && p[2])) throw new StyleError('Cannot use non-periodic boundaries with ewald/electrode');
    if (this.slabflag) {
      // slab ewald/electrode: p p f (the z boundary is fixed, so it is not periodic here)
      if (!xyPeriodic || zPeriodic) throw new StyleError('Incorrect boundaries with slab ewald/electrod (use boundary p p f)');
      this.boundcorr = this.slabflag === 1 ? new SlabDipole() : new Slab2d();
    } else {
      this.boundcorr = new NoBoundaryCorrection();
    }
    this.cutoff = cutCoul;
    this.twoChargeForce = s.units.qqr2e * (QELECTRON[s.units.style] * QELECTRON[s.units.style]) / (ANGSTROM[s.units.style] * ANGSTROM[s.units.style]);
    this.qqrd2e = qqrd2e;
    this.qsumQsq(s);
    if (this.accuracyAbs >= 0) this.accuracyEff = this.accuracyAbs;
    else this.accuracyEff = this.accuracy * this.twoChargeForce;
    this.natoms = s.n;
    const xprd = geom.lx, yprd = geom.ly, zprd = geom.lz;
    // initial G-ewald estimate from the desired accuracy and the real-space cutoff (fluid volume: zprd, not zprd_slab)
    if (!this.gewaldFlag) {
      if (this.accuracyEff <= 0) throw new StyleError('KSpace accuracy must be > 0');
      if (this.q2 === 0) throw new StyleError("Must use 'kspace_modify gewald' for uncharged system");
      let g = this.accuracyEff * Math.sqrt(this.natoms * this.cutoff * xprd * yprd * zprd) / (2.0 * this.q2);
      if (g >= 1.0) g = (1.35 - 0.15 * libmLog(this.accuracyEff)) / this.cutoff;
      else g = Math.sqrt(-libmLog(g)) / this.cutoff;
      this.gEwald = g;
    }
    this.setup(geom);
    this.env = { volume: this.volume, zprdSlab: zprd * this.slabVolfactor, area: this.area, gEwald: this.gEwald, qqrd2e };
    this.initDone = true;
    ctx.log(`Ewald/electrode initialization ...\n  G vector (1/distance) = ${this.gEwald}\n  KSpace vectors: actual max1d max3d = ${this.kcount} ${this.kmax} ${4 * this.kmax ** 3 + 6 * this.kmax ** 2 + 3 * this.kmax}\n                  kxmax kymax kzmax  = ${this.kxmax} ${this.kymax} ${this.kzmax}`);
  }

  /** KSpace::qsum_qsq: total and squared charge (q2 = qsqsum * qqrd2e). */
  private qsumQsq(s: SimState): void {
    let qs = 0, q2s = 0;
    for (let i = 0; i < s.n; i++) { qs += s.q[i]; q2s += s.q[i] * s.q[i]; }
    this.qsum = qs;
    this.qsqsum = q2s;
    this.q2 = q2s * this.qqrd2e;
  }

  /** ewald_electrode.cpp setup(): volume factors, the kmax search (or the user kmax) and the coefficients. */
  private setup(geom: Geometry): void {
    const xprd = geom.lx, yprd = geom.ly, zprd = geom.lz;
    const zprdSlab = zprd * this.slabVolfactor;
    this.volume = xprd * yprd * zprdSlab;
    this.area = xprd * yprd;
    this.unitk = [2 * MY_PI / xprd, 2 * MY_PI / yprd, 2 * MY_PI / zprdSlab];
    if (!this.kewaldFlag) {
      // determine kmax: function of the box size, accuracy and G-ewald
      const n = this.natoms;
      let kx = 1, ky = 1, kz = 1;
      let err = rms(kx, xprd, n, this.q2, this.gEwald);
      while (err > this.accuracyEff) { kx++; err = rms(kx, xprd, n, this.q2, this.gEwald); }
      err = rms(ky, yprd, n, this.q2, this.gEwald);
      while (err > this.accuracyEff) { ky++; err = rms(ky, yprd, n, this.q2, this.gEwald); }
      err = rms(kz, zprdSlab, n, this.q2, this.gEwald);
      while (err > this.accuracyEff) { kz++; err = rms(kz, zprdSlab, n, this.q2, this.gEwald); }
      this.kxmax = kx; this.kymax = ky; this.kzmax = kz;
    } else {
      [this.kxmax, this.kymax, this.kzmax] = this.kEwald;
    }
    this.kmax = Math.max(this.kxmax, this.kymax, this.kzmax);
    const gsqxmx = this.unitk[0] * this.unitk[0] * this.kxmax * this.kxmax;
    const gsqymx = this.unitk[1] * this.unitk[1] * this.kymax * this.kymax;
    const gsqzmx = this.unitk[2] * this.unitk[2] * this.kzmax * this.kzmax;
    this.gsqmx = Math.max(Math.max(gsqxmx, gsqymx), gsqzmx);
    this.gsqmx *= 1.00001;
    this.coeffs();
  }

  /** ewald_electrode.cpp coeffs(): the half-space k-vectors with their Green's function ug, field eg and virial vg. */
  private coeffs(): void {
    const kmax = this.kmax, kxmax = this.kxmax, kymax = this.kymax, kzmax = this.kzmax;
    const u = this.unitk;
    const gsqmx = this.gsqmx;
    const gInvSq = 1.0 / (this.gEwald * this.gEwald);
    const preu = 4.0 * MY_PI / this.volume;
    const kx: number[] = [], ky: number[] = [], kz: number[] = [];
    const ugL: number[] = [], egL: number[] = [], vgL: number[] = [];
    const push = (a: number, b: number, c: number, ugv: number, e0: number, e1: number, e2: number,
      v0: number, v1: number, v2: number, v3: number, v4: number, v5: number): void => {
      kx.push(a); ky.push(b); kz.push(c); ugL.push(ugv);
      egL.push(e0, e1, e2);
      vgL.push(v0, v1, v2, v3, v4, v5);
    };
    // (m,0,0), (0,m,0), (0,0,m); skip (0,0,m) for EW2D (slabflag == 3)
    for (let m = 1; m <= kmax; m++) {
      let sqk = (m * u[0]) * (m * u[0]);
      if (sqk <= gsqmx) {
        const ugv = preu * libmExp(-0.25 * sqk * gInvSq) / sqk;
        const vterm = -2.0 * (1.0 / sqk + 0.25 * gInvSq);
        push(m, 0, 0, ugv, 2.0 * u[0] * m * ugv, 0, 0, 1.0 + vterm * (u[0] * m) * (u[0] * m), 1.0, 1.0, 0, 0, 0);
      }
      sqk = (m * u[1]) * (m * u[1]);
      if (sqk <= gsqmx) {
        const ugv = preu * libmExp(-0.25 * sqk * gInvSq) / sqk;
        const vterm = -2.0 * (1.0 / sqk + 0.25 * gInvSq);
        push(0, m, 0, ugv, 0, 2.0 * u[1] * m * ugv, 0, 1.0, 1.0 + vterm * (u[1] * m) * (u[1] * m), 1.0, 0, 0, 0);
      }
      sqk = (m * u[2]) * (m * u[2]);
      if (sqk <= gsqmx && this.slabflag !== 3) {
        const ugv = preu * libmExp(-0.25 * sqk * gInvSq) / sqk;
        const vterm = -2.0 * (1.0 / sqk + 0.25 * gInvSq);
        push(0, 0, m, ugv, 0, 0, 2.0 * u[2] * m * ugv, 1.0, 1.0, 1.0 + vterm * (u[2] * m) * (u[2] * m), 0, 0, 0);
      }
    }
    // (k,l,0), (k,-l,0)
    for (let k = 1; k <= kxmax; k++) {
      for (let l = 1; l <= kymax; l++) {
        const sqk = (u[0] * k) * (u[0] * k) + (u[1] * l) * (u[1] * l);
        if (sqk <= gsqmx) {
          const ugv = preu * libmExp(-0.25 * sqk * gInvSq) / sqk;
          const vterm = -2.0 * (1.0 / sqk + 0.25 * gInvSq);
          const a = 1.0 + vterm * (u[0] * k) * (u[0] * k), b = 1.0 + vterm * (u[1] * l) * (u[1] * l);
          push(k, l, 0, ugv, 2.0 * u[0] * k * ugv, 2.0 * u[1] * l * ugv, 0, a, b, 1.0, vterm * u[0] * k * u[1] * l, 0, 0);
          push(k, -l, 0, ugv, 2.0 * u[0] * k * ugv, -2.0 * u[1] * l * ugv, 0, a, b, 1.0, -vterm * u[0] * k * u[1] * l, 0, 0);
        }
      }
    }
    // (0,l,m), (0,l,-m)
    for (let l = 1; l <= kymax; l++) {
      for (let m = 1; m <= kzmax; m++) {
        const sqk = (u[1] * l) * (u[1] * l) + (u[2] * m) * (u[2] * m);
        if (sqk <= gsqmx) {
          const ugv = preu * libmExp(-0.25 * sqk * gInvSq) / sqk;
          const vterm = -2.0 * (1.0 / sqk + 0.25 * gInvSq);
          const b = 1.0 + vterm * (u[1] * l) * (u[1] * l), c = 1.0 + vterm * (u[2] * m) * (u[2] * m);
          push(0, l, m, ugv, 0, 2.0 * u[1] * l * ugv, 2.0 * u[2] * m * ugv, 1.0, b, c, 0, 0, vterm * u[1] * l * u[2] * m);
          push(0, l, -m, ugv, 0, 2.0 * u[1] * l * ugv, -2.0 * u[2] * m * ugv, 1.0, b, c, 0, 0, -vterm * u[1] * l * u[2] * m);
        }
      }
    }
    // (k,0,m), (k,0,-m)
    for (let k = 1; k <= kxmax; k++) {
      for (let m = 1; m <= kzmax; m++) {
        const sqk = (u[0] * k) * (u[0] * k) + (u[2] * m) * (u[2] * m);
        if (sqk <= gsqmx) {
          const ugv = preu * libmExp(-0.25 * sqk * gInvSq) / sqk;
          const vterm = -2.0 * (1.0 / sqk + 0.25 * gInvSq);
          const a = 1.0 + vterm * (u[0] * k) * (u[0] * k), c = 1.0 + vterm * (u[2] * m) * (u[2] * m);
          push(k, 0, m, ugv, 2.0 * u[0] * k * ugv, 0, 2.0 * u[2] * m * ugv, a, 1.0, c, 0, vterm * u[0] * k * u[2] * m, 0);
          push(k, 0, -m, ugv, 2.0 * u[0] * k * ugv, 0, -2.0 * u[2] * m * ugv, a, 1.0, c, 0, -vterm * u[0] * k * u[2] * m, 0);
        }
      }
    }
    // (k,l,m), (k,-l,m), (k,l,-m), (k,-l,-m)
    for (let k = 1; k <= kxmax; k++) {
      for (let l = 1; l <= kymax; l++) {
        for (let m = 1; m <= kzmax; m++) {
          const sqk = (u[0] * k) * (u[0] * k) + (u[1] * l) * (u[1] * l) + (u[2] * m) * (u[2] * m);
          if (sqk <= gsqmx) {
            const ugv = preu * libmExp(-0.25 * sqk * gInvSq) / sqk;
            const vterm = -2.0 * (1.0 / sqk + 0.25 * gInvSq);
            const a = 1.0 + vterm * (u[0] * k) * (u[0] * k), b = 1.0 + vterm * (u[1] * l) * (u[1] * l);
            const c = 1.0 + vterm * (u[2] * m) * (u[2] * m);
            const e0 = 2.0 * u[0] * k * ugv, e1 = 2.0 * u[1] * l * ugv, e2 = 2.0 * u[2] * m * ugv;
            const xy = vterm * u[0] * k * u[1] * l, xz = vterm * u[0] * k * u[2] * m, yz = vterm * u[1] * l * u[2] * m;
            push(k, l, m, ugv, e0, e1, e2, a, b, c, xy, xz, yz);
            push(k, -l, m, ugv, e0, -e1, e2, a, b, c, -xy, xz, -yz);
            push(k, l, -m, ugv, e0, e1, -e2, a, b, c, xy, -xz, -yz);
            push(k, -l, -m, ugv, e0, -e1, -e2, a, b, c, -xy, -xz, yz);
          }
        }
      }
    }
    this.kcount = kx.length;
    this.kxvecs = Int32Array.from(kx);
    this.kyvecs = Int32Array.from(ky);
    this.kzvecs = Int32Array.from(kz);
    this.ug = Float64Array.from(ugL);
    this.eg = Float64Array.from(egL);
    this.vg = Float64Array.from(vgL);
    this.sfacrl = new Float64Array(this.kcount);
    this.sfacim = new Float64Array(this.kcount);
  }

  /** Storage for cs/sn of n atoms and k = -K..K (ewald_electrode.cpp create3d_offset with -kmax..kmax). */
  private ensureCs(n: number, K: number): void {
    if (this.csN === n && this.csK === K) return;
    this.csN = n;
    this.csK = K;
    for (let d = 0; d < 3; d++) {
      this.cs[d] = new Float64Array((2 * K + 1) * n);
      this.sn[d] = new Float64Array((2 * K + 1) * n);
    }
    this.eikrStep = -1;
  }

  /**
   * ewald_electrode.cpp eik_dot_r(): cos/sin of every k * x_i for every atom, and the structure factors
   * sum_i q_i exp(i k.x_i) of the k-vectors, in the same order as coeffs().
   */
  private eikDotR(s: SimState): void {
    const n = s.n, K = this.kmax;
    this.ensureCs(n, K);
    const x = s.x, q = s.q, u = this.unitk;
    const cs = this.cs, sn = this.sn;
    for (let ic = 0; ic < 3; ic++) {
      const C = cs[ic], S = sn[ic];
      for (let i = 0; i < n; i++) {
        C[K * n + i] = 1.0;
        S[K * n + i] = 0.0;
        const c1 = libmCos(u[ic] * x[3 * i + ic]), s1 = libmSin(u[ic] * x[3 * i + ic]);
        C[(K + 1) * n + i] = c1;
        S[(K + 1) * n + i] = s1;
        C[(K - 1) * n + i] = c1;
        S[(K - 1) * n + i] = -s1;
      }
      for (let m = 2; m <= K; m++) {
        for (let i = 0; i < n; i++) {
          const cm1 = C[(K + m - 1) * n + i], sm1 = S[(K + m - 1) * n + i];
          const c1 = C[(K + 1) * n + i], s1 = S[(K + 1) * n + i];
          C[(K + m) * n + i] = cm1 * c1 - sm1 * s1;
          S[(K + m) * n + i] = sm1 * c1 + cm1 * s1;
          C[(K - m) * n + i] = C[(K + m) * n + i];
          S[(K - m) * n + i] = -S[(K + m) * n + i];
        }
      }
    }
    const sfacrl = this.sfacrl, sfacim = this.sfacim;
    let nn = 0;
    // structure factor of one pair of cs/sn index sets: sum_i q_i (c * ... ) as in the C++ loops
    const sumq = (f: (i: number) => number): number => {
      let acc = 0;
      for (let i = 0; i < n; i++) acc += q[i] * f(i);
      return acc;
    };
    const c = (d: number, k: number, i: number): number => cs[d][(K + k) * n + i];
    const sv = (d: number, k: number, i: number): number => sn[d][(K + k) * n + i];
    // (k,0,0), (0,l,0), (0,0,m)
    for (let m = 1; m <= K; m++) {
      for (let ic = 0; ic < 3; ic++) {
        const sqk = m * u[ic] * m * u[ic];
        if (sqk <= this.gsqmx) {
          const cstr = sumq((i) => c(ic, m, i));
          const sstr = sumq((i) => sv(ic, m, i));
          if (this.slabflag !== 3 || ic < 2) { sfacrl[nn] = cstr; sfacim[nn++] = sstr; }
        }
      }
    }
    // (k,l,0), (k,-l,0)
    for (let k = 1; k <= this.kxmax; k++) {
      for (let l = 1; l <= this.kymax; l++) {
        const sqk = (k * u[0] * k * u[0]) + (l * u[1] * l * u[1]);
        if (sqk <= this.gsqmx) {
          const cstr1 = sumq((i) => c(0, k, i) * c(1, l, i) - sv(0, k, i) * sv(1, l, i));
          const sstr1 = sumq((i) => sv(0, k, i) * c(1, l, i) + c(0, k, i) * sv(1, l, i));
          const cstr2 = sumq((i) => c(0, k, i) * c(1, l, i) + sv(0, k, i) * sv(1, l, i));
          const sstr2 = sumq((i) => sv(0, k, i) * c(1, l, i) - c(0, k, i) * sv(1, l, i));
          sfacrl[nn] = cstr1; sfacim[nn++] = sstr1;
          sfacrl[nn] = cstr2; sfacim[nn++] = sstr2;
        }
      }
    }
    // (0,l,m), (0,l,-m)
    for (let l = 1; l <= this.kymax; l++) {
      for (let m = 1; m <= this.kzmax; m++) {
        const sqk = (l * u[1] * l * u[1]) + (m * u[2] * m * u[2]);
        if (sqk <= this.gsqmx) {
          const cstr1 = sumq((i) => c(1, l, i) * c(2, m, i) - sv(1, l, i) * sv(2, m, i));
          const sstr1 = sumq((i) => sv(1, l, i) * c(2, m, i) + c(1, l, i) * sv(2, m, i));
          const cstr2 = sumq((i) => c(1, l, i) * c(2, m, i) + sv(1, l, i) * sv(2, m, i));
          const sstr2 = sumq((i) => sv(1, l, i) * c(2, m, i) - c(1, l, i) * sv(2, m, i));
          sfacrl[nn] = cstr1; sfacim[nn++] = sstr1;
          sfacrl[nn] = cstr2; sfacim[nn++] = sstr2;
        }
      }
    }
    // (k,0,m), (k,0,-m)
    for (let k = 1; k <= this.kxmax; k++) {
      for (let m = 1; m <= this.kzmax; m++) {
        const sqk = (k * u[0] * k * u[0]) + (m * u[2] * m * u[2]);
        if (sqk <= this.gsqmx) {
          const cstr1 = sumq((i) => c(0, k, i) * c(2, m, i) - sv(0, k, i) * sv(2, m, i));
          const sstr1 = sumq((i) => sv(0, k, i) * c(2, m, i) + c(0, k, i) * sv(2, m, i));
          const cstr2 = sumq((i) => c(0, k, i) * c(2, m, i) + sv(0, k, i) * sv(2, m, i));
          const sstr2 = sumq((i) => sv(0, k, i) * c(2, m, i) - c(0, k, i) * sv(2, m, i));
          sfacrl[nn] = cstr1; sfacim[nn++] = sstr1;
          sfacrl[nn] = cstr2; sfacim[nn++] = sstr2;
        }
      }
    }
    // (k,l,m), (k,-l,m), (k,l,-m), (k,-l,-m)
    for (let k = 1; k <= this.kxmax; k++) {
      for (let l = 1; l <= this.kymax; l++) {
        for (let m = 1; m <= this.kzmax; m++) {
          const sqk = (k * u[0] * k * u[0]) + (l * u[1] * l * u[1]) + (m * u[2] * m * u[2]);
          if (sqk <= this.gsqmx) {
            const clpm = (i: number) => c(1, l, i) * c(2, m, i) - sv(1, l, i) * sv(2, m, i);
            const slpm = (i: number) => sv(1, l, i) * c(2, m, i) + c(1, l, i) * sv(2, m, i);
            const cstr1 = sumq((i) => c(0, k, i) * clpm(i) - sv(0, k, i) * slpm(i));
            const sstr1 = sumq((i) => sv(0, k, i) * clpm(i) + c(0, k, i) * slpm(i));
            const clpm2 = (i: number) => c(1, l, i) * c(2, m, i) + sv(1, l, i) * sv(2, m, i);
            const slpm2 = (i: number) => -sv(1, l, i) * c(2, m, i) + c(1, l, i) * sv(2, m, i);
            const cstr2 = sumq((i) => c(0, k, i) * clpm2(i) - sv(0, k, i) * slpm2(i));
            const sstr2 = sumq((i) => sv(0, k, i) * clpm2(i) + c(0, k, i) * slpm2(i));
            const clpm3 = (i: number) => c(1, l, i) * c(2, m, i) + sv(1, l, i) * sv(2, m, i);
            const slpm3 = (i: number) => sv(1, l, i) * c(2, m, i) - c(1, l, i) * sv(2, m, i);
            const cstr3 = sumq((i) => c(0, k, i) * clpm3(i) - sv(0, k, i) * slpm3(i));
            const sstr3 = sumq((i) => sv(0, k, i) * clpm3(i) + c(0, k, i) * slpm3(i));
            const clpm4 = (i: number) => c(1, l, i) * c(2, m, i) - sv(1, l, i) * sv(2, m, i);
            const slpm4 = (i: number) => -sv(1, l, i) * c(2, m, i) - c(1, l, i) * sv(2, m, i);
            const cstr4 = sumq((i) => c(0, k, i) * clpm4(i) - sv(0, k, i) * slpm4(i));
            const sstr4 = sumq((i) => sv(0, k, i) * clpm4(i) + c(0, k, i) * slpm4(i));
            sfacrl[nn] = cstr1; sfacim[nn++] = sstr1;
            sfacrl[nn] = cstr2; sfacim[nn++] = sstr2;
            sfacrl[nn] = cstr3; sfacim[nn++] = sstr3;
            sfacrl[nn] = cstr4; sfacim[nn++] = sstr4;
          }
        }
      }
    }
    if (nn !== this.kcount) throw new StyleError(`ewald/electrode internal error: ${nn} structure factors for ${this.kcount} k-vectors`);
  }

  /** update_eikr(): recompute the cos/sin tables once per timestep (or always when enforced). */
  private updateEikr(s: SimState, enforce: boolean): void {
    if (this.eikrStep < s.step || enforce) {
      this.eikrStep = s.step;
      this.eikDotR(s);
    }
  }

  /** The k-space energy and forces, the virial, and the boundary correction (EwaldElectrode::compute). */
  compute(kc: KSpaceCompute): void {
    if (!this.initDone) throw new StyleError('kspace_style ewald/electrode used before init');
    if (kc.eatom || kc.vatom) throw new StyleError('compute pe/atom and stress/atom with kspace_style ewald/electrode are not supported by the browser engine');
    const s = kc.s, f = kc.f, q = s.q, n = s.n;
    const qscale = kc.qqrd2e;
    this.qsumQsq(s);
    if (this.qsqsum === 0) return;
    this.updateEikr(s, true);
    const cs = this.cs, sn = this.sn, K = this.kmax;
    const c = (d: number, k: number, i: number): number => cs[d][(K + k) * n + i];
    const sv = (d: number, k: number, i: number): number => sn[d][(K + k) * n + i];
    const ek = new Float64Array(3 * n);
    const sfacrl = this.sfacrl, sfacim = this.sfacim;
    for (let k = 0; k < this.kcount; k++) {
      const kx = this.kxvecs[k], ky = this.kyvecs[k], kz = this.kzvecs[k];
      for (let i = 0; i < n; i++) {
        const cypz = c(1, ky, i) * c(2, kz, i) - sv(1, ky, i) * sv(2, kz, i);
        const sypz = sv(1, ky, i) * c(2, kz, i) + c(1, ky, i) * sv(2, kz, i);
        const exprl = c(0, kx, i) * cypz - sv(0, kx, i) * sypz;
        const expim = sv(0, kx, i) * cypz + c(0, kx, i) * sypz;
        const partial = expim * sfacrl[k] - exprl * sfacim[k];
        ek[3 * i] += partial * this.eg[3 * k];
        ek[3 * i + 1] += partial * this.eg[3 * k + 1];
        ek[3 * i + 2] += partial * this.eg[3 * k + 2];
      }
    }
    // convert E-field to force (wire is not supported, so x and y always; slabflag != 2 always here)
    for (let i = 0; i < n; i++) {
      f[3 * i] += qscale * q[i] * ek[3 * i];
      f[3 * i + 1] += qscale * q[i] * ek[3 * i + 1];
      f[3 * i + 2] += qscale * q[i] * ek[3 * i + 2];
    }
    // energy and virial
    let energy = 0;
    const virial = new Float64Array(6);
    for (let k = 0; k < this.kcount; k++) {
      const s2 = sfacrl[k] * sfacrl[k] + sfacim[k] * sfacim[k];
      energy += this.ug[k] * s2;
      for (let j = 0; j < 6; j++) virial[j] += this.ug[k] * s2 * this.vg[6 * k + j];
    }
    energy -= this.gEwald * this.qsqsum / MY_PIS + MY_PI2 * this.qsum * this.qsum / (this.gEwald * this.gEwald * this.volume);
    energy *= qscale;
    for (let j = 0; j < 6; j++) kc.acc.vlong[j] += virial[j] * qscale;
    kc.acc.elong += energy;
    // boundary correction (energy and forces)
    const env = { ...this.env, qqrd2e: qscale };
    kc.acc.elong += this.boundcorr.computeCorr(s, f, this.qsum, env);
  }

  // ------------------------------------------------------------ electrode interface

  /** compute_vector(): k-space part of the b vector (potential at the sensor atoms from the source charges). */
  computeVector(s: SimState, vec: Float64Array, sensorBit: number, sourceBit: number, invert: boolean): void {
    this.updateEikr(s, false);
    const n = s.n, K = this.kmax;
    const q = s.q, mask = s.mask;
    const c = (d: number, k: number, i: number): number => this.cs[d][(K + k) * n + i];
    const sv = (d: number, k: number, i: number): number => this.sn[d][(K + k) * n + i];
    const qCos = new Float64Array(this.kcount), qSin = new Float64Array(this.kcount);
    for (let k = 0; k < this.kcount; k++) {
      const kx = this.kxvecs[k], ky = this.kyvecs[k], kz = this.kzvecs[k];
      let qc = 0, qs = 0;
      for (let i = 0; i < n; i++) {
        if (!inGroup(mask[i], sourceBit, invert)) continue;   // only the source atoms
        const cosKxKy = c(0, kx, i) * c(1, ky, i) - sv(0, kx, i) * sv(1, ky, i);
        const sinKxKy = sv(0, kx, i) * c(1, ky, i) + c(0, kx, i) * sv(1, ky, i);
        const cosKr = cosKxKy * c(2, kz, i) - sinKxKy * sv(2, kz, i);
        const sinKr = sinKxKy * c(2, kz, i) + cosKxKy * sv(2, kz, i);
        qc += q[i] * cosKr;
        qs += q[i] * sinKr;
      }
      qCos[k] = qc;
      qSin[k] = qs;
    }
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & sensorBit)) continue;
      let bi = 0;
      for (let k = 0; k < this.kcount; k++) {
        const kx = this.kxvecs[k], ky = this.kyvecs[k], kz = this.kzvecs[k];
        const cosKxKy = c(0, kx, i) * c(1, ky, i) - sv(0, kx, i) * sv(1, ky, i);
        const sinKxKy = sv(0, kx, i) * c(1, ky, i) + c(0, kx, i) * sv(1, ky, i);
        const cosKr = cosKxKy * c(2, kz, i) - sinKxKy * sv(2, kz, i);
        const sinKr = sinKxKy * c(2, kz, i) + cosKxKy * sv(2, kz, i);
        bi += 2 * this.ug[k] * (cosKr * qCos[k] + sinKr * qSin[k]);
      }
      vec[i] += bi;
    }
  }

  computeVectorCorr(s: SimState, vec: Float64Array, sensorBit: number, sourceBit: number, invert: boolean): void {
    this.updateEikr(s, false);
    this.boundcorr.vectorCorr(s, vec, sensorBit, sourceBit, invert, this.env);
  }

  /**
   * compute_matrix(): k-space part of the Coulomb matrix between the electrode atoms (imat[i] = matrix index, -1 for
   * atoms outside the electrodes). Fills the lower triangle and mirrors it, as the C++ does.
   */
  computeMatrix(s: SimState, imat: Int32Array, matrix: Float64Array, ngroup: number): void {
    this.updateEikr(s, false);
    const n = s.n, K = this.kmax;
    const c = (d: number, k: number, i: number): number => this.cs[d][(K + k) * n + i];
    const sv = (d: number, k: number, i: number): number => this.sn[d][(K + k) * n + i];
    const group: number[] = [];
    for (let i = 0; i < n; i++) if (imat[i] >= 0) group.push(i);
    for (const i of group) {
      for (const j of group) {
        if (imat[j] > imat[i]) continue;   // matrix is symmetric, skip upper triangular matrix
        let aij = 0.0;
        for (let k = 0; k < this.kcount; k++) {
          const kx = this.kxvecs[k], ky = this.kyvecs[k], kz = this.kzvecs[k];
          const cosKxKyI = c(0, kx, i) * c(1, ky, i) - sv(0, kx, i) * sv(1, ky, i);
          const sinKxKyI = sv(0, kx, i) * c(1, ky, i) + c(0, kx, i) * sv(1, ky, i);
          const cosI = cosKxKyI * c(2, kz, i) - sinKxKyI * sv(2, kz, i);
          const sinI = sinKxKyI * c(2, kz, i) + cosKxKyI * sv(2, kz, i);
          const cosKxKyJ = c(0, kx, j) * c(1, ky, j) - sv(0, kx, j) * sv(1, ky, j);
          const sinKxKyJ = sv(0, kx, j) * c(1, ky, j) + c(0, kx, j) * sv(1, ky, j);
          const cosJ = cosKxKyJ * c(2, kz, j) - sinKxKyJ * sv(2, kz, j);
          const sinJ = sinKxKyJ * c(2, kz, j) + cosKxKyJ * sv(2, kz, j);
          aij += 2.0 * this.ug[k] * (cosI * cosJ + sinI * sinJ);
        }
        matrix[imat[i] * ngroup + imat[j]] += aij;
        if (imat[i] !== imat[j]) matrix[imat[j] * ngroup + imat[i]] += aij;
      }
    }
  }

  computeMatrixCorr(s: SimState, imat: Int32Array, matrix: Float64Array, ngroup: number): void {
    this.updateEikr(s, false);
    this.boundcorr.matrixCorr(s, imat, matrix, ngroup, this.env);
  }
}
