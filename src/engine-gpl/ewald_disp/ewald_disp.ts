/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/ewald_disp.cpp and ewald_disp.h at commit b332f28 (Pieter in 't Veld, SNL, and the LAMMPS
 * developers), with the KSpace helpers they use from src/kspace.cpp (qsum_qsq, two_charge, slab handling) and
 * MathSpecial::powint (src/math_special.h). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * kspace_style ewald/disp and ewald/disp/dipole: the Ewald sum of the terms the pair style hands to k-space (its
 * ewald_order): 1/r Coulomb, 1/r^6 dispersion with geometric or arithmetic mixing (seven structure factors for the
 * binomial expansion of the arithmetic sigma), and point dipoles (with the charge-dipole cross terms and torques).
 * Native structure kept: g_ewald from the Coulomb estimate or, without charges, from the Newton solve of the
 * real-space error (Kolafa-Perram for r^-6, Wang-Holm for dipoles); g_ewald_6 = g_ewald unless kspace_modify
 * gewald/disp sets it; kmax per dimension from the rms estimate and the spherical k cutoff gsqmx; the half-space k list
 * in x-major order with the incremental e^(ik.r) products; the per-term energy, virial and self terms; the per-atom
 * energy and virial; the EW3DC slab correction. The coefficient sums of the dispersion and dipole terms are taken at
 * init (native coeff_sums_flag).
 */
import { KSpaceBase, ANGSTROM, QELECTRON, modifyNoEffect } from '../../engine/force/kspace/ewald';
import { Pair, StyleError, type KSpaceCompute, type StyleContext } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import { nativeOrder } from '../../engine/atoms';
import type { Geometry } from '../../engine/domain';
import { PairLJLongCoulLong } from '../../engine/force/pair/lj_long';
import { PairHybrid } from '../../engine/force/pair/hybrid';
import { exp } from '../libm/exp';
import { log } from '../libm/log';
import { pow } from '../libm/pow';
import { sin, cos } from '../libm/sincos';
import { erfc } from '../libm/erf';

const MY_PI = 3.14159265358979323846;
const MY_PIS = 1.77245385090551602729;
const MY_2PI = 6.28318530717958647692;
/** kspace.cpp / ewald_disp.cpp SMALL. */
const SMALL = 0.00001;

const TERM_COUL = 0, TERM_DISP_GEOM = 1, TERM_DISP_ARITH = 2, TERM_DIPOLE = 3, NTERMS = 4;
const EWALD_COUL = 1 << 1, EWALD_DIPOLE = 1 << 3, EWALD_DISP = 1 << 6;
/** Sum channels per term. */
const NSUMS_TERM = [1, 1, 7, 1];
const MAX_NSUMS = 10;

/** MathSpecial::powint: x^n by repeated squaring. */
const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x, yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

/** What a pair style hands to ewald/disp (native extract of ewald_order, ewald_mix, cut_coul, B, epsilon, sigma). */
export interface EwaldDispPair {
  /** Bits EWALD_COUL | EWALD_DIPOLE | EWALD_DISP (pair extract "ewald_order"). */
  ewaldOrder(): number;
  /** The pair's mixing rule (pair extract "ewald_mix"). */
  readonly mix: string;
  /** The Coulomb cutoff of the style (native cut_coul, also with flag_coul off). */
  ewaldCut(): number;
  /** epsilon[i][i] and sigma[i][i] of a type after the pair's init. */
  epsilonOf(i: number): number;
  sigmaOf(i: number): number;
  /** Sets the G-ewald parameters the real-space terms use. */
  setEwald(gEwald: number, gEwald6: number): void;
}

/** The ewald/disp view of the core lj/long/coul/long style. */
const ljLongView = (p: PairLJLongCoulLong): EwaldDispPair => ({
  ewaldOrder: () => (p.coulLongFlag ? EWALD_COUL : 0) | (p.ljLong ? EWALD_DISP : 0),
  mix: p.mix,
  ewaldCut: () => p.cutCoulNative,
  epsilonOf: (i) => p.p.get('epsilon', i, i),
  sigmaOf: (i) => p.p.get('sigma', i, i),
  setEwald: (_g, g6) => { p.gEwaldDisp = g6; },
});

const isEwaldDispPair = (p: unknown): p is { ewaldDisp(): EwaldDispPair } =>
  typeof (p as { ewaldDisp?: unknown }).ewaldDisp === 'function';

export class KSpaceEwaldDisp extends KSpaceBase {
  readonly name: string;
  private mixflag = 0;
  private gewaldUser6 = 0;
  private gEwald6 = 0;
  private termflag = [0, 0, 0, 0];
  private nterms = 0;
  private nsums = 0;
  private kmax = 0;
  private kcount = 0;
  private gsqmx = 0;
  private unitk = new Float64Array(6);
  private hvec = new Float64Array(0);
  private kvec = new Int32Array(0);
  private kenergy = new Float64Array(0);
  private kvirial = new Float64Array(0);
  private sfac = new Float64Array(0);
  private ekr = new Float64Array(0);
  private B = new Float64Array(0);
  private coeffSumX = new Float64Array(MAX_NSUMS);
  private coeffSumX2 = new Float64Array(MAX_NSUMS);
  private energySelf = new Float64Array(NTERMS);
  private virialSelf = new Float64Array(NTERMS);
  private q2 = 0; private b2 = 0; private M2 = 0;
  private accuracyForce = 0;
  private mumurd2e = 1;
  private natomsOriginal = 0;
  private view: EwaldDispPair | null = null;
  private pair: Pair | null = null;
  private setupKey = '';
  private triclinic = false;
  private slabVolfactor = 1;

  constructor(name: 'ewald/disp' | 'ewald/disp/dipole' = 'ewald/disp') {
    super();
    this.name = name;
  }

  /** EwaldDisp::settings: one argument, the relative accuracy (absolute value, at most 1). */
  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError(`Illegal kspace_style ${this.styleShown} command`);
    const v = Number(args[0]);
    if (args[0].trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${args[0]}' in input script or data file`);
    this.accuracy = Math.abs(v);
    if (this.accuracy > 1.0) throw new StyleError(`Invalid relative accuracy ${this.accuracy} for kspace_style ${this.styleShown}`);
  }

  modify(key: string, values: string[]): number {
    switch (key) {
      case 'gewald/disp': {
        const v = Number(values[0]);
        if (!Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${values[0] ?? ''}' for kspace_modify gewald/disp`);
        this.gewaldUser6 = v;
        return 1;
      }
      case 'mix/disp': {
        const v = values[0];
        if (v === 'pair') this.mixflag = 0;
        else if (v === 'geom') this.mixflag = 1;
        else if (v === 'none') this.mixflag = 2;
        else throw new StyleError(`Unknown kspace_modify mix/disp option ${v ?? ''}`);
        return 1;
      }
      // kspace.cpp modify_params stores these for every solver; ewald_disp.cpp reads none of them
      case 'kmax/ewald': return 3;
      case 'mesh': case 'mesh/disp': return 3;
      case 'order': case 'order/disp': case 'force/disp/real': case 'force/disp/kspace': return 1;
      case 'splittol': case 'eigtol': {
        if (Number(values[0]) >= 1.0) throw new StyleError('Kspace_modify splittol value must be smaller than one');
        return 1;
      }
      case 'disp/auto': {
        if (!['yes', 'no', 'on', 'off', 'true', 'false', '1', '0'].includes(values[0] ?? '')) {
          throw new StyleError(`Expected boolean parameter instead of '${values[0] ?? ''}' for kspace_modify disp/auto`);
        }
        return 1;
      }
      case 'slab':
        if (values[0] === 'auto') throw new StyleError(`kspace_modify slab auto is not supported by kspace_style ${this.name} in the browser engine`);
        return super.modify(key, values);
      default: return super.modify(key, values);
    }
  }

  /** The pair view (native extract), or null with the native refusal. */
  private viewOf(pair: Pair | null): { view: EwaldDispPair | null; cutoff: number | null } {
    if (!pair) return { view: null, cutoff: null };
    let p: Pair = pair;
    if (pair instanceof PairHybrid) {
      // PairHybrid::extract: the first sub-style that has ewald_order (cut_coul: the last one that has it)
      const sub = pair.subs.map((x) => x.style).find((x) => x instanceof PairLJLongCoulLong || isEwaldDispPair(x));
      if (sub) p = sub;
    }
    if (p instanceof PairLJLongCoulLong) return { view: ljLongView(p), cutoff: p.cutCoulNative };
    if (isEwaldDispPair(p)) { const v = p.ewaldDisp(); return { view: v, cutoff: v.ewaldCut() }; }
    const c = pair.extract('cut_coul');
    return { view: null, cutoff: typeof c === 'number' ? c : null };
  }

  linkPair(pair: Pair | null): void { this.pair = pair; }
  /**
   * ewald_disp.cpp: dispersionflag = 1 in the constructor, cleared in init when the pair gives no dispersion term
   * ("if (!termflag[TERM_DISP_GEOM] && !termflag[TERM_DISP_ARITH]) dispersionflag = 0;"), before pair_check.
   */
  override dispersion(): boolean {
    const { view } = this.viewOf(this.pair);
    return view !== null && (view.ewaldOrder() & EWALD_DISP) !== 0;
  }

  /**
   * ewald_disp.cpp: dipoleflag = 1 in the constructor, cleared in init when the pair gives no dipole term
   * ("if (!termflag[TERM_DIPOLE]) dipoleflag = 0;"), before pair_check.
   */
  override dipole(): boolean {
    const { view } = this.viewOf(this.pair);
    return view !== null && (view.ewaldOrder() & EWALD_DIPOLE) !== 0;
  }

  /** EwaldDisp::init (and the first setup()). */
  init(s: SimState, geom: Geometry, _cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    ctx.log('EwaldDisp initialization ...');
    this.kmax = -1;
    this.nterms = this.nsums = 0;
    this.triclinic = s.box.triclinic;
    if (s.dimension === 2) throw new StyleError('Cannot use EwaldDisp with 2d simulation');
    const p = geom.periodic;
    const slab = this.slab > 1;
    if (!slab && (!p[0] || !p[1] || !p[2])) throw new StyleError('Cannot use non-periodic boundaries with EwaldDisp');
    if (slab) {
      const b = s.box.boundary;
      if (!p[0] || !p[1] || b[2][0] !== 'f' || b[2][1] !== 'f') throw new StyleError('Incorrect boundaries with slab EwaldDisp');
      if (s.box.triclinic && (geom.yz !== 0 || geom.xz !== 0)) {
        throw new StyleError('Triclinic slab (EW3DC) correction requires xz = yz = 0 (the slab normal must be the z axis); xy tilt is allowed');
      }
    }
    this.slabVolfactor = slab ? this.slab : 1;
    this.mumurd2e = qqrd2e;
    const { view, cutoff } = this.viewOf(this.pair);
    if (!view && cutoff === null) throw new StyleError('KSpace style is incompatible with Pair style');
    this.view = view;
    const ewaldOrder = view ? view.ewaldOrder() : EWALD_COUL;
    const ewaldMix = view ? view.mix : 'geometric';
    this.termflag = [0, 0, 0, 0];
    for (let i = 0; i <= 6; i++) {
      if (!(ewaldOrder & (1 << i))) continue;
      let t = -1;
      if (i === 1) t = TERM_COUL;
      else if (i === 3) t = TERM_DIPOLE;
      else if (i === 6) {
        if (this.mixflag === 2) throw new StyleError('kspace_modify mix/disp none is not supported by kspace_style ewald/disp');
        if (this.mixflag === 1 || ewaldMix === 'geometric') t = TERM_DISP_GEOM;
        else if (ewaldMix === 'arithmetic') t = TERM_DISP_ARITH;
        else throw new StyleError('Unsupported mixing rule in kspace_style ewald/disp');
      } else throw new StyleError('Unsupported order in kspace_style ewald/disp');
      this.termflag[t] = 1;
      this.nterms++;
      this.nsums += NSUMS_TERM[t];
    }
    this.initCoeffs(s);
    this.initCoeffSums(s);
    if (this.termflag[TERM_COUL]) this.qsumQsq(s, qqrd2e, ctx);
    else this.qsqsum = this.qsum = 0.0;
    this.natomsOriginal = s.n;
    // turn off coulombic if no charge
    if (this.termflag[TERM_COUL] && this.qsqsum === 0.0) {
      this.termflag[TERM_COUL] = 0;
      this.nterms -= 1;
      this.nsums -= 1;
    }
    let bsbsum = 0.0;
    this.M2 = 0.0;
    if (this.termflag[TERM_DISP_GEOM]) bsbsum = this.coeffSumX2[1];
    if (this.termflag[TERM_DISP_ARITH]) bsbsum = this.coeffSumX2[2];
    if (this.termflag[TERM_DIPOLE]) this.M2 = this.coeffSumX2[9];
    if (this.termflag[TERM_DIPOLE] && s.units.style === 'electron') throw new StyleError("Cannot (yet) use 'electron' units with dipoles");
    if (this.qsqsum === 0.0 && bsbsum === 0.0 && this.M2 === 0.0) {
      throw new StyleError('Cannot use Ewald/disp solver on system without charged, dipole, or LJ particles');
    }
    if (Math.abs(this.qsum) > SMALL) ctx.log(`WARNING: System is not charge neutral, net charge = ${Number(this.qsum.toPrecision(8))}`);
    const u = s.units;
    const ang = ANGSTROM[u.style], qe = QELECTRON[u.style];
    const twoChargeForce = (u.qqr2e * (qe * qe)) / (ang * ang);
    this.accuracyForce = this.forceAbs >= 0 ? this.forceAbs : this.accuracy * twoChargeForce;
    const accuracy = this.accuracyForce;
    this.q2 = this.qsqsum * qqrd2e;
    this.M2 *= this.mumurd2e;
    this.b2 = bsbsum;
    const natoms = s.n;
    const cut = cutoff ?? 0;
    const vol = geom.lx * geom.ly * geom.lz;
    let g = this.gewaldUser > 0 ? this.gewaldUser : 0;
    if (!(this.gewaldUser > 0)) {
      if (this.termflag[TERM_COUL]) {
        if (accuracy <= 0.0) throw new StyleError('KSpace accuracy must be > 0');
        if (this.q2 === 0.0) throw new StyleError("Must use 'kspace_modify gewald' for uncharged system");
        g = (accuracy * Math.sqrt(natoms * cut * vol)) / (2.0 * this.q2);
        if (g >= 1.0) g = (1.35 - 0.15 * log(accuracy)) / cut;
        else g = Math.sqrt(-log(g)) / cut;
      } else if (this.termflag[TERM_DIPOLE] || this.termflag[TERM_DISP_GEOM] || this.termflag[TERM_DISP_ARITH]) {
        // the old method as the initial guess of the Newton solver
        g = (1.35 - 0.15 * log(accuracy)) / cut;
        const gNew = this.newtonSolve(g, cut, natoms, vol, this.termflag[TERM_DIPOLE] ? this.M2 : this.b2, accuracy);
        if (gNew > 0.0) g = gNew;
        else ctx.log('WARNING: Ewald/disp Newton solver failed, using old method to estimate g_ewald');
      }
    }
    this.gEwald = g;
    ctx.log(`  G vector = ${Number(g.toPrecision(8))},   accuracy = ${Number(accuracy.toPrecision(8))}`);
    // the coulomb g_ewald applies to dispersion unless it is explicitly set
    this.gEwald6 = this.gewaldUser6 !== 0 ? this.gewaldUser6 : g;
    this.view?.setEwald(this.gEwald, this.gEwald6);
    this.setup(s, geom, ctx);
  }

  /** KSpace::qsum_qsq (warn_nonneutral = 1: a warning, at init only). */
  private qsumQsq(s: SimState, _qqrd2e: number, _ctx: StyleContext | null): void {
    // in native atom order (KSpace::qsum_qsq sums the local atoms in their storage order)
    let qs = 0, q2 = 0;
    for (const i of nativeOrder(s)) { qs += s.q[i]; q2 += s.q[i] * s.q[i]; }
    this.qsum = qs;
    this.qsqsum = q2;
  }

  updateChargeSums(s: SimState): void {
    if (this.termflag[TERM_COUL]) this.qsumQsq(s, this.mumurd2e, null);
    this.natomsOriginal = s.n;
  }

  /** domain->h_inv (domain.cpp set_global_box). */
  private hInv(geom: Geometry): Float64Array {
    const h = [geom.lx, geom.ly, geom.lz, geom.yz, geom.xz, geom.xy];
    const hi = new Float64Array(6);
    hi[0] = 1.0 / h[0]; hi[1] = 1.0 / h[1]; hi[2] = 1.0 / h[2];
    hi[3] = -h[3] / (h[1] * h[2]);
    hi[4] = (h[3] * h[5] - h[1] * h[4]) / (h[0] * h[1] * h[2]);
    hi[5] = -h[5] / (h[0] * h[1]);
    return hi;
  }

  private static key(geom: Geometry): string {
    return `${geom.lx} ${geom.ly} ${geom.lz} ${geom.xy} ${geom.xz} ${geom.yz}`;
  }

  /** EwaldDisp::setup: volume, unitk, kmax from the rms estimate, the k-vector list and the coefficients. */
  private setup(s: SimState, geom: Geometry, ctx: StyleContext | null): void {
    this.volume = geom.lx * geom.ly * geom.lz * this.slabVolfactor;
    const unitk = this.unitk;
    unitk.set(this.hInv(geom));
    for (let k = 0; k < 6; k++) unitk[k] *= 2.0 * MY_PI;
    unitk[2] /= this.slabVolfactor;
    const accuracy = this.accuracyForce;
    if (accuracy >= 1) {
      this.kmax = 0;
      throw new StyleError('KSpace accuracy too low');
    }
    const natoms = s.n;
    const kdim = (prd: number): number => {
      let km = 1;
      let err = this.rms(km, prd, natoms);
      while (err > accuracy) {
        km++;
        err = this.rms(km, prd, natoms);
      }
      return km;
    };
    const kxmax = kdim(geom.lx), kymax = kdim(geom.ly), kzmax = kdim(geom.lz * this.slabVolfactor);
    this.kmax = Math.max(kxmax, kymax, kzmax);
    const gsqxmx = unitk[0] * unitk[0] * kxmax * kxmax;
    const gsqymx = unitk[1] * unitk[1] * kymax * kymax;
    const gsqzmx = unitk[2] * unitk[2] * kzmax * kzmax;
    this.gsqmx = Math.max(gsqxmx, gsqymx, gsqzmx) * 1.00001;
    this.reallocate();
    this.coeffs();
    this.initCoeffs(s);
    this.initSelf();
    ctx?.log(`  vectors: kmax = ${this.kmax}, kcount = ${this.kcount}`);
    this.setupKey = KSpaceEwaldDisp.key(geom);
  }

  /** EwaldDisp::rms: the rms force error of one dimension with km vectors. */
  private rms(km: number, prd: number, natoms: number): number {
    let value = 0.0;
    if (natoms === 0) natoms = 1;
    const g = this.gEwald, g2 = g * g;
    value += ((2.0 * this.q2 * g) / prd) * Math.sqrt(1.0 / (MY_PI * km * natoms)) * exp((-MY_PI * MY_PI * km * km) / (g2 * prd * prd));
    const g2_6 = this.gEwald6 * this.gEwald6;
    const g6 = g2_6 * g2_6 * g2_6;
    value += ((this.b2 * g6) / Math.sqrt(natoms * this.volume)) * Math.sqrt(prd / (2.0 * MY_PI * km)) * exp((-MY_PI * MY_PI * km * km) / (g2_6 * prd * prd));
    value += ((8.0 * MY_PI * this.M2) / this.volume) * g * Math.sqrt((2.0 * MY_PI * km * km * km) / (15.0 * natoms)) * exp(-(((MY_PI * km) / (g * prd)) * ((MY_PI * km) / (g * prd)))); // ewald_disp.cpp: pow(..., 2.0) is a product in the native binary (gcc folds a literal 2.0)
    return value;
  }

  /** EwaldDisp::reallocate: the half-space k vectors within gsqmx, x-major then y then z. */
  private reallocate(): void {
    const kmax = this.kmax, unitk = this.unitk;
    const h: number[] = [], kv: number[] = [];
    for (let ix = 0; ix <= kmax; ix++) {
      for (let iy = -kmax; iy <= kmax; iy++) {
        for (let iz = -kmax; iz <= kmax; iz++) {
          if (ix === 0 && iy === 0 && iz === 0) continue;
          if (ix === 0 && iy < 0) continue;
          if (ix === 0 && iy === 0 && iz < 0) continue;
          const h0 = unitk[0] * ix;
          const h1 = unitk[5] * ix + unitk[1] * iy;
          const h2 = unitk[4] * ix + unitk[3] * iy + unitk[2] * iz;
          if (h0 * h0 + h1 * h1 + h2 * h2 <= this.gsqmx) {
            h.push(h0, h1, h2);
            kv.push(ix + kmax, iy + kmax, iz + kmax);
          }
        }
      }
    }
    this.kcount = h.length / 3;
    this.hvec = Float64Array.from(h);
    this.kvec = Int32Array.from(kv);
    this.kenergy = new Float64Array(this.kcount * this.nterms);
    this.kvirial = new Float64Array(6 * this.kcount * this.nterms);
    this.sfac = new Float64Array(2 * this.kcount * this.nsums);
  }

  /** EwaldDisp::coeffs: energy and virial prefactors of each k vector and term. */
  private coeffs(): void {
    const eta2 = 0.25 / (this.gEwald * this.gEwald);
    const eta2_6 = 0.25 / (this.gEwald6 * this.gEwald6);
    const tf = this.termflag, ke = this.kenergy, kv = this.kvirial;
    let ne = 0, nv = 0;
    const put = (c1: number, c2: number, h: number[]): void => {
      ke[ne++] = c1;
      kv[nv++] = c1 - c2 * h[0] * h[0];
      kv[nv++] = c1 - c2 * h[1] * h[1];
      kv[nv++] = c1 - c2 * h[2] * h[2];
      kv[nv++] = -c2 * h[1] * h[0];
      kv[nv++] = -c2 * h[2] * h[0];
      kv[nv++] = -c2 * h[2] * h[1];
    };
    for (let k = 0; k < this.kcount; k++) {
      const h = [this.hvec[3 * k], this.hvec[3 * k + 1], this.hvec[3 * k + 2]];
      const h2 = h[0] * h[0] + h[1] * h[1] + h[2] * h[2];
      const bk2 = h2 * eta2;
      const expb2 = exp(-bk2);
      if (tf[TERM_COUL]) {
        const c1 = expb2 / h2;
        put(c1, (2.0 * c1 * (1.0 + bk2)) / h2, h);
      }
      if (tf[TERM_DISP_GEOM] || tf[TERM_DISP_ARITH]) {
        const bk2_6 = h2 * eta2_6;
        const expb2_6 = exp(-bk2_6);
        const b1 = Math.sqrt(bk2_6);
        const h1 = Math.sqrt(h2);
        const ce = MY_PIS * erfc(b1);
        const c1 = -h1 * h2 * (ce + ((0.5 / bk2_6 - 1.0) * expb2_6) / b1);
        const c2 = 3.0 * h1 * (ce - expb2_6 / b1);
        put(c1, c2, h);
      }
      if (tf[TERM_DIPOLE]) {
        const c1 = expb2 / h2;
        put(c1, (2.0 * c1 * (1.0 + bk2)) / h2, h);
      }
    }
  }

  /** EwaldDisp::init_coeffs: the per-type dispersion coefficients from the pair style (B = lj4 or epsilon, sigma). */
  private initCoeffs(s: SimState): void {
    const n = s.ntypes, v = this.view;
    if (!v) return;
    if (this.termflag[TERM_DISP_GEOM]) {
      this.B = new Float64Array(n + 1);
      for (let i = 1; i <= n; i++) {
        // pair_lj_long_coul_long.cpp init_one: lj4 = 4.0 * epsilon * pow(sigma, 6.0)
        const lj4 = 4.0 * v.epsilonOf(i) * pow(v.sigmaOf(i), 6.0);
        this.B[i] = Math.sqrt(Math.abs(lj4));
      }
    }
    if (this.termflag[TERM_DISP_ARITH]) {
      this.B = new Float64Array(7 * n + 7);
      const c = [1.0, Math.sqrt(6.0), Math.sqrt(15.0), Math.sqrt(20.0), Math.sqrt(15.0), Math.sqrt(6.0), 1.0];
      for (let i = 1; i <= n; i++) {
        const epsI = Math.sqrt(v.epsilonOf(i));
        const sigmaI = v.sigmaOf(i);
        let sigmaN = 1.0;
        for (let j = 0; j < 7; j++) {
          this.B[7 * i + j] = sigmaN * epsI * c[j];
          sigmaN *= sigmaI;
        }
      }
    }
  }

  /** EwaldDisp::init_coeff_sums (once per init). */
  private initCoeffSums(s: SimState): void {
    const X = this.coeffSumX, X2 = this.coeffSumX2;
    X.fill(0); X2.fill(0);
    const type = s.type, B = this.B;
    // in native atom order: the Newton solve of g_ewald follows the last bits of these sums
    const ord = nativeOrder(s);
    if (this.termflag[TERM_DISP_GEOM]) {
      for (const i of ord) { X[1] += B[type[i]]; X2[1] += B[type[i]] * B[type[i]]; }
    }
    if (this.termflag[TERM_DISP_ARITH]) {
      for (const i of ord) {
        const o = 7 * type[i];
        X2[2] += B[o] * B[o + 6];
        for (let k = 2; k < 9; k++) X[k] += B[o + k - 2];
      }
    }
    if (this.termflag[TERM_DIPOLE] && s.mu) {
      for (const i of ord) X2[9] += s.mu[4 * i + 3] * s.mu[4 * i + 3];
    }
  }

  /** EwaldDisp::init_self: the self energy and virial of each term. */
  private initSelf(): void {
    const g1 = this.gEwald, g2 = g1 * g1, g3 = g1 * g2;
    const g3_6 = this.gEwald6 * this.gEwald6 * this.gEwald6;
    const qscale = this.mumurd2e;
    const X = this.coeffSumX, X2 = this.coeffSumX2, V = this.volume;
    const es = this.energySelf, vs = this.virialSelf;
    es.fill(0); vs.fill(0);
    if (this.termflag[TERM_COUL]) {
      vs[TERM_COUL] = ((-0.5 * MY_PI * qscale) / (g2 * V)) * this.qsum * this.qsum;
      es[TERM_COUL] = (this.qsqsum * qscale * g1) / MY_PIS - vs[TERM_COUL];
    }
    if (this.termflag[TERM_DISP_GEOM]) {
      vs[TERM_DISP_GEOM] = ((MY_PI * MY_PIS * g3_6) / (6.0 * V)) * X[1] * X[1];
      es[TERM_DISP_GEOM] = (-X2[1] * g3_6 * g3_6) / 12.0 + vs[TERM_DISP_GEOM];
    }
    if (this.termflag[TERM_DISP_ARITH]) {
      vs[TERM_DISP_ARITH] = ((MY_PI * MY_PIS * g3_6) / (48.0 * V)) * (X[2] * X[8] + X[3] * X[7] + X[4] * X[6] + 0.5 * X[5] * X[5]);
      es[TERM_DISP_ARITH] = (-X2[2] * g3_6 * g3_6) / 3.0 + vs[TERM_DISP_ARITH];
    }
    if (this.termflag[TERM_DIPOLE]) {
      vs[TERM_DIPOLE] = 0;
      es[TERM_DIPOLE] = (X2[9] * this.mumurd2e * 2.0 * g3) / 3.0 / MY_PIS - vs[TERM_DIPOLE];
    }
  }

  /** EwaldDisp::init_self_peratom. */
  private selfPeratom(s: SimState): { e: Float64Array; v: Float64Array } {
    const n = s.n;
    const e = new Float64Array(NTERMS * n), v = new Float64Array(NTERMS * n);
    const g1 = this.gEwald, g2 = g1 * g1, g3 = g1 * g2;
    const g3_6 = this.gEwald6 * this.gEwald6 * this.gEwald6;
    const qscale = this.mumurd2e, V = this.volume, X = this.coeffSumX, B = this.B;
    if (this.termflag[TERM_COUL]) {
      const ce = (qscale * g1) / MY_PIS, cv = (-0.5 * MY_PI * qscale) / (g2 * V);
      for (let i = 0; i < n; i++) {
        v[NTERMS * i + TERM_COUL] = cv * s.q[i] * this.qsum;
        e[NTERMS * i + TERM_COUL] = ce * s.q[i] * s.q[i] - v[NTERMS * i + TERM_COUL];
      }
    }
    if (this.termflag[TERM_DISP_GEOM]) {
      const ce = (-g3_6 * g3_6) / 12.0, cv = (MY_PI * MY_PIS * g3_6) / (6.0 * V);
      for (let i = 0; i < n; i++) {
        const b = B[s.type[i]];
        v[NTERMS * i + TERM_DISP_GEOM] = cv * b * X[1];
        e[NTERMS * i + TERM_DISP_GEOM] = ce * b * b + v[NTERMS * i + TERM_DISP_GEOM];
      }
    }
    if (this.termflag[TERM_DISP_ARITH]) {
      const ce = (-g3_6 * g3_6) / 3.0, cv = (0.5 * MY_PI * MY_PIS * g3_6) / (48.0 * V);
      for (let i = 0; i < n; i++) {
        const o = 7 * s.type[i];
        for (let k = 2; k < 9; k++) v[NTERMS * i + TERM_DISP_ARITH] += cv * X[k] * B[o + 8 - k];
        e[NTERMS * i + TERM_DISP_ARITH] = ce * B[o] * B[o + 6] + v[NTERMS * i + TERM_DISP_ARITH];
      }
    }
    if (this.termflag[TERM_DIPOLE] && s.mu) {
      const ce = (this.mumurd2e * 2.0 * g3) / 3.0 / MY_PIS;
      for (let i = 0; i < n; i++) {
        v[NTERMS * i + TERM_DIPOLE] = 0.0;
        e[NTERMS * i + TERM_DIPOLE] = ce * s.mu[4 * i + 3] * s.mu[4 * i + 3];
      }
    }
    return { e, v };
  }

  compute(kc: KSpaceCompute): void {
    if (!this.kmax) return;
    const s = kc.s, geom = kc.geom;
    if (EwaldKeyChanged(this.setupKey, geom)) this.setup(s, geom, null);
    const eflagAtom = kc.eatom !== null, vflagAtom = kc.vatom !== null;
    if (this.termflag[TERM_DIPOLE] && (!s.mu || !kc.torque)) throw new StyleError('kspace_style ewald/disp with dipoles needs atom attributes mu and torque');
    const self = eflagAtom || vflagAtom ? this.selfPeratom(s) : null;
    this.eikDotR(s);
    this.computeForce(s, kc.f, kc.torque ?? null);
    if (s.n !== this.natomsOriginal) {
      if (this.termflag[TERM_COUL]) this.qsumQsq(s, this.mumurd2e, null);
      this.natomsOriginal = s.n;
    }
    let energy = this.computeEnergy();
    if (kc.eatom) this.computeEnergyPeratom(s, kc.eatom, self!.e);
    const virial = this.computeVirial();
    if (this.termflag[TERM_DIPOLE]) this.computeVirialDipole(s, virial, kc.vatom);
    if (kc.vatom) this.computeVirialPeratom(s, kc.vatom, self!.v);
    if (this.slab > 1) energy += this.slabcorr(s, geom, kc.f, kc.torque ?? null, kc.eatom);
    kc.acc.elong += energy;
    for (let c = 0; c < 6; c++) kc.acc.vlong[c] += virial[c];
  }

  /** EwaldDisp::eik_dot_r: e^(i j k0.r) per atom and dimension, and the structure factors of each channel. */
  private eikDotR(s: SimState): void {
    const kmax = this.kmax, nz = 2 * kmax + 1, n = s.n, x = s.x, unitk = this.unitk;
    if (this.ekr.length < 6 * nz * n) this.ekr = new Float64Array(6 * nz * n);
    const z = this.ekr, sf = this.sfac, tf = this.termflag, B = this.B, kv = this.kvec, hv = this.hvec;
    sf.fill(0);
    // the structure factors are summed in native atom order
    for (const i of nativeOrder(s)) {
      const base = 6 * nz * i;
      // z[kmax+j] holds e^(i j k0 . r) for x, y, z (6 doubles: x.re x.im y.re y.im z.re z.im)
      const o0 = base + 6 * kmax;
      z[o0] = 1; z[o0 + 1] = 0; z[o0 + 2] = 1; z[o0 + 3] = 0; z[o0 + 4] = 1; z[o0 + 5] = 0;
      let ax: number, ay: number, az: number;
      if (this.triclinic) {
        ax = unitk[0] * x[3 * i] + unitk[5] * x[3 * i + 1] + unitk[4] * x[3 * i + 2];
        ay = unitk[1] * x[3 * i + 1] + unitk[3] * x[3 * i + 2];
        az = x[3 * i + 2] * unitk[2];
      } else {
        ax = x[3 * i] * unitk[0];
        ay = x[3 * i + 1] * unitk[1];
        az = x[3 * i + 2] * unitk[2];
      }
      const z1xr = cos(ax), z1xi = sin(ax), z1yr = cos(ay), z1yi = sin(ay), z1zr = cos(az), z1zi = sin(az);
      for (let j = 1; j <= kmax; j++) {
        const p = base + 6 * (kmax + j), q = base + 6 * (kmax + j - 1), m = base + 6 * (kmax - j);
        // c_rmult(z[kmax+j].x, z[kmax+j-1].x, z1.x)
        let tr = z[q], ti = z[q + 1];
        z[p] = tr * z1xr - ti * z1xi; z[p + 1] = tr * z1xi + ti * z1xr;
        tr = z[q + 2]; ti = z[q + 3];
        z[p + 2] = tr * z1yr - ti * z1yi; z[p + 3] = tr * z1yi + ti * z1yr;
        z[m + 2] = z[p + 2]; z[m + 3] = -z[p + 3];
        tr = z[q + 4]; ti = z[q + 5];
        z[p + 4] = tr * z1zr - ti * z1zi; z[p + 5] = tr * z1zi + ti * z1zr;
        z[m + 4] = z[p + 4]; z[m + 5] = -z[p + 5];
      }
      const qi = tf[TERM_COUL] ? s.q[i] : 0.0;
      const bi = tf[TERM_DISP_GEOM] ? B[s.type[i]] : 0.0;
      const ci = tf[TERM_DISP_ARITH] ? 7 * s.type[i] : 0;
      const mu = s.mu;
      let ch = 0;
      let kxold = -1, kyold = -1;
      let cxr = 0, cxi = 0, zxyr = 0, zxyi = 0;
      for (let k = 0; k < this.kcount; k++) {
        const kx = kv[3 * k], ky = kv[3 * k + 1], kz = kv[3 * k + 2];
        if (ky !== kyold) {
          if (kx !== kxold) { cxr = z[base + 6 * kx]; cxi = z[base + 6 * kx + 1]; kxold = kx; }
          // c_rmult(zxy, z[ky].y, cx)
          const yr = z[base + 6 * ky + 2], yi = z[base + 6 * ky + 3];
          zxyr = yr * cxr - yi * cxi; zxyi = yr * cxi + yi * cxr;
          kyold = ky;
        }
        // c_rmult(zxyz, z[kz].z, zxy)
        const zr = z[base + 6 * kz + 4], zi = z[base + 6 * kz + 5];
        const re = zr * zxyr - zi * zxyi, im = zr * zxyi + zi * zxyr;
        if (tf[TERM_COUL]) { sf[ch++] += re * qi; sf[ch++] += im * qi; }
        if (tf[TERM_DISP_GEOM]) { sf[ch++] += re * bi; sf[ch++] += im * bi; }
        if (tf[TERM_DISP_ARITH]) {
          for (let m = 0; m < 7; m++) { sf[ch++] += re * B[ci + m]; sf[ch++] += im * B[ci + m]; }
        }
        if (tf[TERM_DIPOLE]) {
          const muk = mu![4 * i] * hv[3 * k] + mu![4 * i + 1] * hv[3 * k + 1] + mu![4 * i + 2] * hv[3 * k + 2];
          sf[ch++] += re * muk; sf[ch++] += im * muk;
        }
      }
    }
  }

  /** zc = z[kz].z * conj(zxy) at k vector k of atom i, with the cached zx and zxy (see eikDotR). */
  private forEachK(i: number, fn: (k: number, zr: number, zi: number) => void): void {
    const kmax = this.kmax, nz = 2 * kmax + 1, z = this.ekr, kv = this.kvec;
    const base = 6 * nz * i;
    let kxold = -1, kyold = -1;
    let zxr = 0, zxi = 0, zxyr = 0, zxyi = 0;
    for (let k = 0; k < this.kcount; k++) {
      const kx = kv[3 * k], ky = kv[3 * k + 1], kz = kv[3 * k + 2];
      if (ky !== kyold) {
        if (kx !== kxold) { zxr = z[base + 6 * kx]; zxi = z[base + 6 * kx + 1]; kxold = kx; }
        const yr = z[base + 6 * ky + 2], yi = z[base + 6 * ky + 3];
        zxyr = yr * zxr - yi * zxi; zxyi = yr * zxi + yi * zxr;
        kyold = ky;
      }
      // c_crmult(zc, z[kz].z, zxy): zc = z * conj(zxy)
      const tr = z[base + 6 * kz + 4], ti = z[base + 6 * kz + 5];
      fn(k, tr * zxyr - ti * zxyi, -tr * zxyi - ti * zxyr);
    }
  }

  /** EwaldDisp::compute_force (forces, and torques of dipoles). */
  private computeForce(s: SimState, f: Float64Array, tq: Float64Array | null): void {
    const tf = this.termflag, ke = this.kenergy, sf = this.sfac, hv = this.hvec, B = this.B, mu = s.mu;
    const qscale = this.mumurd2e, V = this.volume;
    const c = [(8.0 * MY_PI * qscale) / V, (2.0 * MY_PI * MY_PIS) / (12.0 * V), (2.0 * MY_PI * MY_PIS) / (192.0 * V), (8.0 * MY_PI * this.mumurd2e) / V];
    const eksum = new Float64Array(3 * MAX_NSUMS);
    const mui = [0.0, 0.0, 0.0];
    for (let i = 0; i < s.n; i++) {
      let n = 0, ne = 0, ncoul = 0;
      eksum.fill(0);
      if (tf[TERM_DIPOLE]) { mui[0] = c[TERM_DIPOLE] * mu![4 * i]; mui[1] = c[TERM_DIPOLE] * mu![4 * i + 1]; mui[2] = c[TERM_DIPOLE] * mu![4 * i + 2]; }
      this.forEachK(i, (k, zr, zi) => {
        const hx = hv[3 * k], hy = hv[3 * k + 1], hz = hv[3 * k + 2];
        if (tf[TERM_COUL]) {
          const im = ke[ne++] * (zi * sf[2 * n] + sf[2 * n + 1] * zr);
          if (tf[TERM_DIPOLE]) ncoul = n;
          n++;
          eksum[0] += hx * im; eksum[1] += hy * im; eksum[2] += hz * im;
        }
        if (tf[TERM_DISP_GEOM]) {
          const im = ke[ne++] * (zi * sf[2 * n] + sf[2 * n + 1] * zr);
          n++;
          eksum[3] += hx * im; eksum[4] += hy * im; eksum[5] += hz * im;
        }
        if (tf[TERM_DISP_ARITH]) {
          const ck = ke[ne++];
          for (let m = 2; m < 9; m++) {
            const im = ck * (zi * sf[2 * n] + sf[2 * n + 1] * zr);
            n++;
            eksum[3 * m] += hx * im; eksum[3 * m + 1] += hy * im; eksum[3 * m + 2] += hz * im;
          }
        }
        if (tf[TERM_DIPOLE]) {
          const muk = mui[0] * hx + mui[1] * hy + mui[2] * hz;
          let im = ke[ne] * (zi * sf[2 * n] + sf[2 * n + 1] * zr) * muk;
          let im2 = ke[ne] * (zr * sf[2 * n] - sf[2 * n + 1] * zi);
          eksum[27] += hx * im; eksum[28] += hy * im; eksum[29] += hz * im;
          tq![3 * i] += -mui[1] * hz * im2 + mui[2] * hy * im2;
          tq![3 * i + 1] += -mui[2] * hx * im2 + mui[0] * hz * im2;
          tq![3 * i + 2] += -mui[0] * hy * im2 + mui[1] * hx * im2;
          if (tf[TERM_COUL]) {
            const qi = s.q[i] * c[TERM_COUL];
            im = -ke[ne] * (zr * sf[2 * ncoul] - sf[2 * ncoul + 1] * zi) * muk;
            im += ke[ne] * (zr * sf[2 * n] - sf[2 * n + 1] * zi) * qi;
            eksum[27] += hx * im; eksum[28] += hy * im; eksum[29] += hz * im;
            im2 = ke[ne] * (zr * sf[2 * ncoul + 1] + sf[2 * ncoul] * zi);
            im2 += -ke[ne] * (zr * sf[2 * n + 1] - sf[2 * n + 1] * zr);
            tq![3 * i] += -mui[1] * hz * im2 + mui[2] * hy * im2;
            tq![3 * i + 1] += -mui[2] * hx * im2 + mui[0] * hz * im2;
            tq![3 * i + 2] += -mui[0] * hy * im2 + mui[1] * hx * im2;
          }
          n++;
          ne++;
        }
      });
      if (tf[TERM_COUL]) {
        const qi = s.q[i] * c[TERM_COUL];
        f[3 * i] -= eksum[0] * qi; f[3 * i + 1] -= eksum[1] * qi; f[3 * i + 2] -= eksum[2] * qi;
      }
      if (tf[TERM_DISP_GEOM]) {
        const bi = B[s.type[i]] * c[TERM_DISP_GEOM];
        f[3 * i] -= eksum[3] * bi; f[3 * i + 1] -= eksum[4] * bi; f[3 * i + 2] -= eksum[5] * bi;
      }
      if (tf[TERM_DISP_ARITH]) {
        const o = 7 * s.type[i];
        for (let m = 2; m < 9; m++) {
          const c2 = B[o + 8 - m] * c[TERM_DISP_ARITH];
          f[3 * i] -= eksum[3 * m] * c2; f[3 * i + 1] -= eksum[3 * m + 1] * c2; f[3 * i + 2] -= eksum[3 * m + 2] * c2;
        }
      }
      if (tf[TERM_DIPOLE]) {
        f[3 * i] -= eksum[27]; f[3 * i + 1] -= eksum[28]; f[3 * i + 2] -= eksum[29];
      }
    }
  }

  /** The per-term energy constants (compute_energy, compute_energy_peratom, compute_virial). */
  private energyConsts(): number[] {
    const V = this.volume, qscale = this.mumurd2e;
    return [(4.0 * MY_PI * qscale) / V, (2.0 * MY_PI * MY_PIS) / (24.0 * V), (2.0 * MY_PI * MY_PIS) / (192.0 * V), (4.0 * MY_PI * this.mumurd2e) / V];
  }

  /** EwaldDisp::compute_energy. */
  private computeEnergy(): number {
    const tf = this.termflag, ke = this.kenergy, sf = this.sfac, c = this.energyConsts();
    const esum = [0, 0, 0, 0];
    let n = 0, ne = 0, ncoul = 0;
    for (let k = 0; k < this.kcount; k++) {
      if (tf[TERM_COUL]) {
        esum[TERM_COUL] += ke[ne++] * (sf[2 * n] * sf[2 * n] + sf[2 * n + 1] * sf[2 * n + 1]);
        if (tf[TERM_DIPOLE]) ncoul = n;
        n++;
      }
      if (tf[TERM_DISP_GEOM]) {
        esum[TERM_DISP_GEOM] += ke[ne++] * (sf[2 * n] * sf[2 * n] + sf[2 * n + 1] * sf[2 * n + 1]);
        n++;
      }
      if (tf[TERM_DISP_ARITH]) {
        const r = this.arithPair(n);
        n += 7;
        esum[TERM_DISP_ARITH] += ke[ne++] * r;
      }
      if (tf[TERM_DIPOLE]) {
        esum[TERM_DIPOLE] += ke[ne] * (sf[2 * n] * sf[2 * n] + sf[2 * n + 1] * sf[2 * n + 1]);
        if (tf[TERM_COUL]) esum[TERM_DIPOLE] += ke[ne] * 2.0 * (sf[2 * n] * sf[2 * ncoul + 1] - sf[2 * n + 1] * sf[2 * ncoul]);
        ne++;
        n++;
      }
    }
    let energy = 0.0;
    for (let t = 0; t < NTERMS; t++) energy += c[t] * esum[t] - this.energySelf[t];
    return energy;
  }

  /** The arithmetic-mixing product of the seven structure factors at channel n (compute_energy, compute_virial). */
  private arithPair(n: number): number {
    const sf = this.sfac;
    const d = (a: number, b: number): number => sf[2 * (n + a)] * sf[2 * (n + b)] + sf[2 * (n + a) + 1] * sf[2 * (n + b) + 1];
    return d(0, 6) + d(1, 5) + d(2, 4) + 0.5 * d(3, 3);
  }

  /** EwaldDisp::compute_energy_peratom. */
  private computeEnergyPeratom(s: SimState, eatom: Float64Array, eself: Float64Array): void {
    const tf = this.termflag, ke = this.kenergy, sf = this.sfac, hv = this.hvec, B = this.B, mu = s.mu, c = this.energyConsts();
    const esum = new Float64Array(MAX_NSUMS);
    const mui = [0.0, 0.0, 0.0];
    for (let i = 0; i < s.n; i++) {
      let n = 0, ne = 0, ncoul = 0;
      esum.fill(0);
      if (tf[TERM_DIPOLE]) { mui[0] = c[TERM_DIPOLE] * mu![4 * i]; mui[1] = c[TERM_DIPOLE] * mu![4 * i + 1]; mui[2] = c[TERM_DIPOLE] * mu![4 * i + 2]; }
      this.forEachK(i, (k, zr, zi) => {
        if (tf[TERM_COUL]) {
          esum[0] += ke[ne++] * (sf[2 * n] * zr - sf[2 * n + 1] * zi);
          if (tf[TERM_DIPOLE]) ncoul = n;
          n++;
        }
        if (tf[TERM_DISP_GEOM]) { esum[1] += ke[ne++] * (sf[2 * n] * zr - sf[2 * n + 1] * zi); n++; }
        if (tf[TERM_DISP_ARITH]) {
          const ck = ke[ne++];
          for (let m = 2; m < 9; m++) { esum[m] += ck * (sf[2 * n] * zr - sf[2 * n + 1] * zi); n++; }
        }
        if (tf[TERM_DIPOLE]) {
          const muk = mui[0] * hv[3 * k] + mui[1] * hv[3 * k + 1] + mui[2] * hv[3 * k + 2];
          esum[9] += ke[ne] * (sf[2 * n] * zr - sf[2 * n + 1] * zi) * muk;
          if (tf[TERM_COUL]) {
            const qj = s.q[i] * c[TERM_COUL];
            esum[9] += ke[ne] * (sf[2 * ncoul + 1] * zr + sf[2 * ncoul] * zi) * muk;
            esum[9] -= ke[ne] * (sf[2 * n] * zi + sf[2 * n + 1] * zr) * qj;
          }
          n++;
          ne++;
        }
      });
      if (tf[TERM_COUL]) eatom[i] += esum[0] * (s.q[i] * c[TERM_COUL]) - eself[NTERMS * i + TERM_COUL];
      if (tf[TERM_DISP_GEOM]) eatom[i] += esum[1] * (B[s.type[i]] * c[TERM_DISP_GEOM]) - eself[NTERMS * i + TERM_DISP_GEOM];
      if (tf[TERM_DISP_ARITH]) {
        const o = 7 * s.type[i];
        for (let m = 2; m < 9; m++) eatom[i] += 0.5 * esum[m] * (B[o + 8 - m] * c[TERM_DISP_ARITH]);
        eatom[i] -= eself[NTERMS * i + TERM_DISP_ARITH];
      }
      if (tf[TERM_DIPOLE]) eatom[i] += esum[9] - eself[NTERMS * i + TERM_DIPOLE];
    }
  }

  /** EwaldDisp::compute_virial. */
  private computeVirial(): Float64Array {
    const tf = this.termflag, kv = this.kvirial, sf = this.sfac, c = this.energyConsts();
    const vsum = new Float64Array(6 * NTERMS);
    let n = 0, nv = 0, ncoul = 0;
    for (let k = 0; k < this.kcount; k++) {
      if (tf[TERM_COUL]) {
        const r = sf[2 * n] * sf[2 * n] + sf[2 * n + 1] * sf[2 * n + 1];
        if (tf[TERM_DIPOLE]) ncoul = n;
        n++;
        for (let m = 0; m < 6; m++) vsum[6 * TERM_COUL + m] += kv[nv + m] * r;
        nv += 6;
      }
      if (tf[TERM_DISP_GEOM]) {
        const r = sf[2 * n] * sf[2 * n] + sf[2 * n + 1] * sf[2 * n + 1];
        n++;
        for (let m = 0; m < 6; m++) vsum[6 * TERM_DISP_GEOM + m] += kv[nv + m] * r;
        nv += 6;
      }
      if (tf[TERM_DISP_ARITH]) {
        const r = this.arithPair(n);
        n += 7;
        for (let m = 0; m < 6; m++) vsum[6 * TERM_DISP_ARITH + m] += kv[nv + m] * r;
        nv += 6;
      }
      if (tf[TERM_DIPOLE]) {
        const r = sf[2 * n] * sf[2 * n] + sf[2 * n + 1] * sf[2 * n + 1];
        for (let m = 0; m < 6; m++) vsum[6 * TERM_DIPOLE + m] += kv[nv + m] * r;
        if (tf[TERM_COUL]) {
          const r2 = 2.0 * (sf[2 * n] * sf[2 * ncoul + 1] - sf[2 * n + 1] * sf[2 * ncoul]);
          for (let m = 0; m < 6; m++) vsum[6 * TERM_DIPOLE + m] += kv[nv + m] * r2;
        }
        nv += 6;
        n++;
      }
    }
    const virial = new Float64Array(6);
    for (let t = 0; t < NTERMS; t++) {
      if (!tf[t]) continue;
      for (let m = 0; m < 6; m++) vsum[6 * t + m] *= c[t];
      for (let m = 0; m < 6; m++) virial[m] += vsum[6 * t + m];
      for (let m = 0; m < 3; m++) virial[m] -= this.virialSelf[t];
    }
    return virial;
  }

  /** EwaldDisp::compute_virial_dipole: the extra virial of the dipole torques. */
  private computeVirialDipole(s: SimState, virial: Float64Array, vatom: Float64Array | null): void {
    const tf = this.termflag, ke = this.kenergy, sf = this.sfac, hv = this.hvec, mu = s.mu!;
    const V = this.volume;
    const cdip = (8.0 * MY_PI * this.mumurd2e) / V;
    const ndip = tf[TERM_COUL] + tf[TERM_DISP_GEOM] + 7 * tf[TERM_DISP_ARITH];
    const nedip = tf[TERM_COUL] + tf[TERM_DISP_GEOM] + tf[TERM_DISP_ARITH];
    const total = new Float64Array(6);
    const vsum = new Float64Array(6);
    for (let i = 0; i < s.n; i++) {
      vsum.fill(0);
      const m0 = cdip * mu[4 * i], m1 = cdip * mu[4 * i + 1], m2 = cdip * mu[4 * i + 2];
      this.forEachK(i, (k, zr, zi) => {
        const sd = 2 * (k * this.nsums + ndip);
        const kk = ke[k * this.nterms + nedip];
        let im = kk * (zr * sf[sd] - sf[sd + 1] * zi);
        if (tf[TERM_COUL]) {
          const sc = 2 * (k * this.nsums);
          im += kk * (zi * sf[sc] + sf[sc + 1] * zr);
        }
        const hx = hv[3 * k], hy = hv[3 * k + 1], hz = hv[3 * k + 2];
        vsum[0] -= m0 * hx * im; vsum[1] -= m1 * hy * im; vsum[2] -= m2 * hz * im;
        vsum[3] -= m0 * hy * im; vsum[4] -= m0 * hz * im; vsum[5] -= m1 * hz * im;
      });
      for (let m = 0; m < 6; m++) total[m] -= vsum[m];
      if (vatom) for (let m = 0; m < 6; m++) vatom[6 * i + m] -= vsum[m];
    }
    for (let m = 0; m < 6; m++) virial[m] += total[m];
  }

  /** EwaldDisp::compute_virial_peratom. */
  private computeVirialPeratom(s: SimState, vatom: Float64Array, vself: Float64Array): void {
    const tf = this.termflag, kv = this.kvirial, sf = this.sfac, hv = this.hvec, B = this.B, mu = s.mu, c = this.energyConsts();
    const vsum = new Float64Array(6 * MAX_NSUMS);
    const mui = [0.0, 0.0, 0.0];
    for (let i = 0; i < s.n; i++) {
      let n = 0, nv = 0, ncoul = 0;
      vsum.fill(0);
      if (tf[TERM_DIPOLE]) { mui[0] = c[TERM_DIPOLE] * mu![4 * i]; mui[1] = c[TERM_DIPOLE] * mu![4 * i + 1]; mui[2] = c[TERM_DIPOLE] * mu![4 * i + 2]; }
      this.forEachK(i, (k, zr, zi) => {
        if (tf[TERM_COUL]) {
          if (tf[TERM_DIPOLE]) ncoul = n;
          const r = sf[2 * n] * zr - sf[2 * n + 1] * zi;
          n++;
          for (let m = 0; m < 6; m++) vsum[m] += kv[nv + m] * r;
          nv += 6;
        }
        if (tf[TERM_DISP_GEOM]) {
          const r = sf[2 * n] * zr - sf[2 * n + 1] * zi;
          n++;
          for (let m = 0; m < 6; m++) vsum[6 + m] += kv[nv + m] * r;
          nv += 6;
        }
        if (tf[TERM_DISP_ARITH]) {
          for (let t = 2; t < 9; t++) {
            const r = sf[2 * n] * zr - sf[2 * n + 1] * zi;
            n++;
            for (let m = 0; m < 6; m++) vsum[6 * t + m] += kv[nv + m] * r;
          }
          nv += 6;
        }
        if (tf[TERM_DIPOLE]) {
          const muk = mui[0] * hv[3 * k] + mui[1] * hv[3 * k + 1] + mui[2] * hv[3 * k + 2];
          let r = (sf[2 * n] * zr - sf[2 * n + 1] * zi) * muk;
          for (let m = 0; m < 6; m++) vsum[54 + m] += kv[nv + m] * r;
          if (tf[TERM_COUL]) {
            const qj = s.q[i] * c[TERM_COUL];
            r = (sf[2 * ncoul + 1] * zr + sf[2 * ncoul] * zi) * muk;
            r += -(sf[2 * n] * zi + sf[2 * n + 1] * zr) * qj;
            for (let m = 0; m < 6; m++) vsum[54 + m] += kv[nv + m] * r;
          }
          nv += 6;
          n++;
        }
      });
      if (tf[TERM_COUL]) { const qi = s.q[i] * c[TERM_COUL]; for (let m = 0; m < 6; m++) vatom[6 * i + m] += vsum[m] * qi; }
      if (tf[TERM_DISP_GEOM]) { const bi = B[s.type[i]] * c[TERM_DISP_GEOM]; for (let m = 0; m < 6; m++) vatom[6 * i + m] += vsum[6 + m] * bi; }
      if (tf[TERM_DISP_ARITH]) {
        const o = 7 * s.type[i];
        for (let t = 2; t < 9; t++) {
          const c2 = B[o + 8 - t] * c[TERM_DISP_ARITH];
          for (let m = 0; m < 6; m++) vatom[6 * i + m] += 0.5 * vsum[6 * t + m] * c2;
        }
      }
      if (tf[TERM_DIPOLE]) for (let m = 0; m < 6; m++) vatom[6 * i + m] += vsum[54 + m];
      for (let t = 0; t < NTERMS; t++) if (tf[t]) for (let m = 0; m < 3; m++) vatom[6 * i + m] -= vself[NTERMS * i + t];
    }
  }

  /** EwaldDisp::slabcorr (EW3DC, with the non-neutral extension); returns the energy correction. */
  private slabcorr(s: SimState, geom: Geometry, f: Float64Array, torque: Float64Array | null, eatom: Float64Array | null): number {
    const q = s.q, x = s.x, n = s.n;
    const zprdSlab = geom.lz * this.slab;
    let dipole = 0.0;
    for (let i = 0; i < n; i++) dipole += q[i] * x[3 * i + 2];
    const dip = this.termflag[TERM_DIPOLE] && s.mu;
    if (dip) for (let i = 0; i < n; i++) dipole += s.mu![4 * i + 2];
    const dipoleAll = dipole;
    let dipoleR2 = 0.0;
    if (eatom || Math.abs(this.qsum) > SMALL) {
      if (dip) throw new StyleError('Cannot (yet) use kspace slab correction with long-range dipoles and non-neutral systems or per-atom energy');
      for (let i = 0; i < n; i++) dipoleR2 += q[i] * x[3 * i + 2] * x[3 * i + 2];
    }
    const qsum = this.qsum;
    const eSlab = (MY_2PI * (dipoleAll * dipoleAll - qsum * dipoleR2 - (qsum * qsum * zprdSlab * zprdSlab) / 12.0)) / this.volume;
    const qscale = this.mumurd2e;
    if (eatom) {
      const efact = (qscale * MY_2PI) / this.volume;
      for (let i = 0; i < n; i++) {
        eatom[i] += efact * q[i] * (x[3 * i + 2] * dipoleAll - 0.5 * (dipoleR2 + qsum * x[3 * i + 2] * x[3 * i + 2]) - (qsum * zprdSlab * zprdSlab) / 12.0);
      }
    }
    const ffact = qscale * ((-4.0 * MY_PI) / this.volume);
    for (let i = 0; i < n; i++) f[3 * i + 2] += ffact * q[i] * (dipoleAll - qsum * x[3 * i + 2]);
    if (dip && torque) {
      for (let i = 0; i < n; i++) {
        torque[3 * i] += ffact * dipoleAll * s.mu![4 * i + 1];
        torque[3 * i + 1] += -ffact * dipoleAll * s.mu![4 * i];
      }
    }
    return qscale * eSlab;
  }

  /** EwaldDisp::NewtonSolve with f and derivf (the real-space error minus the accuracy). */
  private newtonSolve(x0: number, Rc: number, natoms: number, vol: number, b2: number, accuracy: number): number {
    const maxit = 10000, tol = 0.00001;
    const fn = (x: number): number => this.fErr(x, Rc, natoms, vol, b2, accuracy);
    const derivf = (x: number): number => { const h = 0.000001; return (fn(x + h) - fn(x)) / h; };
    let x = x0;
    for (let i = 0; i < maxit; i++) {
      const dfx = derivf(x);
      if (dfx === 0.0 || Number.isNaN(dfx)) return -1;
      let dx = fn(x) / dfx;
      while (x - dx <= 0.0) dx *= 0.5;
      x = x - dx;
      if (Math.abs(dx) < tol) return x;
      if (Number.isNaN(x)) return -1;
    }
    return -1;
  }

  private fErr(x: number, Rc: number, natoms: number, vol: number, b2: number, accuracy: number): number {
    const a = Rc * x;
    let f = 0.0;
    if (this.termflag[TERM_DIPOLE]) {
      const rg2 = a * a, rg4 = rg2 * rg2, rg6 = rg4 * rg2;
      const Cc = 4.0 * rg4 + 6.0 * rg2 + 3.0;
      const Dc = 8.0 * rg6 + 20.0 * rg4 + 30.0 * rg2 + 15.0;
      f = (b2 / Math.sqrt(vol * powint(x, 4) * powint(Rc, 9) * natoms)) * Math.sqrt((13.0 / 6.0) * Cc * Cc + (2.0 / 15.0) * Dc * Dc - (13.0 / 15.0) * Cc * Dc) * exp(-rg2) - accuracy;
    } else if (this.termflag[TERM_DISP_GEOM] || this.termflag[TERM_DISP_ARITH]) {
      const a2 = a * a;
      f = ((b2 * Math.sqrt(MY_PI) * powint(x, 5)) / Math.sqrt(vol * Rc * natoms)) * (6.0 / (a2 * a2 * a2) + 6.0 / (a2 * a2) + 3.0 / a2 + 1.0) * exp(-a2) - accuracy;
    }
    return f;
  }
}

/** True when the box changed since the last set-up (fix nh, deform and change_box call kspace->setup()). */
const EwaldKeyChanged = (key: string, geom: Geometry): boolean => key !== `${geom.lx} ${geom.ly} ${geom.lz} ${geom.xy} ${geom.xz} ${geom.yz}`;

export { modifyNoEffect, EWALD_COUL, EWALD_DIPOLE, EWALD_DISP };
