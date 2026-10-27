/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIPOLE/pair_lj_long_dipole_long.cpp and pair_lj_long_dipole_long.h at commit b332f28 (Pieter in 't Veld,
 * SNL), with Pair::mix_energy and Pair::mix_distance (src/pair.cpp) and the constants of src/ewald_const.h. LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style lj/long/dipole/long flag_lj flag_coul cutoff (cutoff2): LJ (cut, or long with the r^-6 term in k-space)
 * and the real-space Ewald sum of point charges and point dipoles (the B0..B3 series of Wang and Holm with the
 * polynomial erfc of eqn 2.8; the excluded part of special pairs removed with eqns 2.13). flag_coul long hands both the
 * charge and the dipole term to kspace_style ewald/disp. As natively, the dispersion real-space term uses the kspace
 * style's g_ewald (not g_ewald_6), the forces and torques are summed per pair, and the global virial is the fdotr sum.
 */
import { Pair, PairParams, StyleError, type AdaptAccess, type KSpace, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { pow } from '../libm/pow';
import { exp } from '../libm/exp';
import { EWALD_COUL, EWALD_DIPOLE, EWALD_DISP, type EwaldDispPair } from './ewald_disp';

const MY_PIS = 1.77245385090551602729;
// ewald_const.h
const EWALD_P = 0.3275911;
const A1 = 0.254829592, A2 = -0.284496736, A3 = 1.421413741, A4 = -1.453152027, A5 = 1.061405429;

const numeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  return v;
};

export class PairLJLongDipoleLong extends Pair {
  readonly name = 'lj/long/dipole/long';
  virialFdotr = true;
  private ewaldOrder = 0;
  private ewaldOff = 0;
  private cutLJGlobal = 0;
  cutCoul = 0;
  private p!: PairParams;
  private eps = new Float64Array(0); private sig = new Float64Array(0);
  private cutLJsq = new Float64Array(0);
  /**
   * init_one's cutsq (it sets cutsq itself, so a fix adapt re-init changes it) and init_style's cut_coulsq (as native
   * LAMMPS does, src/DIPOLE/pair_lj_long_dipole_long.cpp at b332f28).
   */
  private cutsqNow = new Float64Array(0);
  private coulSqInit = 0;
  private lj1 = new Float64Array(0); private lj2 = new Float64Array(0);
  private lj3 = new Float64Array(0); private lj4 = new Float64Array(0);
  private offset = new Float64Array(0);
  private gEwaldLR = 0;
  private kspace: KSpace | null = null;

  /** PairLJLongDipoleLong::options: long sets the term's bit, off disables it, cut leaves both clear. */
  private options(w: string | undefined, mask: number): void {
    if (w === undefined) throw new StyleError('Illegal pair_style lj/long/dipole/long command');
    if (w === 'long') this.ewaldOrder |= mask;
    else if (w === 'off') this.ewaldOff |= mask;
    else if (w !== 'cut') throw new StyleError('Illegal pair_style lj/long/dipole/long command');
  }

  settings(args: string[], ctx: StyleContext): void {
    if (args.length !== 3 && args.length !== 4) throw new StyleError('Illegal pair_style command');
    this.ewaldOrder = 0;
    this.ewaldOff = 0;
    this.options(args[0], EWALD_DISP);
    this.options(args[1], EWALD_DIPOLE);
    this.options(args[1], EWALD_COUL);
    if (this.ewaldOrder & EWALD_DISP) ctx.log('WARNING: Geometric mixing assumed for 1/r^6 coefficients');
    if (this.ewaldOrder === (EWALD_DIPOLE | EWALD_DISP)) ctx.log('WARNING: Using largest cut-off for lj/long/dipole/long long long');
    if (args[2] === undefined) throw new StyleError('Cutoffs missing in pair_style lj/long/dipole/long');
    if (!((this.ewaldOrder ^ this.ewaldOff) & EWALD_DIPOLE)) throw new StyleError('Coulombic cut not supported in pair_style lj/long/dipole/long');
    this.cutLJGlobal = numeric(args[2]);
    if (args.length === 4 && this.ewaldOrder === (EWALD_COUL | EWALD_DIPOLE | EWALD_DISP)) {
      throw new StyleError('Only one cut-off allowed when requesting all long');
    }
    this.cutCoul = args.length === 4 ? numeric(args[3]) : this.cutLJGlobal;
    this.coulLong = (this.ewaldOrder & (EWALD_DIPOLE | EWALD_DISP)) !== 0;
    // pair_lj_long_dipole_long.cpp settings: dispersionflag = 0 unless the dispersion flag is long or off
    this.dispersionFlag = ((this.ewaldOrder ^ this.ewaldOff) & EWALD_DISP) !== 0;
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 4 || args.length > 5) throw new StyleError('Incorrect args for pair coefficients');
    const e = numeric(args[2]), s = numeric(args[3]);
    const c = args.length === 5 ? numeric(args[4]) : this.cutLJGlobal;
    this.p.setRange(args[0], args[1], [e, s, c]);
  }

  /** Pair::mix_energy. */
  private mixEnergy(e1: number, e2: number, s1: number, s2: number): number {
    if (this.mix === 'sixthpower') return (2.0 * Math.sqrt(e1 * e2) * pow(s1, 3.0) * pow(s2, 3.0)) / (pow(s1, 6.0) + pow(s2, 6.0));
    return Math.sqrt(e1 * e2);
  }

  /** Pair::mix_distance. */
  private mixDistance(s1: number, s2: number): number {
    if (this.mix === 'geometric') return Math.sqrt(s1 * s2);
    if (this.mix === 'arithmetic') return 0.5 * (s1 + s2);
    return pow(0.5 * (pow(s1, 6.0) + pow(s2, 6.0)), 1.0 / 6.0);
  }

  linkKSpace(k: KSpace | null): void { this.kspace = k; }

  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (s && s.units.style === 'electron') throw new StyleError("Cannot (yet) use 'electron' units with dipoles");
    if (s && !s.q && (this.ewaldOrder & EWALD_COUL)) throw new StyleError('Invoking coulombic in pair style lj/long/dipole/long requires atom attribute q');
    if (s && (!s.mu || !s.torque)) throw new StyleError('Pair lj/long/dipole/long requires atom attributes mu, torque');
    if (ctx.newtonPair === false) throw new StyleError('pair_style lj/long/dipole/long with newton off is not supported by the browser engine');
    const k = this.kspace;
    if (this.ewaldOrder & (EWALD_DIPOLE | EWALD_DISP)) {
      if (!k) throw new StyleError('Pair style requires a KSpace style');
      const disp = k.name === 'ewald/disp' || k.name === 'ewald/disp/dipole';
      if ((this.ewaldOrder & EWALD_DIPOLE) && !disp) throw new StyleError('Pair style requires use of kspace_style with dipole support');
      if ((this.ewaldOrder & EWALD_DISP) && !disp && k.name !== 'pppm/disp') throw new StyleError('Pair style requires use of kspace_style with dispersion support');
    }
    const nt = this.ntypes + 1;
    for (const key of ['eps', 'sig', 'cutLJsq', 'cutsqNow', 'lj1', 'lj2', 'lj3', 'lj4', 'offset'] as const) this[key] = new Float64Array(nt * nt);
    this.coulSqInit = this.cutCoul * this.cutCoul;
  }

  /** fix adapt cut_coul (ewald_cut the same) and cut_vdwl: native cut_coul and cut_lj_global. */
  override adaptParam(name: string): AdaptAccess | null {
    if (name === 'cut_coul' || name === 'ewald_cut') return { dim: 0, get: () => this.cutCoul, set: (v) => { this.cutCoul = v; } };
    if (name === 'cut_vdwl') return { dim: 0, get: () => this.cutLJGlobal, set: (v) => { this.cutLJGlobal = v; } };
    return null;
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    const disp = (this.ewaldOrder & EWALD_DISP) !== 0;
    let eps: number, sig: number, cutLJ: number;
    if (disp || !p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError('All pair coeffs are not set');
      eps = this.mixEnergy(p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j));
      sig = this.mixDistance(p.get('sigma', i, i), p.get('sigma', j, j));
      cutLJ = disp ? this.cutLJGlobal : this.mixDistance(p.get('cut', i, i), p.get('cut', j, j));
    } else {
      eps = p.get('epsilon', i, j); sig = p.get('sigma', i, j); cutLJ = p.get('cut', i, j);
    }
    const cut = Math.max(cutLJ, this.cutCoul);
    let offset = 0.0;
    if (this.shift && cutLJ > 0.0) {
      const ratio = sig / cutLJ;
      offset = 4.0 * eps * (pow(ratio, 12.0) - pow(ratio, 6.0));
    }
    const nt = this.ntypes + 1;
    for (const t of [i * nt + j, j * nt + i]) {
      this.eps[t] = eps; this.sig[t] = sig;
      this.cutLJsq[t] = cutLJ * cutLJ;
      this.cutsqNow[t] = cut * cut;
      this.lj1[t] = 48.0 * eps * pow(sig, 12.0);
      this.lj2[t] = 24.0 * eps * pow(sig, 6.0);
      this.lj3[t] = 4.0 * eps * pow(sig, 12.0);
      this.lj4[t] = 4.0 * eps * pow(sig, 6.0);
      this.offset[t] = offset;
    }
    return cut;
  }

  /** The ewald/disp view (native extract of ewald_order, ewald_mix, cut_coul, epsilon, sigma). */
  ewaldDisp(): EwaldDispPair {
    const nt = this.ntypes + 1;
    return {
      ewaldOrder: () => this.ewaldOrder,
      mix: this.mix,
      ewaldCut: () => this.cutCoul,
      epsilonOf: (i) => this.eps[i * nt + i],
      sigmaOf: (i) => this.sig[i * nt + i],
      setEwald: (g) => { this.gEwaldLR = g; },
    };
  }

  extract(name: string): unknown {
    // pair_lj_long_dipole_long.cpp extract: "sigma" and "epsilon" (dim 2: the mixed arrays after init_one)
    const nt = this.ntypes + 1;
    if (name === 'sigma') return this.sig.length === nt * nt ? this.sig : this.p.p('sigma');
    if (name === 'epsilon') return this.eps.length === nt * nt ? this.eps : this.p.p('epsilon');
    return name === 'cut_coul' ? this.cutCoul : undefined;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.mu || !s.torque) throw new StyleError('Pair lj/long/dipole/long requires atom attributes mu, torque');
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const owner = pc.nb.owner;
    const mu = s.mu;
    const tq = new Float64Array(3 * pc.nall);
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const qqrd2e = pc.qqrd2e;
    const nt = this.ntypes + 1;
    const order3 = this.ewaldOrder & EWALD_DIPOLE, order6 = this.ewaldOrder & EWALD_DISP;
    const gE = this.gEwaldLR || this.gEwald;
    const g2 = gE * gE, g6 = g2 * g2 * g2, g8 = g6 * g2;
    const C1 = (2.0 * gE) / MY_PIS, C2 = 2.0 * g2 * C1, C3 = 2.0 * g2 * C2;
    const cutCoulsq = this.coulSqInit;
    const { cutLJsq, lj1, lj2, lj3, lj4, offset } = this;
    const cutsq = this.cutsqNow;
    let evdwlSum = 0, ecoulSum = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const oi = owner[i];
      const qi = q[i];
      const ti = type[i] * nt;
      const xi0 = x[3 * i], xi1 = x[3 * i + 1], xi2 = x[3 * i + 2];
      const mui0 = mu[4 * oi], mui1 = mu[4 * oi + 1], mui2 = mu[4 * oi + 2];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const ni = jj >>> SBBITS;
        const j = jj & NEIGHMASK;
        const d0 = xi0 - x[3 * j], d1 = xi1 - x[3 * j + 1], d2 = xi2 - x[3 * j + 2];
        const rsq = d0 * d0 + d1 * d1 + d2 * d2;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const r2inv = 1.0 / rsq;
        let fd0 = 0.0, fd1 = 0.0, fd2 = 0.0;
        let ti0 = 0.0, ti1 = 0.0, ti2 = 0.0, tj0 = 0.0, tj1 = 0.0, tj2 = 0.0;
        let muj0 = 0.0, muj1 = 0.0, muj2 = 0.0;
        let forceCoul = 0.0, ecoul = 0.0;
        if (order3 && rsq < cutCoulsq) {
          const oj = owner[j];
          muj0 = mu[4 * oj]; muj1 = mu[4 * oj + 1]; muj2 = mu[4 * oj + 2];
          const r = Math.sqrt(rsq);
          const grij = gE * r;
          let fq = exp(-grij * grij) * qqrd2e;
          let B0 = 1.0 / (1.0 + EWALD_P * grij);
          B0 *= (((((A5 * B0 + A4) * B0 + A3) * B0 + A2) * B0 + A1) * fq) / r;
          let B1 = (B0 + C1 * fq) * r2inv;
          const B2 = (3.0 * B1 + C2 * fq) * r2inv;
          let B3 = (5.0 * B2 + C3 * fq) * r2inv;
          let mudi = mui0 * d0 + mui1 * d1 + mui2 * d2;
          let mudj = muj0 * d0 + muj1 * d1 + muj2 * d2;
          const muij = mui0 * muj0 + mui1 * muj1 + mui2 * muj2;
          const qj = q[j];
          const G0 = qi * qj;
          const G1 = qi * mudj - qj * mudi + muij;
          const G2 = -mudi * mudj;
          forceCoul = G0 * B1 + G1 * B2 + G2 * B3;
          mudi *= B2; mudj *= B2;
          ti0 = mudj * d0 + (qj * d0 - muj0) * B1;
          ti1 = mudj * d1 + (qj * d1 - muj1) * B1;
          ti2 = mudj * d2 + (qj * d2 - muj2) * B1;
          tj0 = mudi * d0 - (qi * d0 + mui0) * B1;
          tj1 = mudi * d1 - (qi * d1 + mui1) * B1;
          tj2 = mudi * d2 - (qi * d2 + mui2) * B1;
          ecoul = G0 * B0 + G1 * B1 + G2 * B2;
          if (ni > 0) {
            fq = (qqrd2e * (1.0 - sC[ni])) / r;
            forceCoul -= fq * ((3.0 * G1 + 15.0 * G2 * r2inv) * r2inv + G0) * r2inv;
            ecoul -= fq * ((G1 + 3.0 * G2 * r2inv) * r2inv + G0);
            B1 -= fq * r2inv;
          }
          B0 = mudj + qj * B1; B3 = -qi * B1 + mudi;
          if (ni > 0) B0 -= (fq * 3.0 * mudj * r2inv * r2inv) / B2;
          if (ni > 0) B3 -= (fq * 3.0 * mudi * r2inv * r2inv) / B2;
          fd0 = B0 * mui0 + B3 * muj0;
          fd1 = B0 * mui1 + B3 * muj1;
          fd2 = B0 * mui2 + B3 * muj2;
          if (ni > 0) {
            ti0 -= fq * ((3.0 * mudj * r2inv * r2inv * d0) / B2 + (qj * r2inv * d0 - muj0 * r2inv));
            ti1 -= fq * ((3.0 * mudj * r2inv * r2inv * d1) / B2 + (qj * r2inv * d1 - muj1 * r2inv));
            ti2 -= fq * ((3.0 * mudj * r2inv * r2inv * d2) / B2 + (qj * r2inv * d2 - muj2 * r2inv));
            tj0 -= fq * ((3.0 * mudi * r2inv * r2inv * d0) / B2 - (qi * r2inv * d0 + mui0 * r2inv));
            tj1 -= fq * ((3.0 * mudi * r2inv * r2inv * d1) / B2 - (qi * r2inv * d1 + mui1 * r2inv));
            tj2 -= fq * ((3.0 * mudi * r2inv * r2inv * d2) / B2 - (qi * r2inv * d2 + mui2 * r2inv));
          }
        }
        let forceLJ = 0.0, evdwl = 0.0;
        if (rsq < cutLJsq[t]) {
          const r6inv = r2inv * r2inv * r2inv;
          if (order6) {
            const r12inv = r6inv * r6inv;
            const gr2 = g2 * rsq, a2 = 1.0 / gr2;
            const expterm = a2 * exp(-gr2) * lj4[t];
            const g6term = g6 * ((a2 + 1.0) * a2 + 0.5) * expterm;
            const g8term = g8 * (((6.0 * a2 + 6.0) * a2 + 3.0) * a2 + 1.0) * expterm * rsq;
            if (ni === 0) {
              forceLJ = r12inv * lj1[t] - g8term;
              evdwl = r12inv * lj3[t] - g6term;
            } else {
              const factor = sL[ni], tt = r6inv * (1.0 - factor);
              forceLJ = factor * r12inv * lj1[t] - g8term + tt * lj2[t];
              evdwl = factor * r12inv * lj3[t] - g6term + tt * lj4[t];
            }
          } else if (ni === 0) {
            forceLJ = r6inv * (r6inv * lj1[t] - lj2[t]);
            evdwl = r6inv * (r6inv * lj3[t] - lj4[t]) - offset[t];
          } else {
            const factor = sL[ni];
            forceLJ = factor * r6inv * (r6inv * lj1[t] - lj2[t]);
            evdwl = factor * (r6inv * (r6inv * lj3[t] - lj4[t]) - offset[t]);
          }
          forceLJ *= r2inv;
        }
        const fpair = forceCoul + forceLJ;
        const fx = d0 * fpair + fd0, fy = d1 * fpair + fd1, fz = d2 * fpair + fd2;
        f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
        tq[3 * i] += mui1 * ti2 - mui2 * ti1;
        tq[3 * i + 1] += mui2 * ti0 - mui0 * ti2;
        tq[3 * i + 2] += mui0 * ti1 - mui1 * ti0;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        tq[3 * j] += muj1 * tj2 - muj2 * tj1;
        tq[3 * j + 1] += muj2 * tj0 - muj0 * tj2;
        tq[3 * j + 2] += muj0 * tj1 - muj1 * tj0;
        evdwlSum += evdwl;
        ecoulSum += ecoul;
        // ev_tally_xyz (newton on): half of the energy and of the virial to each atom
        if (pc.eatom) { const h = 0.5 * (evdwl + ecoul); pc.eatom[i] += h; pc.eatom[j] += h; }
        if (pc.vatom) {
          const va = pc.vatom;
          const w = [d0 * fx, d1 * fy, d2 * fz, d0 * fy, d0 * fz, d1 * fz];
          for (let c = 0; c < 6; c++) { va[6 * i + c] += 0.5 * w[c]; va[6 * j + c] += 0.5 * w[c]; }
        }
      }
    }
    pc.nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwlSum;
    pc.acc.ecoul += ecoulSum;
  }
}
