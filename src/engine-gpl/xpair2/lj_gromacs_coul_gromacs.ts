/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_lj_gromacs_coul_gromacs.cpp
 * and pair_lj_gromacs_coul_gromacs.h (PairLJGromacsCoulGromacs: pair_style lj/gromacs/coul/gromacs). The source names its
 * contributors (Mark Stevens, Sandia National Laboratories, who wrote the GROMACS-style switching of the pair styles).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2; this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The GROMACS-style shifted Lennard-Jones and Coulomb potentials (docs.lammps.org/pair_gromacs.html): both the force and
 * the energy are switched smoothly to zero between the inner and outer cutoffs. "None of the GROMACS pair styles support
 * the pair_modify shift option"; "None ... support the pair_modify tail option"; the mixing is geometric by default.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, type PairSep } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum, fmtCoeff } from '../../engine/force/util';
import { hasCharge } from '../../engine/atoms';
import { pow } from '../libm/pow';
import { forEachTypePair, mixDistanceNative, mixEnergyNative } from './common';

export class PairLJGromacsCoulGromacs extends Pair {
  readonly name: string = 'lj/gromacs/coul/gromacs';
  virialFdotr = true;
  coulLong = false;
  cutLJInner = 0; cutLJ = 0; cutCoulInner = 0; cutCoul = 0;
  private cutLJInnersq = 0; private cutLJsq = 0; private cutCoulInnersq = 0; private cutCoulsq = 0; private cutBothsq = 0;
  private coulsw1 = 0; private coulsw2 = 0; private coulsw3 = 0; private coulsw4 = 0; private coulsw5 = 0;
  private p!: PairParams;
  private lj1 = new Float64Array(0);
  private lj2 = new Float64Array(0);
  private lj3 = new Float64Array(0);
  private lj4 = new Float64Array(0);
  private ljsw1 = new Float64Array(0);
  private ljsw2 = new Float64Array(0);
  private ljsw3 = new Float64Array(0);
  private ljsw4 = new Float64Array(0);
  private ljsw5 = new Float64Array(0);
  /** force->qqrd2e as the last compute() saw it (single() takes it from the force field). */
  private qqrd2eSingle = 1;

  settings(args: string[]): void {
    // pair_lj_gromacs_coul_gromacs.cpp settings(): two or four arguments
    if (args.length !== 2 && args.length !== 4) throw new StyleError('Illegal pair_style command');
    this.cutLJInner = parseNum(args[0], 'cutoff');
    this.cutLJ = parseNum(args[1], 'cutoff');
    if (args.length === 2) {
      this.cutCoulInner = this.cutLJInner;
      this.cutCoul = this.cutLJ;
    } else {
      this.cutCoulInner = parseNum(args[2], 'coulomb inner cutoff');
      this.cutCoul = parseNum(args[3], 'coulomb cutoff');
    }
    if (this.cutLJInner <= 0.0 || this.cutCoulInner < 0.0) throw new StyleError('Illegal pair_style command');
    if (this.cutLJInner > this.cutLJ || this.cutCoulInner > this.cutCoul) throw new StyleError('Illegal pair_style command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma']);
    const nt = ntypes + 1;
    const mk = () => new Float64Array(nt * nt);
    this.lj1 = mk(); this.lj2 = mk(); this.lj3 = mk(); this.lj4 = mk();
    this.ljsw1 = mk(); this.ljsw2 = mk(); this.ljsw3 = mk(); this.ljsw4 = mk(); this.ljsw5 = mk();
  }

  coeff(args: string[]): void {
    // pair_lj_gromacs_coul_gromacs.cpp coeff(): exactly I J epsilon sigma
    if (args.length !== 4) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    forEachTypePair(args[0], args[1], this.ntypes, (i, j) => this.p.set(i, j, [eps, sig]));
  }

  initStyle(ctx: StyleContext): void {
    // init_style(): the charge attribute, then the switching radii squared
    if (!ctx.s || !hasCharge(ctx.s)) throw new StyleError('Pair style lj/gromacs/coul/gromacs requires atom attribute q');
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    this.cutLJInnersq = this.cutLJInner * this.cutLJInner;
    this.cutLJsq = this.cutLJ * this.cutLJ;
    this.cutCoulInnersq = this.cutCoulInner * this.cutCoulInner;
    this.cutCoulsq = this.cutCoul * this.cutCoul;
    this.cutBothsq = Math.max(this.cutLJsq, this.cutCoulsq);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
      p.setMixed(i, j, 'epsilon', mixEnergyNative(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'sigma', mixDistanceNative(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
    }
    const cut = Math.max(this.cutLJ, this.cutCoul);
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j);
    const cLJ = this.cutLJ, cLJi = this.cutLJInner, cC = this.cutCoul, cCi = this.cutCoulInner;
    const k1 = i * (this.ntypes + 1) + j;
    this.lj1[k1] = 48.0 * eps * pow(sig, 12.0);
    this.lj2[k1] = 24.0 * eps * pow(sig, 6.0);
    this.lj3[k1] = 4.0 * eps * pow(sig, 12.0);
    this.lj4[k1] = 4.0 * eps * pow(sig, 6.0);
    const r6inv = 1.0 / pow(cLJ, 6.0);
    const r8inv = 1.0 / pow(cLJ, 8.0);
    let t = cLJ - cLJi;
    let t2inv = 1.0 / (t * t);
    let t3inv = t2inv / t;
    const t3 = 1.0 / t3inv;
    const a6 = (7.0 * cLJi - 10.0 * cLJ) * r8inv * t2inv;
    const b6 = (9.0 * cLJ - 7.0 * cLJi) * r8inv * t3inv;
    const a12 = (13.0 * cLJi - 16.0 * cLJ) * r6inv * r8inv * t2inv;
    const b12 = (15.0 * cLJ - 13.0 * cLJi) * r6inv * r8inv * t3inv;
    const c6 = r6inv - t3 * (6.0 * a6 / 3.0 + 6.0 * b6 * t / 4.0);
    const c12 = r6inv * r6inv - t3 * (12.0 * a12 / 3.0 + 12.0 * b12 * t / 4.0);
    this.ljsw1[k1] = this.lj1[k1] * a12 - this.lj2[k1] * a6;
    this.ljsw2[k1] = this.lj1[k1] * b12 - this.lj2[k1] * b6;
    this.ljsw3[k1] = -this.lj3[k1] * 12.0 * a12 / 3.0 + this.lj4[k1] * 6.0 * a6 / 3.0;
    this.ljsw4[k1] = -this.lj3[k1] * 12.0 * b12 / 4.0 + this.lj4[k1] * 6.0 * b6 / 4.0;
    this.ljsw5[k1] = -this.lj3[k1] * c12 + this.lj4[k1] * c6;
    const r3inv = 1.0 / pow(cC, 3.0);
    t = cC - cCi;
    t2inv = 1.0 / (t * t);
    t3inv = t2inv / t;
    const a1 = (2.0 * cCi - 5.0 * cC) * r3inv * t2inv;
    const b1 = (4.0 * cC - 2.0 * cCi) * r3inv * t3inv;
    this.coulsw1 = a1;
    this.coulsw2 = b1;
    this.coulsw3 = -a1 / 3.0;
    this.coulsw4 = -b1 / 4.0;
    this.coulsw5 = 1.0 / cC - t * t * t * (a1 / 3.0 + b1 * t / 4.0);
    const k2 = j * (this.ntypes + 1) + i;
    this.lj1[k2] = this.lj1[k1];
    this.lj2[k2] = this.lj2[k1];
    this.lj3[k2] = this.lj3[k1];
    this.lj4[k2] = this.lj4[k1];
    this.ljsw1[k2] = this.ljsw1[k1];
    this.ljsw2[k2] = this.ljsw2[k1];
    this.ljsw3[k2] = this.ljsw3[k1];
    this.ljsw4[k2] = this.ljsw4[k1];
    this.ljsw5[k2] = this.ljsw5[k1];
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const qqrd2e = pc.qqrd2e;
    this.qqrd2eSingle = qqrd2e;
    const { cutBothsq, cutCoulsq, cutLJsq, cutCoulInnersq, cutLJInnersq } = this;
    const { cutCoulInner, cutLJInner, coulsw1, coulsw2, coulsw3, coulsw4, coulsw5 } = this;
    const { lj1, lj2, lj3, lj4, ljsw1, ljsw2, ljsw3, ljsw4, ljsw5 } = this;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qtmp = q[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const factorCoul = sC[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutBothsq) {
          const r2inv = 1.0 / rsq;
          const qj = q[j];
          let forcecoul = 0;
          let tc = 0;
          // skip if qi or qj = 0.0 since this potential may be used as coarse-grain model with many uncharged atoms
          if (rsq < cutCoulsq && qtmp !== 0.0 && qj !== 0.0) {
            forcecoul = qqrd2e * qtmp * qj * Math.sqrt(r2inv);
            if (rsq > cutCoulInnersq) {
              const r = Math.sqrt(rsq);
              tc = r - cutCoulInner;
              const fswitchcoul = qqrd2e * qtmp * qj * r * tc * tc * (coulsw1 + coulsw2 * tc);
              forcecoul += fswitchcoul;
            }
          }
          const jtype = type[j];
          const t = itype * nt + jtype;
          let forcelj = 0, r6inv = 0, tlj = 0;
          if (rsq < cutLJsq) {
            r6inv = r2inv * r2inv * r2inv;
            forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
            if (rsq > cutLJInnersq) {
              const r = Math.sqrt(rsq);
              tlj = r - cutLJInner;
              const fswitch = r * tlj * tlj * (ljsw1[t] + ljsw2[t] * tlj);
              forcelj += fswitch;
            }
          }
          const fpair = (factorCoul * forcecoul + factorLJ * forcelj) * r2inv;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          let e = 0, ec = 0;
          if (rsq < cutCoulsq) {
            ec = qqrd2e * qtmp * qj * (Math.sqrt(r2inv) - coulsw5);
            if (rsq > cutCoulInnersq) {
              // tc is the switch distance of this pair (the charges make the term vanish when either is zero)
              const tcs = Math.sqrt(rsq) - cutCoulInner;
              const ecoulswitch = tcs * tcs * tcs * (coulsw3 + coulsw4 * tcs);
              ec += qqrd2e * qtmp * qj * ecoulswitch;
            }
            ec *= factorCoul;
          }
          if (rsq < cutLJsq) {
            e = r6inv * (lj3[t] * r6inv - lj4[t]);
            e += ljsw5[t];
            if (rsq > cutLJInnersq) {
              const eswitch = tlj * tlj * tlj * (ljsw3[t] + ljsw4[t] * tlj);
              e += eswitch;
            }
            e *= factorLJ;
          }
          evdwl += e;
          ecoul += ec;
          if (tally) tallyAtom(pc, i, j, e + ec, fpair, delx, dely, delz);
        }
      }
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number, _del?: PairSep) {
    const qqrd2e = this.qqrd2eSingle;
    const t = itype * (this.ntypes + 1) + jtype;
    const r2inv = 1.0 / rsq;
    const { coulsw1, coulsw2, coulsw3, coulsw4, coulsw5, cutCoulInner, cutLJInner } = this;
    let forcecoul = 0, tc = 0;
    if (rsq < this.cutCoulsq) {
      forcecoul = qqrd2e * qi * qj * Math.sqrt(r2inv);
      if (rsq > this.cutCoulInnersq) {
        const r = Math.sqrt(rsq);
        tc = r - cutCoulInner;
        const fswitchcoul = qqrd2e * qi * qj * r * tc * tc * (coulsw1 + coulsw2 * tc);
        forcecoul += fswitchcoul;
      }
    }
    let forcelj = 0, r6inv = 0, tlj = 0;
    if (rsq < this.cutLJsq) {
      r6inv = r2inv * r2inv * r2inv;
      forcelj = r6inv * (this.lj1[t] * r6inv - this.lj2[t]);
      if (rsq > this.cutLJInnersq) {
        const r = Math.sqrt(rsq);
        tlj = r - cutLJInner;
        const fswitch = r * tlj * tlj * (this.ljsw1[t] + this.ljsw2[t] * tlj);
        forcelj += fswitch;
      }
    }
    const fforce = (factorCoul * forcecoul + factorLJ * forcelj) * r2inv;
    let eng = 0.0;
    if (rsq < this.cutCoulsq) {
      let phicoul = qqrd2e * qi * qj * (Math.sqrt(r2inv) - coulsw5);
      if (rsq > this.cutCoulInnersq) {
        const phiswitchcoul = qqrd2e * qi * qj * tc * tc * tc * (coulsw3 + coulsw4 * tc);
        phicoul += phiswitchcoul;
      }
      eng += factorCoul * phicoul;
    }
    if (rsq < this.cutLJsq) {
      let philj = r6inv * (this.lj3[t] * r6inv - this.lj4[t]);
      philj += this.ljsw5[t];
      if (rsq > this.cutLJInnersq) {
        const phiswitch = tlj * tlj * tlj * (this.ljsw3[t] + this.ljsw4[t] * tlj);
        philj += phiswitch;
      }
      eng += factorLJ * philj;
    }
    return { eng, fforce };
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) out.push(`${i} ${j} ${fmtCoeff(this.p.get('epsilon', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))}`);
    }
    return out;
  }

  extract(name: string): unknown {
    if (name === 'epsilon' || name === 'sigma') return this.p.p(name);
    return undefined;
  }
}
