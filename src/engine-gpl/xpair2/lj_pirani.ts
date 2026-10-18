/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_lj_pirani.cpp and
 * pair_lj_pirani.h (PairLJPirani: pair_style lj/pirani, the Pirani "ILJ" potential). The source names its contributors
 * (Fundamental Physics Institute, http://intermol.iff.csic.es/; the optimization by Axel Kohlmeyer, Temple University).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Pirani potential (docs.lammps.org/pair_lj_pirani.html): E = eps [ gamma/(n(r) - gamma) (rm/r)^n(r)
 * - n(r)/(n(r) - gamma) (rm/r)^gamma ] with n(r) = alpha (r/rm)^2 + beta. The respa inner and middle loops of the
 * source are not provided (the engine has no rRESPA), so the pair is evaluated as the plain compute() does.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, type PairSep } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum, fmtCoeff } from '../../engine/force/util';
import { pow } from '../libm/pow';
import { forEachTypePair } from './common';
import { log } from '../libm/log';

/** The Pirani force-over-r and energy of one pair (compute() and single() of pair_lj_pirani.cpp). */
const piraniPair = (r: number, rm: number, alpha: number, beta: number, gamma: number, eps: number, viaPow: boolean): { forceilj: number; ilj: number } => {
  const rx = r / rm;
  const nX = alpha * rx * rx + beta;
  const powRxNX = pow(1.0 / rx, nX);
  const powRxGamma = pow(1.0 / rx, gamma);
  const dn = nX - gamma;
  // compute() squares with square() (x * x); single() calls pow(x, 2.0), as the two native functions do
  const dn2 = viaPow ? pow(dn, 2.0) : dn * dn;
  const filj1 = -2.0 * alpha * gamma * rx * powRxNX / (dn2 * rm);
  const filj2 = +2.0 * alpha * rx * nX * powRxGamma / (dn2 * rm);
  const filj3 = -2.0 * alpha * rx * powRxGamma / (rm * dn);
  const filj4 = +2.0 * alpha * gamma * (rx / rm) * log(1 / rx) * powRxNX / dn;
  const filj5 = -1.0 * gamma * nX * powRxNX / (r * dn);
  const filj6 = +1.0 * gamma * nX * powRxGamma / (r * dn);
  // F = -dV/dr
  const forceilj = -eps * (filj1 + filj2 + filj3 + filj4 + filj5 + filj6);
  const ilj1 = eps * gamma * pow(1 / rx, nX) / (nX - gamma);
  const ilj2 = -eps * nX * pow(1 / rx, gamma) / (nX - gamma);
  return { forceilj, ilj: ilj1 + ilj2 };
};

export class PairLJPirani extends Pair {
  readonly name: string = 'lj/pirani';
  virialFdotr = true;
  cutGlobal = 0;
  private p!: PairParams;
  private offset = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Pair style ilj/cut must have exactly one argument: cutoff distance');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    // native settings(): every explicitly set pair takes the new global cutoff ("reset cutoffs that have been explicitly set")
    if (this.p) for (let i = 1; i <= this.ntypes; i++) for (let j = i; j <= this.ntypes; j++) if (this.p.isSet(i, j)) this.p.setMixed(i, j, 'cut', this.cutGlobal);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['alpha', 'beta', 'gamma', 'rm', 'epsilon', 'cut']);
    const nt = ntypes + 1;
    this.offset = new Float64Array(nt * nt);
  }

  coeff(args: string[]): void {
    // pair_lj_pirani.cpp coeff(): 7 or 8 coefficients: 5 for the ILJ, 2 for the pair, 1 for the cutoff (optional)
    if (args.length < 7 || args.length > 8) throw new StyleError('Incorrect args for pair coefficients');
    const alpha = parseNum(args[2], 'alpha');
    const beta = parseNum(args[3], 'beta');
    const gamma = parseNum(args[4], 'gamma');
    const rm = parseNum(args[5], 'rm');
    const eps = parseNum(args[6], 'epsilon');
    const cutOne = args.length === 8 ? parseNum(args[7], 'cutoff') : this.cutGlobal;
    if (rm <= 0.0 || eps < 0.0 || gamma <= 0.0) throw new StyleError('Illegal ILJ coefficients');
    const vals = [alpha, beta, gamma, rm, eps, cutOne];
    forEachTypePair(args[0], args[1], this.ntypes, (i, j) => this.p.set(i, j, vals));
  }

  initStyle(ctx: StyleContext): void {
    // pair_lj_pirani.rst: "does not support the pair_modify tail option"
    this.noTermLog(ctx, 'tail');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    // init_one(): no mixing; the pair must have been set
    if (!p.isSet(i, j)) throw new StyleError('All pair coeffs are not set');
    const alpha = p.get('alpha', i, j), beta = p.get('beta', i, j), gamma = p.get('gamma', i, j);
    const rm = p.get('rm', i, j), eps = p.get('epsilon', i, j), cut = p.get('cut', i, j);
    const nt = this.ntypes + 1;
    if (this.shift && cut > 0.0) {
      const r = cut / rm;
      const nx = alpha * r * r + beta;
      this.offset[i * nt + j] = eps * ((gamma / (nx - gamma)) * pow(1 / r, nx) - (nx / (nx - gamma)) * pow(1 / r, gamma));
    } else this.offset[i * nt + j] = 0.0;
    this.offset[j * nt + i] = this.offset[i * nt + j];
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const alpha = this.p.p('alpha'), beta = this.p.p('beta'), gamma = this.p.p('gamma');
    const rmA = this.p.p('rm'), epsA = this.p.p('epsilon');
    const { cutsq, offset } = this;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      let fxtmp = 0, fytmp = 0, fztmp = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = itype * nt + type[j];
        if (rsq < cutsq[t]) {
          const r = Math.sqrt(rsq);
          const { forceilj, ilj } = piraniPair(r, rmA[t], alpha[t], beta[t], gamma[t], epsA[t], false);
          const fpair = factorLJ * forceilj / r;
          fxtmp += delx * fpair;
          fytmp += dely * fpair;
          fztmp += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          const e = factorLJ * (ilj - offset[t]);
          evdwl += e;
          if (tally) tallyAtom(pc, i, j, e, fpair, delx, dely, delz);
        }
      }
      f[3 * i] += fxtmp;
      f[3 * i + 1] += fytmp;
      f[3 * i + 2] += fztmp;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number, _del?: PairSep) {
    const t = itype * (this.ntypes + 1) + jtype;
    const r = Math.sqrt(rsq);
    const { forceilj, ilj } = piraniPair(r, this.p.get('rm', itype, jtype), this.p.get('alpha', itype, jtype),
      this.p.get('beta', itype, jtype), this.p.get('gamma', itype, jtype), this.p.get('epsilon', itype, jtype), true);
    return { eng: factorLJ * (ilj - this.offset[t]), fforce: factorLJ * forceilj / r };
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('alpha', i, i))} ${fmtCoeff(this.p.get('beta', i, i))} ${fmtCoeff(this.p.get('gamma', i, i))} ${fmtCoeff(this.p.get('rm', i, i))} ${fmtCoeff(this.p.get('epsilon', i, i))}`);
    }
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('alpha', i, j))} ${fmtCoeff(this.p.get('beta', i, j))} ${fmtCoeff(this.p.get('gamma', i, j))} ${fmtCoeff(this.p.get('rm', i, j))} ${fmtCoeff(this.p.get('epsilon', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    if (['alpha', 'beta', 'gamma', 'rm', 'epsilon'].includes(name)) return this.p.p(name);
    return undefined;
  }
}
