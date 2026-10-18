/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_cosine_squared.cpp
 * and pair_cosine_squared.h (pair_style cosine/squared, Contributing authors: Eugen Rozic, University College London).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style cosine/squared cutoff; pair_coeff I J epsilon sigma [cutoff] [wca] (docs.lammps.org/pair_cosine_squared.html):
 *   E = -epsilon             r < sigma without wca (wca: the WCA repulsion of the 12-6 form)
 *   E = -epsilon cos^2(pi (r - sigma) / (2 w))   sigma < r < sigma + w,  w = cutoff - sigma
 * Native has no mixing (init_one errors for an unset pair), refuses pair_modify mix, and has no shift or tail term.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { fmtCoeff, parseNum } from '../../engine/force/util';
import { pow } from '../libm/pow';
import { sinFma as sin, cosFma as cos } from '../libm/sincos_fma';
import { setPairRange, resetSetCutoffs, table, mirror } from './common';

/** MY_PI (math_const.h): the double nearest pi. */
const MY_PI = 3.14159265358979323846;

export class PairCosineSquared extends Pair {
  readonly name: string = 'cosine/squared';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  w = new Float64Array(0); lj12e = new Float64Array(0); lj6e = new Float64Array(0);
  lj12f = new Float64Array(0); lj6f = new Float64Array(0);

  constructor() {
    super();
    // pair_cosine_squared.cpp modify_params(): "pair_modify mix not supported for pair_style cosine/squared"
    Object.defineProperty(this, 'mix', {
      configurable: true,
      get: () => 'geometric',
      set: () => { throw new StyleError('pair_modify mix not supported for pair_style cosine/squared'); },
    });
  }

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style command (wrong number of params)');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    // native settings() resets the cutoffs of the pairs with j > i only
    resetSetCutoffs(this.p, 'cut', this.cutGlobal, this.ntypes, (i) => i + 1);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cut', 'wca']);
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (args.length < 4 || args.length > 6) throw new StyleError('Incorrect args for pair coefficients (too few or too many)');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    let cutOne = this.cutGlobal;
    let wca = 0;
    if (args.length === 6) {
      cutOne = parseNum(args[4], 'cutoff');
      if (args[5] === 'wca') wca = 1;
      else throw new StyleError('Incorrect args for pair coefficients (unknown option)');
    } else if (args.length === 5) {
      if (args[4] === 'wca') wca = 1;
      else cutOne = parseNum(args[4], 'cutoff');
    }
    if (cutOne < sig) {
      throw new StyleError('Incorrect args for pair coefficients (cutoff < sigma)');
    } else if (cutOne === sig) {
      if (wca === 0) throw new StyleError('Incorrect args for pair coefficients (cutoff = sigma w/o wca)');
      ctx.log('WARNING: Cosine/squared set to WCA only (cutoff = sigma)');
    }
    setPairRange(this.p, args[0], args[1], [eps, sig, cutOne, wca], 'Incorrect args for pair coefficients (none set)', this.ntypes);
  }

  initStyle(ctx: StyleContext): void {
    // pair_modify shift / tail: native sets offset_flag and tail_flag to 0 with a warning; this style has neither term
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    if (this.w.length !== (this.ntypes + 1) ** 2) {
      this.w = table(this.ntypes); this.lj12e = table(this.ntypes); this.lj6e = table(this.ntypes);
      this.lj12f = table(this.ntypes); this.lj6f = table(this.ntypes);
    }
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    // pair_cosine_squared.cpp init_one(): "Mixing not supported in pair_style cosine/squared"
    if (!p.isSet(i, j)) throw new StyleError('Mixing not supported in pair_style cosine/squared');
    const nt = this.ntypes + 1;
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), cut = p.get('cut', i, j);
    const k = i * nt + j;
    this.w[k] = cut - sig;
    if (p.get('wca', i, j)) {
      this.lj12e[k] = eps * pow(sig, 12.0);
      this.lj6e[k] = 2.0 * eps * pow(sig, 6.0);
      this.lj12f[k] = 12.0 * eps * pow(sig, 12.0);
      this.lj6f[k] = 12.0 * eps * pow(sig, 6.0);
    }
    for (const t of [this.w, this.lj12e, this.lj6e, this.lj12f, this.lj6f]) mirror(t, nt, i, j);
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const eps = this.p.p('epsilon'), sigma = this.p.p('sigma'), cutP = this.p.p('cut'), wcaP = this.p.p('wca');
    const { w, lj12e, lj6e, lj12f, lj6f } = this;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl0 = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (!(rsq < cutsq[t])) continue;
        const factor = sLJ[jj >>> SBBITS];
        const r = Math.sqrt(rsq);
        let fpair: number, evdwl: number;
        if (r <= sigma[t]) {
          if (wcaP[t]) {
            const r2inv = 1.0 / rsq;
            const r6inv = r2inv * r2inv * r2inv;
            const forceLJ = r6inv * (lj12f[t] * r6inv - lj6f[t]);
            fpair = factor * forceLJ * r2inv;
            evdwl = factor * r6inv * (lj12e[t] * r6inv - lj6e[t]);
            if (sigma[t] === cutP[t]) evdwl += factor * eps[t];
          } else {
            fpair = 0.0;
            evdwl = -factor * eps[t];
          }
        } else {
          const forceCos = -(MY_PI * eps[t] / (2.0 * w[t])) * sin(MY_PI * (r - sigma[t]) / w[t]);
          fpair = factor * forceCos / r;
          const cosone = cos(MY_PI * (r - sigma[t]) / (2.0 * w[t]));
          evdwl = -factor * eps[t] * cosone * cosone;
        }
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        evdwl0 += evdwl;
        if (tally) tallyAtom(pc, i, j, evdwl, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl0;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    const eps = this.p.get('epsilon', itype, jtype), sig = this.p.get('sigma', itype, jtype), cut = this.p.get('cut', itype, jtype);
    const r = Math.sqrt(rsq);
    let force: number, energy: number;
    if (r <= sig) {
      if (this.p.get('wca', itype, jtype)) {
        const r2inv = 1.0 / rsq;
        const r6inv = r2inv * r2inv * r2inv;
        force = r6inv * (this.lj12f[t] * r6inv - this.lj6f[t]) * r2inv;
        energy = r6inv * (this.lj12e[t] * r6inv - this.lj6e[t]);
        if (sig === cut) energy += eps;
      } else {
        force = 0.0;
        energy = -eps;
      }
    } else {
      const cosone = cos(MY_PI * (r - sig) / (2.0 * this.w[t]));
      force = -(MY_PI * eps / (2.0 * this.w[t])) * sin(MY_PI * (r - sig) / this.w[t]) / r;
      energy = -eps * cosone * cosone;
    }
    return { eng: factorLJ * energy, fforce: factorLJ * force };
  }

  dataCoeffs(): string[] | null {
    // "%d %g %g %g %s": the wca word is empty (a trailing space) for a pair without it
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      const g = (n: string) => fmtCoeff(this.p.get(n, i, i));
      out.push(`${i} ${g('epsilon')} ${g('sigma')} ${g('cut')} ${this.p.get('wca', i, i) ? 'wca' : ''}`);
    }
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const g = (n: string) => fmtCoeff(this.p.get(n, i, j));
        out.push(`${i} ${j} ${g('epsilon')} ${g('sigma')} ${g('cut')} ${this.p.get('wca', i, j) ? 'wca' : ''}`);
      }
    }
    return out;
  }
}
