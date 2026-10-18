/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_beck.cpp and
 * pair_beck.h (pair_style beck, Contributing author: Jonathan Zimmerman (Sandia)). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * pair_style beck cutoff; pair_coeff I J AA BB aa alpha beta [cutoff] (docs.lammps.org/pair_beck.html): the Beck
 * potential, a generalised Lennard-Jones form with a range-dependent exponent. Native init_one() mixes nothing, so
 * every pair must be set; native has no shift or tail term.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum } from '../../engine/force/util';
import { exp } from '../libm/exp';
import { powint, setPairRange, resetSetCutoffs } from './common';

const NAMES = ['AA', 'BB', 'aa', 'alpha', 'beta', 'cut'];

export class PairBeck extends Pair {
  readonly name: string = 'beck';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;

  settings(args: string[]): void {
    // pair_beck.cpp settings(): "if (narg != 1) error->all(FLERR, "Illegal pair_style command")"
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    resetSetCutoffs(this.p, 'cut', this.cutGlobal, this.ntypes);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, NAMES);
  }

  coeff(args: string[]): void {
    // pair_beck.cpp coeff(): narg must be 7 or 8; the 8th word is the pair cutoff
    if (args.length !== 7 && args.length !== 8) throw new StyleError('Incorrect args for pair coefficients');
    const vals = args.slice(2, 7).map((w, k) => parseNum(w, NAMES[k]));
    const cut = args.length === 8 ? parseNum(args[7], 'cutoff') : this.cutGlobal;
    setPairRange(this.p, args[0], args[1], [...vals, cut], 'Incorrect args for pair coefficients', this.ntypes);
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    return this.p.get('cut', i, j);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const AA = this.p.p('AA'), BB = this.p.p('BB'), aa = this.p.p('aa'), alpha = this.p.p('alpha'), beta = this.p.p('beta');
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
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
        const r5 = rsq * rsq * r;
        const aaij = aa[t], alphaij = alpha[t], betaij = beta[t];
        const term1 = aaij * aaij + rsq;
        const term2 = powint(term1, -5);
        const term3 = 21.672 + 30.0 * aaij * aaij + 6.0 * rsq;
        const term4 = alphaij + r5 * betaij;
        const term5 = alphaij + 6.0 * r5 * betaij;
        const rinv = 1.0 / r;
        let forceBeck = AA[t] * exp(-1.0 * r * term4) * term5;
        forceBeck -= BB[t] * r * term2 * term3;
        const fpair = factor * forceBeck * rinv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        const term6 = powint(term1, -3);
        const term1inv = 1.0 / term1;
        let e = AA[t] * exp(-1.0 * r * term4);
        e -= BB[t] * term6 * (1.0 + (2.709 + 3.0 * aaij * aaij) * term1inv);
        e *= factor;
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    const AA = this.p.p('AA'), BB = this.p.p('BB'), aa = this.p.p('aa'), alpha = this.p.p('alpha'), beta = this.p.p('beta');
    const r = Math.sqrt(rsq);
    const r5 = rsq * rsq * r;
    const aaij = aa[t], alphaij = alpha[t], betaij = beta[t];
    const term1 = aaij * aaij + rsq;
    const term2 = powint(term1, -5);
    const term3 = 21.672 + 30.0 * aaij * aaij + 6.0 * rsq;
    const term4 = alphaij + r5 * betaij;
    const term5 = alphaij + 6.0 * r5 * betaij;
    const rinv = 1.0 / r;
    let forceBeck = AA[t] * exp(-1.0 * r * term4) * term5;
    forceBeck -= BB[t] * r * term2 * term3;
    const fforce = factorLJ * forceBeck * rinv;
    const term6 = powint(term1, -3);
    const term1inv = 1.0 / term1;
    let phi = AA[t] * exp(-1.0 * r * term4);
    phi -= BB[t] * term6 * (1.0 + (2.709 + 3.0 * aaij * aaij) * term1inv);
    return { eng: factorLJ * phi, fforce };
  }

  // pair_beck.cpp has no extract(): the base class returns nothing for any name
}
