/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_born_gauss.cpp and
 * pair_born_gauss.h (pair_style born/gauss, Contributing author: Axel Kohlmeyer, Temple University). This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * pair_style born/gauss cutoff; pair_coeff I J biga0 alpha biga1 beta r0 [cutoff] (docs.lammps.org/pair_born_gauss.html):
 *   E = biga0 exp(-alpha r) - biga1 exp(-beta (r - r0)^2)   r < rc, minus the shift when pair_modify shift yes.
 * Native init_one() mixes nothing. Native's coeff() reads the optional cutoff only when the word count is 10, which a
 * line of 7 or 8 words never has, so the pair takes the global cutoff whatever the 8th word says (kept as native).
 */
import { Pair, PairParams, StyleError, type PairCompute, type PairSep, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { fmtCoeff, parseNum } from '../../engine/force/util';
import { exp } from '../libm/exp';
import { setPairRange, resetSetCutoffs, table } from './common';

const NAMES = ['biga0', 'alpha', 'biga1', 'beta', 'r0', 'cut'];

export class PairBornGauss extends Pair {
  readonly name: string = 'born/gauss';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  offset = new Float64Array(0);

  settings(args: string[]): void {
    // pair_born_gauss.cpp settings(): "Pair style born/gauss must have exactly one argument"
    if (args.length !== 1) throw new StyleError('Pair style born/gauss must have exactly one argument');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    resetSetCutoffs(this.p, 'cut', this.cutGlobal, this.ntypes);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, NAMES);
  }

  coeff(args: string[]): void {
    // pair_born_gauss.cpp coeff(): narg is 7 or 8; cut_one stays cut_global (see the header)
    if (args.length < 7 || args.length > 8) throw new StyleError('Incorrect args for pair coefficients');
    const vals = args.slice(2, 7).map((w, k) => parseNum(w, NAMES[k]));
    setPairRange(this.p, args[0], args[1], [...vals, this.cutGlobal], 'Incorrect args for pair coefficients', this.ntypes);
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'tail');
    if (this.offset.length !== (this.ntypes + 1) ** 2) this.offset = table(this.ntypes);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.ntypes + 1;
    const cut = p.get('cut', i, j);
    let off = 0;
    if (this.shift) {
      const dr = cut - p.get('r0', i, j);
      off = p.get('biga0', i, j) * exp(-p.get('alpha', i, j) * cut) - p.get('biga1', i, j) * exp(-p.get('beta', i, j) * dr * dr);
    }
    this.offset[i * nt + j] = this.offset[j * nt + i] = off;
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const biga0 = this.p.p('biga0'), alpha = this.p.p('alpha'), biga1 = this.p.p('biga1'), beta = this.p.p('beta'), r0 = this.p.p('r0');
    const offset = this.offset;
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
        const dr = r - r0[t];
        const aexp = biga0[t] * exp(-alpha[t] * r);
        const bexp = biga1[t] * exp(-beta[t] * dr * dr);
        let fpair = alpha[t] * aexp;
        fpair -= 2.0 * beta[t] * dr * bexp;
        fpair *= factor / r;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        const e = factor * (aexp - bexp - offset[t]);
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number, _del?: PairSep) {
    const t = itype * (this.ntypes + 1) + jtype;
    const r = Math.sqrt(rsq);
    const dr = r - this.p.p('r0')[t];
    const aexp = this.p.p('biga0')[t] * exp(-this.p.p('alpha')[t] * r);
    const bexp = this.p.p('biga1')[t] * exp(-this.p.p('beta')[t] * dr * dr);
    const fforce = factorLJ * (this.p.p('alpha')[t] * aexp - 2.0 * dr * this.p.p('beta')[t] * bexp) / r;
    return { eng: factorLJ * (aexp - bexp - this.offset[t]), fforce };
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      const g = (n: string) => fmtCoeff(this.p.get(n, i, i));
      out.push(`${i} ${g('biga0')} ${g('alpha')} ${g('biga1')} ${g('beta')} ${g('r0')}`);
    }
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const g = (n: string) => fmtCoeff(this.p.get(n, i, j));
        out.push(`${i} ${j} ${g('biga0')} ${g('alpha')} ${g('biga1')} ${g('beta')} ${g('r0')} ${g('cut')}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    // pair_born_gauss.cpp extract(): biga0, biga1, r0 (dim 2)
    if (name === 'biga0' || name === 'biga1' || name === 'r0') return this.p.p(name);
    return undefined;
  }
}
