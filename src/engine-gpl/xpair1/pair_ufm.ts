/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_ufm.cpp and
 * pair_ufm.h (pair_style ufm, Contributing author: Rodolfo Paula Leite, Unicamp/Brazil). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * pair_style ufm cutoff; pair_coeff I J epsilon sigma [cutoff] (docs.lammps.org/pair_ufm.html): the Uniform Force Field,
 *   E = -epsilon ln(1 - exp(-(r/sigma)^2)) - offset,   r < rc
 * The pair mixes epsilon, sigma and the cutoff with the pair_modify mix rule when it is not set.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { fmtCoeff, parseNum } from '../../engine/force/util';
import { pow } from '../libm/pow';
import { exp } from '../libm/exp';
import { log } from '../libm/log';
import { setPairRange, resetSetCutoffs, table, mirror, mixDistance, mixEnergy } from './common';

export class PairUFM extends Pair {
  readonly name: string = 'ufm';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  uf1 = new Float64Array(0); uf2 = new Float64Array(0); uf3 = new Float64Array(0); offset = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    // native settings() resets the cutoffs of the pairs with j > i only
    resetSetCutoffs(this.p, 'cut', this.cutGlobal, this.ntypes, (i) => i + 1);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'scale', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 4 || args.length > 5) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const cutOne = args.length === 5 ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    setPairRange(this.p, args[0], args[1], [eps, sig, 1.0, cutOne], 'Incorrect args for pair coefficients', this.ntypes);
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'tail');
    if (this.uf1.length !== (this.ntypes + 1) ** 2) {
      this.uf1 = table(this.ntypes); this.uf2 = table(this.ntypes); this.uf3 = table(this.ntypes); this.offset = table(this.ntypes);
    }
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
      p.setMixed(i, j, 'epsilon', mixEnergy(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'sigma', mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
      p.setMixed(i, j, 'scale', 1.0);
    }
    const nt = this.ntypes + 1;
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), cut = p.get('cut', i, j);
    const k = i * nt + j;
    this.uf1[k] = 2.0 * eps / pow(sig, 2.0);
    this.uf2[k] = 1.0 / pow(sig, 2.0);
    this.uf3[k] = eps;
    if (this.shift) {
      const ratio = pow(cut / sig, 2.0);
      this.offset[k] = -eps * log(1.0 - exp(-ratio));
    } else this.offset[k] = 0.0;
    for (const t of [this.uf1, this.uf2, this.uf3, this.offset]) mirror(t, nt, i, j);
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const scale = this.p.p('scale');
    const { uf1, uf2, uf3, offset } = this;
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
        const expuf = exp(-rsq * uf2[t]);
        const fpair = factor * scale[t] * uf1[t] * expuf / (1.0 - expuf);
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        let e = -uf3[t] * log(1.0 - expuf) - offset[t];
        e *= factor;
        evdwl0 += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl0;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    const expuf = exp(-rsq * this.uf2[t]);
    const fforce = factorLJ * this.uf1[t] * expuf / (1.0 - expuf);
    const phiuf = -this.uf3[t] * log(1.0 - expuf) - this.offset[t];
    return { eng: factorLJ * phiuf, fforce };
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const g = (n: string) => fmtCoeff(this.p.get(n, i, j));
        out.push(`${i} ${j} ${g('epsilon')} ${g('sigma')} ${g('cut')}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    // pair_ufm.cpp extract(): epsilon, sigma, scale (dim 2)
    if (name === 'epsilon' || name === 'sigma' || name === 'scale') return this.p.p(name);
    return undefined;
  }
}
