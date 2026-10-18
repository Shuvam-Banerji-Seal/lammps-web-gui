/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_wf_cut.cpp and
 * pair_wf_cut.h (pair_style wf/cut, Contributing authors: Xipeng Wang, Simon Ramirez-Hinestrosa). This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * pair_style wf/cut cutoff; pair_coeff I J epsilon sigma nu mu [cutoff] (docs.lammps.org/pair_wf_cut.html): the Wang-Frenkel
 * potential, a generalised (2nu, 2mu) form that vanishes smoothly at the cutoff rc,
 *   E = e0nm ((sigma^2mu / r^2mu) - 1) ((rc^2mu / r^2mu) - 1)^2nu   r < rc
 * with e0nm = eps 2 nu (rc/sigma)^2mu ((1 + 2nu) / (2 nu) / ((rc/sigma)^2mu - 1))^(2nu+1). Native has no mixing and no
 * shift or tail term.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { fmtCoeff, parseInt_, parseNum } from '../../engine/force/util';
import { powint, setPairRange, resetSetCutoffs, table, mirror } from './common';

export class PairWFCut extends Pair {
  readonly name: string = 'wf/cut';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  nm = new Float64Array(0); e0nm = new Float64Array(0); rcmu = new Float64Array(0); sigmaMu = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    resetSetCutoffs(this.p, 'cut', this.cutGlobal, this.ntypes);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'nu', 'mu', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 6 || args.length > 7) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const nu = parseInt_(args[4], 'nu');
    const mu = parseInt_(args[5], 'mu');
    const cutOne = args.length === 7 ? parseNum(args[6], 'cutoff') : this.cutGlobal;
    setPairRange(this.p, args[0], args[1], [eps, sig, nu, mu, cutOne], 'Incorrect args for pair coefficients', this.ntypes);
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    if (this.nm.length !== (this.ntypes + 1) ** 2) {
      this.nm = table(this.ntypes); this.e0nm = table(this.ntypes); this.rcmu = table(this.ntypes); this.sigmaMu = table(this.ntypes);
    }
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    // pair_wf_cut.cpp init_one(): "All pair coeffs are not set" (no mixing)
    if (!p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.ntypes + 1;
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), nu = p.get('nu', i, j), mu = p.get('mu', i, j);
    const cut = p.get('cut', i, j);
    const k = i * nt + j;
    this.nm[k] = nu * mu;
    this.e0nm[k] = eps * 2.0 * nu * powint(cut / sig, 2 * mu)
      * powint((1 + 2.0 * nu) / (2.0 * nu) / (powint(cut / sig, 2 * mu) - 1.0), 2 * nu + 1);
    this.rcmu[k] = powint(cut, 2 * mu);
    this.sigmaMu[k] = powint(sig, 2 * mu);
    for (const t of [this.nm, this.e0nm, this.rcmu, this.sigmaMu]) mirror(t, nt, i, j);
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const nu = this.p.p('nu'), mu = this.p.p('mu');
    const { nm, e0nm, rcmu, sigmaMu } = this;
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
        const r2inv = 1.0 / rsq;
        const rminv = powint(r2inv, mu[t]);
        const rm = sigmaMu[t] * rminv - 1.0;
        const rn = rcmu[t] * rminv - 1.0;
        const forcenm = 2.0 * mu[t] * sigmaMu[t] * powint(rn, 2 * nu[t])
          + 4.0 * nm[t] * rcmu[t] * rm * powint(rn, 2 * nu[t] - 1);
        const fpair = factor * e0nm[t] * forcenm * powint(r2inv, mu[t] + 1);
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        let e = e0nm[t] * (rm * powint(rn, 2 * nu[t]));
        e *= factor;
        evdwl0 += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl0;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    const nu = this.p.p('nu')[t], mu = this.p.p('mu')[t];
    const r2inv = 1.0 / rsq;
    const rminv = powint(r2inv, mu);
    const rm = this.sigmaMu[t] * rminv - 1.0;
    const rn = this.rcmu[t] * rminv - 1.0;
    const forcenm = 2.0 * mu * this.sigmaMu[t] * powint(rn, 2 * nu) + 4.0 * this.nm[t] * this.rcmu[t] * rm * powint(rn, 2 * nu - 1);
    const fforce = factorLJ * this.e0nm[t] * forcenm * powint(r2inv, mu + 1);
    const phinm = this.e0nm[t] * rm * powint(rn, 2 * nu);
    return { eng: factorLJ * phinm, fforce };
  }

  dataCoeffs(): string[] | null {
    // "%d %g %g %d %d": nu and mu are integers
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))} ${this.p.get('nu', i, i)} ${this.p.get('mu', i, i)}`);
    }
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const g = (n: string) => fmtCoeff(this.p.get(n, i, j));
        out.push(`${i} ${j} ${g('epsilon')} ${g('sigma')} ${this.p.get('nu', i, j)} ${this.p.get('mu', i, j)} ${g('cut')}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    // pair_wf_cut.cpp extract(): epsilon, sigma, nu, mu (dim 2)
    if (name === 'epsilon' || name === 'sigma' || name === 'nu' || name === 'mu') return this.p.p(name);
    return undefined;
  }
}
