/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/FEP/pair_lj_cut_soft_gapsys.cpp, pair_lj_cut_soft_gapsys.h, pair_coul_cut_soft_gapsys.cpp and
 * pair_coul_cut_soft_gapsys.h (classes PairLJCutSoftGapsys and PairCoulCutSoftGapsys, styles lj/cut/soft/gapsys and
 * coul/cut/soft/gapsys) at commit b332f28, by the LAMMPS developers (FEP package, Copyright Sandia Corporation).
 * GNU General Public License. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style lj/cut/soft/gapsys alpha_LJ cutoff; pair_coeff I J epsilon sigma lambda [cutoff]
 * pair_style coul/cut/soft/gapsys sigma_C alpha_C cutoff; pair_coeff I J lambda [cutoff]
 * (docs.lammps.org/pair_lj_cut_soft_gapsys.html and pair_coul_cut_soft_gapsys.html).
 *
 * Inside a soft radius the potential is replaced by a quadratic-plus-constant smoothing, outside the LJ or
 * Coulomb form applies. lj/cut/soft/gapsys: r_inner = alpha_LJ (26 lambda / 7)^(1/6) sigma; for r < r_inner
 *   E = a1 r^2 + a2 r + a3 - offset,  a1 = 12 eps (26 s12 - 7 s6) / r_inner^2,
 *   a2 = -96 eps (7 s12 - 2 s6) / r_inner,  a3 = 28 eps (13 s12 - 4 s6),  s6 = (sigma/r_inner)^6;
 * for r_inner <= r < rc the plain 12-6 LJ energy minus the offset. The force is the derivative of that form.
 * coul/cut/soft/gapsys: r_inner = (1 + sigma_C |q_i q_j|) alpha_C lambda^(1/6); for r < r_inner
 *   E = C q_i q_j (r^2 / r_inner^3 - 3 r / r_inner^2 + 3 / r_inner), and C q_i q_j / r otherwise (rc cutoff).
 * The lambda of lj/cut/soft/gapsys enters only through r_inner. Source notes: mixing of epsilon and sigma uses the
 * pair_modify mix rule; lambda of the two types must agree for a mixed pair ("different lambda values in mix");
 * pair_modify tail is refused (the source's tail sum is not reproduced here); pair_modify shift subtracts the LJ
 * offset from both regions.
 *
 * Measured with native LAMMPS (black box): tests/oracle/w69fep_gapsys_lj.in and w69fep_gapsys_coul.in.
 */
import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { pow as libmPow } from '../libm/pow';

export class PairLJCutSoftGapsys extends Pair {
  readonly name = 'lj/cut/soft/gapsys';
  virialFdotr = true;
  private alphaLJ = 0;
  private cutGlobal = 0;
  p!: PairParams;
  private eps = new Float64Array(0);
  private sig = new Float64Array(0);
  private cutsqT = new Float64Array(0);
  private innerSq = new Float64Array(0);
  private innerT = new Float64Array(0);
  private lj1 = new Float64Array(0);
  private lj2 = new Float64Array(0);
  private lj3 = new Float64Array(0);
  private lj4 = new Float64Array(0);
  private offT = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('usage: pair_style lj/cut/soft/gapsys alpha_LJ cutoff');
    this.alphaLJ = parseNum(args[0], 'alpha_LJ');
    if (this.alphaLJ < 0) throw new StyleError('Pair style lj/cut/soft/gapsys requires alphalj > 0');
    this.cutGlobal = parseNum(args[1], 'cutoff');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'lambda', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 5 || args.length > 6) throw new StyleError('usage: pair_coeff I J epsilon sigma lambda [cutoff] for pair style lj/cut/soft/gapsys');
    const eps = parseNum(args[2], 'epsilon'), sig = parseNum(args[3], 'sigma'), lam = parseNum(args[4], 'lambda');
    if (sig <= 0 || eps <= 0) throw new StyleError('Incorrect args for pair coefficients');
    if (lam < 0 || lam > 1) throw new StyleError('Pair style lj/cut/soft/gapsys requires 0 < lambda < 1');
    const cut = args.length === 6 ? parseNum(args[5], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, lam, cut]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.tail) throw new StyleError('pair_modify tail yes is not supported for pair style lj/cut/soft/gapsys');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j}, pair style lj/cut/soft/gapsys)`);
      if (p.get('lambda', i, i) !== p.get('lambda', j, j)) throw new StyleError('Pair lj/cut/soft/gapsys different lambda values in mix');
      p.setMixed(i, j, 'epsilon', mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'sigma', mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'lambda', p.get('lambda', i, i));
      p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    }
    const nt = this.ntypes + 1;
    if (this.eps.length !== nt * nt) {
      const z = () => new Float64Array(nt * nt);
      this.eps = z(); this.sig = z(); this.cutsqT = z(); this.innerSq = z(); this.innerT = z();
      this.lj1 = z(); this.lj2 = z(); this.lj3 = z(); this.lj4 = z(); this.offT = z();
    }
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), lam = p.get('lambda', i, j), cut = p.get('cut', i, j);
    const lj1 = 48 * eps * libmPow(sig, 12), lj2 = 24 * eps * libmPow(sig, 6), lj3 = 4 * eps * libmPow(sig, 12), lj4 = 4 * eps * libmPow(sig, 6);
    let off = 0;
    if (this.shift && cut > 0) {
      const ratio = sig / cut;
      off = 4 * eps * (libmPow(ratio, 12) - libmPow(ratio, 6));
    }
    const rin = this.alphaLJ * libmPow(26 * lam / 7, 1 / 6) * sig;
    for (const [a, b] of [[i, j], [j, i]]) {
      const k = a * nt + b;
      this.eps[k] = eps; this.sig[k] = sig; this.cutsqT[k] = cut * cut;
      this.innerT[k] = rin; this.innerSq[k] = rin * rin;
      this.lj1[k] = lj1; this.lj2[k] = lj2; this.lj3[k] = lj3; this.lj4[k] = lj4; this.offT[k] = off;
    }
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const fl = sLJ[jj >>> SBBITS];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        let fpair = 0, e = 0;
        if (rsq < this.innerSq[t]) {
          const eps = this.eps[t], sig = this.sig[t], cin = this.innerT[t];
          const s6 = libmPow(sig / cin, 6), s12 = s6 * s6;
          const b1 = (-24 * eps * (26 * s12 - 7 * s6)) / (cin * cin);
          const b2 = (96 * eps * (7 * s12 - 2 * s6)) / cin;
          fpair = fl * (b1 + b2 / Math.sqrt(rsq));
          const a1 = (12 * eps * (26 * s12 - 7 * s6)) / (cin * cin);
          const a2 = (-96 * eps * (7 * s12 - 2 * s6)) / cin;
          const a3 = 28 * eps * (13 * s12 - 4 * s6);
          e = fl * (a1 * rsq + a2 * Math.sqrt(rsq) + a3 - this.offT[t]);
        } else if (rsq < this.cutsqT[t]) {
          const r2inv = 1 / rsq;
          const r6inv = r2inv * r2inv * r2inv;
          fpair = fl * r6inv * (this.lj1[t] * r6inv - this.lj2[t]) * r2inv;
          e = fl * (r6inv * (this.lj3[t] * r6inv - this.lj4[t]) - this.offT[t]);
        } else continue;
        evdwl += e;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl;
  }
}

export class PairCoulCutSoftGapsys extends Pair {
  readonly name = 'coul/cut/soft/gapsys';
  virialFdotr = true;
  private sigmaQ = 0;
  private alphaQ = 0;
  private cutGlobal = 0;
  p!: PairParams;
  private lamT = new Float64Array(0);
  private cutsqT = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 3) throw new StyleError('usage: pair_style coul/cut/soft/gapsys sigma_C alpha_C cutoff');
    this.sigmaQ = parseNum(args[0], 'sigma_C');
    this.alphaQ = parseNum(args[1], 'alpha_C');
    this.cutGlobal = parseNum(args[2], 'cutoff');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['lambda', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 3 || args.length > 4) throw new StyleError('usage: pair_coeff I J lambda [cutoff] for pair style coul/cut/soft/gapsys');
    const cut = args.length === 4 ? parseNum(args[3], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [parseNum(args[2], 'lambda'), cut]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.tail) throw new StyleError('pair_modify tail yes is not supported for pair style coul/cut/soft/gapsys');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j}, pair style coul/cut/soft/gapsys)`);
      if (p.get('lambda', i, i) !== p.get('lambda', j, j)) throw new StyleError('Pair coul/cut/soft/gapsys different lambda values in mix');
      p.setMixed(i, j, 'lambda', p.get('lambda', i, i));
      p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    }
    const nt = this.ntypes + 1;
    if (this.lamT.length !== nt * nt) {
      this.lamT = new Float64Array(nt * nt); this.cutsqT = new Float64Array(nt * nt);
    }
    const lam = p.get('lambda', i, j), cut = p.get('cut', i, j);
    for (const [a, b] of [[i, j], [j, i]]) {
      const k = a * nt + b;
      this.lamT[k] = lam; this.cutsqT[k] = cut * cut;
    }
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const qqrd2e = pc.qqrd2e;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const fc = sC[jj >>> SBBITS];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        const qq = qi * q[j];
        let fpair = 0, e = 0;
        // r_inner = (1 + sigma_C |q_i q_j|) alpha_C lambda^(1/6), the source's cut_inner
        const cin = (1 + this.sigmaQ * Math.abs(qq)) * this.alphaQ * libmPow(this.lamT[t], 1 / 6);
        if (rsq < cin * cin) {
          const forcecoul = fc * qqrd2e * qq;
          fpair = (-2 / libmPow(cin, 3) + 3 / (cin ** 2 * Math.sqrt(rsq))) * forcecoul;
          e = forcecoul * (rsq / libmPow(cin, 3) - (3 * Math.sqrt(rsq)) / cin ** 2 + 3 / cin);
        } else if (rsq < this.cutsqT[t]) {
          const r2inv = 1 / rsq;
          const rinv = Math.sqrt(r2inv);
          const forcecoul = qqrd2e * qq * rinv;
          fpair = fc * forcecoul * r2inv;
          e = fc * qqrd2e * qq * rinv;
        } else continue;
        ecoul += e;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.ecoul += ecoul;
  }
}
