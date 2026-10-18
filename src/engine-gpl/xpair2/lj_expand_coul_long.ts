/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_lj_expand_coul_long.cpp
 * and pair_lj_expand_coul_long.h (PairLJExpandCoulLong: pair_style lj/expand/coul/long). The source names its author
 * (Axel Kohlmeyer, Temple University, after pair_lj_cut_coul_long.cpp). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The 12-6 Lennard-Jones potential of the shifted distance r - delta, with the Ewald real-space Coulomb term
 * (docs.lammps.org/pair_lj_expand.html: "E = 4 epsilon [ (sigma/(r - delta))^12 - (sigma/(r - delta))^6 ]
 * r < r_c + delta"; coul/long as in pair_coul.html). The Coulomb part is the engine's coul/long kernel
 * (src/engine/force/pair/coul_long.ts), with the erfc table of pair_modify table N.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, type PairSep } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum, fmtCoeff } from '../../engine/force/util';
import { hasCharge } from '../../engine/atoms';
import { ErfcTableCache, type ErfcTable } from '../../engine/force/erfc';
import { coulLongPair } from '../../engine/force/pair/coul_long';
import { pow } from '../libm/pow';
import { forEachTypePair, mixDistanceNative, mixEnergyNative } from './common';

export class PairLJExpandCoulLong extends Pair {
  readonly name: string = 'lj/expand/coul/long';
  virialFdotr = true;
  coulLong = true;
  cutLJGlobal = 0;
  cutCoul = 0;
  private p!: PairParams;
  private cutLJsq = new Float64Array(0);
  private lj1 = new Float64Array(0);
  private lj2 = new Float64Array(0);
  private lj3 = new Float64Array(0);
  private lj4 = new Float64Array(0);
  private offsetE = new Float64Array(0);
  private qqrd2eLast = 1;
  private cutCoulsq = 0;
  private readonly erfcTables = new ErfcTableCache();
  /** pair_modify table N (null for 0): erfc.ts makeErfcTable. */
  private erfcTable(g: number): ErfcTable | null { return this.erfcTables.get(this.table, g, this.cutCoul * this.cutCoul, this.tabinnerRsq); }

  settings(args: string[]): void {
    // pair_lj_expand_coul_long.cpp settings(): one or two arguments
    if (args.length < 1 || args.length > 2) throw new StyleError('Illegal pair_style command');
    this.cutLJGlobal = parseNum(args[0], 'cutoff');
    this.cutCoul = args.length === 1 ? this.cutLJGlobal : parseNum(args[1], 'coulomb cutoff');
    // native settings(): every explicitly set pair takes the new global LJ cutoff
    if (this.p) for (let i = 1; i <= this.ntypes; i++) for (let j = i; j <= this.ntypes; j++) if (this.p.isSet(i, j)) this.p.setMixed(i, j, 'cut_lj', this.cutLJGlobal);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'delta', 'cut_lj']);
    const n = (ntypes + 1) * (ntypes + 1);
    this.cutLJsq = new Float64Array(n);
    this.lj1 = new Float64Array(n); this.lj2 = new Float64Array(n);
    this.lj3 = new Float64Array(n); this.lj4 = new Float64Array(n);
    this.offsetE = new Float64Array(n);
  }

  coeff(args: string[]): void {
    // pair_lj_expand_coul_long.cpp coeff(): narg 5 or 6 (I J epsilon sigma delta [cutoff])
    if (args.length < 5 || args.length > 6) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const delta = parseNum(args[4], 'shift');
    const cutLJ = args.length === 6 ? parseNum(args[5], 'cutoff') : this.cutLJGlobal;
    forEachTypePair(args[0], args[1], this.ntypes, (i, j) => this.p.set(i, j, [eps, sig, delta, cutLJ]));
  }

  initStyle(ctx: StyleContext): void {
    // init_style(): the charge attribute (the native message names lj/cut/coul/long, as the source does)
    if (!ctx.s || !hasCharge(ctx.s)) throw new StyleError('Pair style lj/cut/coul/long requires atom attribute q');
    this.cutCoulsq = this.cutCoul * this.cutCoul;
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
      const eps = mixEnergyNative(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j));
      p.setMixed(i, j, 'epsilon', eps);
      p.setMixed(i, j, 'sigma', mixDistanceNative(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'cut_lj', mixDistanceNative(this.mix, p.get('cut_lj', i, i), p.get('cut_lj', j, j)));
      p.setMixed(i, j, 'delta', 0.5 * (p.get('delta', i, i) + p.get('delta', j, j)));
    }
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), cutLJ = p.get('cut_lj', i, j), shift = p.get('delta', i, j);
    // "include TIP4P qdist in full cutoff, qdist = 0.0 if not TIP4P"
    const cut = Math.max(cutLJ + shift, this.cutCoul + 2.0 * 0.0);
    const nt = this.ntypes + 1;
    const k1 = i * nt + j, k2 = j * nt + i;
    this.cutLJsq[k1] = (cutLJ + shift) * (cutLJ + shift);
    this.lj1[k1] = 48.0 * eps * pow(sig, 12.0);
    this.lj2[k1] = 24.0 * eps * pow(sig, 6.0);
    this.lj3[k1] = 4.0 * eps * pow(sig, 12.0);
    this.lj4[k1] = 4.0 * eps * pow(sig, 6.0);
    if (this.shift && cutLJ > 0.0) {
      const ratio = sig / cutLJ;
      this.offsetE[k1] = 4.0 * eps * (pow(ratio, 12.0) - pow(ratio, 6.0));
    } else this.offsetE[k1] = 0.0;
    this.cutLJsq[k2] = this.cutLJsq[k1];
    this.lj1[k2] = this.lj1[k1]; this.lj2[k2] = this.lj2[k1];
    this.lj3[k2] = this.lj3[k1]; this.lj4[k2] = this.lj4[k1];
    this.offsetE[k2] = this.offsetE[k1];
    p.setMixed(j, i, 'epsilon', eps); p.setMixed(j, i, 'sigma', sig);
    p.setMixed(j, i, 'cut_lj', cutLJ); p.setMixed(j, i, 'delta', shift);
    return cut;
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    let etail = 0, ptail = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = i; j < nt; j++) {
        const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
        const rc1 = this.p.get('cut_lj', i, j), shift1 = this.p.get('delta', i, j);
        const ni = count[i], nj = count[j];
        // etail_ij / ptail_ij of init_one (pair_lj_expand_coul_long.cpp at b332f28)
        const sig2 = sig * sig;
        const sig6 = sig2 * sig2 * sig2;
        const rc2 = rc1 * rc1;
        const rc3 = rc2 * rc1;
        const rc6 = rc3 * rc3;
        const rc9 = rc3 * rc6;
        const shift2 = shift1 * shift1;
        const shift3 = shift2 * shift1;
        const etailIJ = 8.0 * Math.PI * ni * nj * eps * sig6 *
          ((1.0 / 9.0 + 2.0 * shift1 / (10.0 * rc1) + shift2 / (11.0 * rc2)) * sig6 / rc9 -
           (1.0 / 3.0 + 2.0 * shift1 / (4.0 * rc1) + shift2 / (5.0 * rc2)) / rc3);
        const ptailIJ = 16.0 * Math.PI * ni * nj * eps * sig6 *
          ((1.0 / 9.0 + 3.0 * shift1 / (10.0 * rc1) +
            3.0 * shift2 / (11.0 * rc2) + shift3 / (12.0 * rc3)) * 2.0 * sig6 / rc9 -
           (1.0 / 3.0 + 3.0 * shift1 / (4.0 * rc1) +
            3.0 * shift2 / (5.0 * rc2) + shift3 / (6.0 * rc3)) / rc3);
        etail += etailIJ; ptail += ptailIJ;
        if (i !== j) { etail += etailIJ; ptail += ptailIJ; }
      }
    }
    return { etail, ptail };
  }

  compute(pc: PairCompute): void {
    this.qqrd2eLast = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { lj1, lj2, lj3, lj4, offsetE, cutLJsq, cutCoulsq } = this;
    const shiftA = this.p.p('delta');
    const { cutsq } = this;
    const g = this.gEwald;
    const tab = this.erfcTable(g);
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
        const jtype = type[j];
        const t = itype * nt + jtype;
        if (rsq < cutsq[t]) {
          let forcecoul = 0.0, ec = 0.0;
          if (rsq < cutCoulsq) {
            const c = coulLongPair(rsq, qtmp, q[j], g, pc.qqrd2e, factorCoul, this.table === 0, tab);
            forcecoul = c.f;
            ec = c.e;
          }
          let forcelj = 0.0, r6inv = 0.0, e = 0.0;
          if (rsq < cutLJsq[t]) {
            const r = Math.sqrt(rsq);
            const rshift = r - shiftA[t];
            const rshiftsq = rshift * rshift;
            const rshift2inv = 1.0 / rshiftsq;
            r6inv = rshift2inv * rshift2inv * rshift2inv;
            forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
            forcelj *= factorLJ / rshift / r;
            e = r6inv * (lj3[t] * r6inv - lj4[t]) - offsetE[t];
            e *= factorLJ;
          }
          // coulLongPair gives forcecoul / rsq, so the Coulomb term enters without the r2inv factor of compute()
          const fpair = forcecoul + forcelj;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
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
    const t = itype * (this.ntypes + 1) + jtype;
    let forcecoul = 0.0, phicoul = 0.0;
    if (rsq < this.cutCoulsq) {
      const c = coulLongPair(rsq, qi, qj, this.gEwald, this.qqrd2eLast, factorCoul, this.table === 0, this.erfcTable(this.gEwald));
      forcecoul = c.f;
      phicoul = c.e;
    }
    let forcelj = 0.0, r6inv = 0.0, philj = 0.0;
    if (rsq < this.cutLJsq[t]) {
      const r = Math.sqrt(rsq);
      const rshift = r - this.p.get('delta', itype, jtype);
      const rshiftsq = rshift * rshift;
      const rshift2inv = 1.0 / rshiftsq;
      r6inv = rshift2inv * rshift2inv * rshift2inv;
      forcelj = r6inv * (this.lj1[t] * r6inv - this.lj2[t]);
      forcelj *= factorLJ / rshift / r;
      philj = r6inv * (this.lj3[t] * r6inv - this.lj4[t]) - this.offsetE[t];
    }
    const fforce = forcecoul + forcelj;
    let eng = phicoul;
    if (rsq < this.cutLJsq[t]) eng += factorLJ * philj;
    return { eng, fforce };
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))} ${fmtCoeff(this.p.get('delta', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('epsilon', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))} ${fmtCoeff(this.p.get('delta', i, j))} ${fmtCoeff(this.p.get('cut_lj', i, j))}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    if (name === 'epsilon' || name === 'sigma') return this.p.p(name);
    if (name === 'delta') return this.p.p('delta');
    return undefined;
  }
}
