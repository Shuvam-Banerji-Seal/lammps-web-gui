/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_lj_mdf.cpp,
 * pair_lennard_mdf.cpp and pair_buck_mdf.cpp (and their .h files; Contributing author: Paolo Raiteri, Curtin University).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The MDF (Mei, Davenport and Fernando) pair styles of docs.lammps.org/pair_mdf.html: the 12-6 Lennard-Jones or
 * Buckingham potential is multiplied by a taper that is 1 inside the inner cutoff and falls smoothly to 0 at the outer
 * one:
 *   d = (r - rin) / (rc - rin),  tt = (1 + 3d + 6d^2)(1 - d)^3,  taper(r) = tt
 * The force adds the derivative of the taper (dt = 30 d^2 (1 - d)^2 r / (rc - rin)) times the unmodified potential.
 * Native has no offset and no tail term for these styles.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { fmtCoeff, parseNum } from '../../engine/force/util';
import { pow } from '../libm/pow';
import { exp } from '../libm/exp';
import { setPairRange, resetSetCutoffs, table, mirror, mixDistance, mixEnergy } from './common';

/** pair_style lj/mdf and lennard/mdf: "pair_style style inner outer" with 0 < inner <= outer (settings()). */
const mdfSettings = (self: Pair & { cutGlobal: number; cutInnerGlobal: number; p: PairParams; ntypes: number }, args: string[]): void => {
  if (args.length !== 2) throw new StyleError('Illegal pair_style command');
  self.cutInnerGlobal = parseNum(args[0], 'inner cutoff');
  self.cutGlobal = parseNum(args[1], 'outer cutoff');
  if (self.cutInnerGlobal <= 0.0 || self.cutInnerGlobal > self.cutGlobal) throw new StyleError('Illegal pair_style command');
  resetSetCutoffs(self.p, 'cut_inner', self.cutInnerGlobal, self.ntypes);
  resetSetCutoffs(self.p, 'cut', self.cutGlobal, self.ntypes);
};

/** The taper factor tt and its derivative term dt of native compute(): the same operation order. */
const taperOf = (r: number, ci: number, c: number): { tt: number; dt: number; dd: number; d: number; dp: number } => {
  const dp = c - ci;
  const d = (r - ci) / dp;
  const dd = 1. - d;
  const tt = (1. + 3. * d + 6. * d * d) * dd * dd * dd;
  const dt = 30. * d * d * dd * dd * r / dp;
  return { tt, dt, dd, d, dp };
};

/** Shared lj/mdf and lennard/mdf loop: the 12-6 form with the lj1..lj4 coefficient tables. */
const mdfLJCompute = (self: MdfLJBase, pc: PairCompute): void => {
  const list = pc.half!;
  const { x, f, type } = pc;
  const nt = self.ntypes + 1;
  const cutsq = self.cutsq;
  const { lj1, lj2, lj3, lj4, ciSq, ci, cc } = self;
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
      const r2inv = 1.0 / rsq;
      const r6inv = r2inv * r2inv * r2inv;
      let forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
      let tt = 1;
      if (rsq > ciSq[t]) {
        const philj = r6inv * (lj3[t] * r6inv - lj4[t]);
        const rr = Math.sqrt(rsq);
        const g = taperOf(rr, ci[t], cc[t]);
        tt = g.tt;
        forcelj = forcelj * tt + philj * g.dt;
      }
      const fpair = factor * forcelj * r2inv;
      f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
      f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
      let e = r6inv * (lj3[t] * r6inv - lj4[t]);
      if (rsq > ciSq[t]) e *= tt;
      e *= factor;
      evdwl += e;
      if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
    }
  }
  pc.acc.evdwl += evdwl;
};

/** Shared single() of the 12-6 MDF styles. */
const mdfLJSingle = (self: MdfLJBase, itype: number, jtype: number, rsq: number, factorLJ: number): { eng: number; fforce: number } => {
  const t = itype * (self.ntypes + 1) + jtype;
  const r2inv = 1.0 / rsq;
  const r6inv = r2inv * r2inv * r2inv;
  let philj = r6inv * (self.lj3[t] * r6inv - self.lj4[t]);
  let forcelj = r6inv * (self.lj1[t] * r6inv - self.lj2[t]);
  if (rsq > self.ciSq[t]) {
    const rr = Math.sqrt(rsq);
    const g = taperOf(rr, self.ci[t], self.cc[t]);
    forcelj = forcelj * g.tt + philj * g.dt;
    philj *= g.tt;
  }
  return { eng: factorLJ * philj, fforce: factorLJ * forcelj * r2inv };
};

interface MdfLJBase {
  ntypes: number;
  cutsq: Float64Array;
  lj1: Float64Array; lj2: Float64Array; lj3: Float64Array; lj4: Float64Array;
  ciSq: Float64Array; ci: Float64Array; cc: Float64Array;
}

/** pair_style lj/mdf inner outer; pair_coeff I J epsilon sigma [inner outer] (docs.lammps.org/pair_mdf.html). */
export class PairLJMDF extends Pair implements MdfLJBase {
  readonly name: string = 'lj/mdf';
  virialFdotr = true;
  cutGlobal = 0;
  cutInnerGlobal = 0;
  p!: PairParams;
  lj1 = new Float64Array(0); lj2 = new Float64Array(0); lj3 = new Float64Array(0); lj4 = new Float64Array(0);
  ciSq = new Float64Array(0); ci = new Float64Array(0); cc = new Float64Array(0);

  settings(args: string[]): void { mdfSettings(this, args); }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cut_inner', 'cut']);
  }

  coeff(args: string[]): void {
    // pair_lennard_mdf.cpp (and pair_lj_mdf.cpp): narg 4 or 6; the inner and outer cutoffs are the global pair in the 4-word form
    if (args.length !== 4 && args.length !== 6) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon'), sig = parseNum(args[3], 'sigma');
    let ci = this.cutInnerGlobal, c = this.cutGlobal;
    if (args.length === 6) { ci = parseNum(args[4], 'inner cutoff'); c = parseNum(args[5], 'outer cutoff'); }
    if (this.cutInnerGlobal <= 0.0 || this.cutInnerGlobal > this.cutGlobal) throw new StyleError('Illegal pair_style command');
    setPairRange(this.p, args[0], args[1], [eps, sig, ci, c], 'Incorrect args for pair coefficients', this.ntypes);
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    const n = (this.ntypes + 1) ** 2;
    if (this.lj1.length !== n) {
      this.lj1 = table(this.ntypes); this.lj2 = table(this.ntypes); this.lj3 = table(this.ntypes); this.lj4 = table(this.ntypes);
      this.ciSq = table(this.ntypes); this.ci = table(this.ntypes); this.cc = table(this.ntypes);
    }
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
      const eps = mixEnergy(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j));
      p.setMixed(i, j, 'epsilon', eps);
      p.setMixed(i, j, 'sigma', mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'cut_inner', mixDistance(this.mix, p.get('cut_inner', i, i), p.get('cut_inner', j, j)));
      p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    }
    const nt = this.ntypes + 1;
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j);
    const ci = p.get('cut_inner', i, j), c = p.get('cut', i, j);
    const k = i * nt + j;
    const s12 = pow(sig, 12.0), s6 = pow(sig, 6.0);
    this.lj1[k] = 48.0 * eps * s12;
    this.lj2[k] = 24.0 * eps * s6;
    this.lj3[k] = 4.0 * eps * s12;
    this.lj4[k] = 4.0 * eps * s6;
    this.ciSq[k] = ci * ci;
    this.ci[k] = ci;
    this.cc[k] = c;
    for (const t of [this.lj1, this.lj2, this.lj3, this.lj4, this.ciSq, this.ci, this.cc]) mirror(t, nt, i, j);
    return c;
  }

  compute(pc: PairCompute): void { mdfLJCompute(this, pc); }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number) {
    return mdfLJSingle(this, itype, jtype, rsq, factorLJ);
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
        out.push(`${i} ${j} ${g('epsilon')} ${g('sigma')} ${g('cut_inner')} ${g('cut')}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    // pair_lj_mdf.cpp extract(): epsilon, sigma (dim 2)
    if (name === 'epsilon' || name === 'sigma') return this.p.p(name);
    return undefined;
  }
}

/** pair_style lennard/mdf inner outer; pair_coeff I J A B [inner outer] (no mixing, no write_data in native). */
export class PairLennardMDF extends Pair implements MdfLJBase {
  readonly name: string = 'lennard/mdf';
  virialFdotr = true;
  cutGlobal = 0;
  cutInnerGlobal = 0;
  p!: PairParams;
  lj1 = new Float64Array(0); lj2 = new Float64Array(0); lj3 = new Float64Array(0); lj4 = new Float64Array(0);
  ciSq = new Float64Array(0); ci = new Float64Array(0); cc = new Float64Array(0);

  settings(args: string[]): void { mdfSettings(this, args); }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['aparm', 'bparm', 'cut_inner', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length !== 4 && args.length !== 6) throw new StyleError('Incorrect args for pair coefficients');
    const a = parseNum(args[2], 'A'), b = parseNum(args[3], 'B');
    let ci = this.cutInnerGlobal, c = this.cutGlobal;
    if (args.length === 6) { ci = parseNum(args[4], 'inner cutoff'); c = parseNum(args[5], 'outer cutoff'); }
    if (ci <= 0.0 || ci > c) throw new StyleError('Illegal pair_coeff command');
    setPairRange(this.p, args[0], args[1], [a, b, ci, c], 'Incorrect args for pair coefficients', this.ntypes);
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    if (this.lj1.length !== (this.ntypes + 1) ** 2) {
      this.lj1 = table(this.ntypes); this.lj2 = table(this.ntypes); this.lj3 = table(this.ntypes); this.lj4 = table(this.ntypes);
      this.ciSq = table(this.ntypes); this.ci = table(this.ntypes); this.cc = table(this.ntypes);
    }
  }

  initOne(i: number, j: number): number {
    // pair_lennard_mdf.cpp init_one(): "All pair coeffs are not set" for an unset pair (no mixing)
    if (!this.p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.ntypes + 1;
    const a = this.p.get('aparm', i, j), b = this.p.get('bparm', i, j);
    const ci = this.p.get('cut_inner', i, j), c = this.p.get('cut', i, j);
    const k = i * nt + j;
    this.ciSq[k] = ci * ci;
    this.lj1[k] = 12.0 * a;
    this.lj2[k] = 6.0 * b;
    this.lj3[k] = a;
    this.lj4[k] = b;
    this.ci[k] = ci;
    this.cc[k] = c;
    for (const t of [this.lj1, this.lj2, this.lj3, this.lj4, this.ciSq, this.ci, this.cc]) mirror(t, nt, i, j);
    return c;
  }

  compute(pc: PairCompute): void { mdfLJCompute(this, pc); }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number) {
    return mdfLJSingle(this, itype, jtype, rsq, factorLJ);
  }

  extract(name: string): unknown {
    // pair_lennard_mdf.cpp extract(): "a" -> aparm, "b" -> bparm
    if (name === 'a') return this.p.p('aparm');
    if (name === 'b') return this.p.p('bparm');
    return undefined;
  }
}

/** pair_style buck/mdf inner outer; pair_coeff I J A rho C [inner outer] (docs.lammps.org/pair_mdf.html). */
export class PairBuckMDF extends Pair {
  readonly name: string = 'buck/mdf';
  virialFdotr = true;
  cutGlobal = 0;
  cutInnerGlobal = 0;
  p!: PairParams;
  rhoinv = new Float64Array(0); buck1 = new Float64Array(0); buck2 = new Float64Array(0);
  ciSq = new Float64Array(0); ci = new Float64Array(0); cc = new Float64Array(0);

  settings(args: string[]): void {
    // pair_buck_mdf.cpp settings(): two words, no range check here (the coeff() check applies)
    if (args.length !== 2) throw new StyleError('Illegal pair_style command');
    this.cutInnerGlobal = parseNum(args[0], 'inner cutoff');
    this.cutGlobal = parseNum(args[1], 'outer cutoff');
    resetSetCutoffs(this.p, 'cut', this.cutGlobal, this.ntypes);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['a', 'rho', 'c', 'cut_inner', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length !== 5 && args.length !== 7) throw new StyleError('Incorrect args for pair coefficients');
    const a = parseNum(args[2], 'A');
    const rho = parseNum(args[3], 'rho');
    if (rho <= 0) throw new StyleError('Incorrect args for pair coefficients');
    const c = parseNum(args[4], 'C');
    let ci = this.cutInnerGlobal, cut = this.cutGlobal;
    if (args.length === 7) { ci = parseNum(args[5], 'inner cutoff'); cut = parseNum(args[6], 'outer cutoff'); }
    if (this.cutInnerGlobal <= 0.0 || this.cutInnerGlobal > this.cutGlobal) throw new StyleError('Illegal pair_style command');
    setPairRange(this.p, args[0], args[1], [a, rho, c, ci, cut], 'Incorrect args for pair coefficients', this.ntypes);
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    if (this.rhoinv.length !== (this.ntypes + 1) ** 2) {
      this.rhoinv = table(this.ntypes); this.buck1 = table(this.ntypes); this.buck2 = table(this.ntypes);
      this.ciSq = table(this.ntypes); this.ci = table(this.ntypes); this.cc = table(this.ntypes);
    }
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.ntypes + 1;
    const a = this.p.get('a', i, j), rho = this.p.get('rho', i, j), c = this.p.get('c', i, j);
    const ci = this.p.get('cut_inner', i, j), cut = this.p.get('cut', i, j);
    const k = i * nt + j;
    this.rhoinv[k] = 1.0 / rho;
    this.buck1[k] = a / rho;
    this.buck2[k] = 6.0 * c;
    this.ciSq[k] = ci * ci;
    this.ci[k] = ci;
    this.cc[k] = cut;
    for (const t of [this.rhoinv, this.buck1, this.buck2, this.ciSq, this.ci, this.cc]) mirror(t, nt, i, j);
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const A = this.p.p('a'), C = this.p.p('c');
    const { rhoinv, buck1, buck2, ciSq, ci, cc } = this;
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
        const r2inv = 1.0 / rsq;
        const r6inv = r2inv * r2inv * r2inv;
        const r = Math.sqrt(rsq);
        const rexp = exp(-r * rhoinv[t]);
        let forcebuck = buck1[t] * r * rexp - buck2[t] * r6inv;
        let tt = 1;
        if (rsq > ciSq[t]) {
          const phibuck = A[t] * rexp - C[t] * r6inv;
          const g = taperOf(r, ci[t], cc[t]);
          tt = g.tt;
          forcebuck = forcebuck * tt + phibuck * g.dt;
        }
        const fpair = factor * forcebuck * r2inv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        let e = A[t] * rexp - C[t] * r6inv;
        if (rsq > ciSq[t]) e *= tt;
        e *= factor;
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    const r2inv = 1.0 / rsq;
    const r6inv = r2inv * r2inv * r2inv;
    const r = Math.sqrt(rsq);
    const rexp = exp(-r * this.rhoinv[t]);
    let forcebuck = this.buck1[t] * r * rexp - this.buck2[t] * r6inv;
    let phibuck = this.p.p('a')[t] * rexp - this.p.p('c')[t] * r6inv;
    if (rsq > this.ciSq[t]) {
      const g = taperOf(r, this.ci[t], this.cc[t]);
      forcebuck = forcebuck * g.tt + phibuck * g.dt;
      phibuck *= g.tt;
    }
    return { eng: factorLJ * phibuck, fforce: factorLJ * forcebuck * r2inv };
  }

  extract(name: string): unknown {
    // pair_buck_mdf.cpp extract(): "a" and "c" (dim 2)
    if (name === 'a' || name === 'c') return this.p.p(name);
    return undefined;
  }
}
