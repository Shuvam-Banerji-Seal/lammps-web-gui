/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/KSPACE/pair_coul_msm.cpp, pair_lj_cut_coul_msm.cpp, pair_born_coul_msm.cpp, pair_buck_coul_msm.cpp and
 * pair_lj_charmm_coul_msm.cpp at commit b332f28 (Stan Moore, SNL), src/CG-SPICA/pair_lj_spica_coul_msm.cpp and
 * lj_spica_common.h (Axel Kohlmeyer, Temple U), the coefficients of their coul/long base styles' init_one, and the table
 * lookups of src/pair.cpp. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public
 * License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The pair styles of kspace_style msm: the short-range part of the MSM splitting,
 *   E = qqrd2e q_i q_j / r (1 - (r/a) gamma(r/a)),  F r = qqrd2e q_i q_j / r (1 + (r/a)^2 dgamma(r/a))
 * inside the Coulomb cutoff a (KSpace::gamma of the kspace style's order), directly for rsq <= tabinner^2 or with
 * pair_modify table 0, else from the bitmapped tables of Pair::init_tables (tables.ts; born/coul/msm and buck/coul/msm
 * always evaluate it directly). With kspace_modify pressure/scalar yes (the MSM default) the Coulomb pairs add no
 * virial tally: the virial of the style is the fdotr sum of its short-range forces alone plus E_coul/3 on the diagonal
 * (the Coulomb forces are kept apart until after that sum, as natively), and a per-atom virial is refused;
 * lj/spica/coul/msm refuses pressure/scalar yes altogether. The short-range terms use the native coefficients of each
 * style's init_one (lj1..lj4, born1..born3, buck1..buck2, with pow and exp of the C library).
 */
import { StyleError, type KSpace, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { PairCoulLong, PairLJCutCoulLong } from '../../engine/force/pair/coul_long';
import { PairBornCoulLong, PairBuckCoulLong } from '../../engine/force/pair/coul_long2';
import { PairLJCharmmCoulLong } from '../../engine/force/pair/charmm';
import { PairLJSpica } from '../../engine/force/pair/class2_spica51';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum } from '../../engine/force/util';
import { pow } from '../libm/pow';
import { exp } from '../libm/exp';
import { KSpaceMSM, type MsmPair } from './msm';
import { initMsmTables, msmGamma, msmDgamma, type CoulTables } from './tables';

const f32 = new Float32Array(1);
const i32 = new Int32Array(f32.buffer);

/** The kspace style a coul/msm style needs: kspace_style msm or msm/cg (native pair_check of both sides). */
const msmOf = (k: KSpace | null, pair: string): KSpaceMSM => {
  if (!k) throw new StyleError('Pair style requires a KSpace style');
  if (!(k instanceof KSpaceMSM)) throw new StyleError(`KSpace style ${k.name} is incompatible with Pair style ${pair}`);
  return k;
};

/** The native tables of one style, rebuilt when the table size, inner cutoff, Coulomb cutoff, qqrd2e or order change. */
class TableCache {
  private key = '';
  private tab: CoulTables | null = null;
  get(bits: number, tabinnerRsq: number, cutCoul: number, qqrd2e: number, order: number): CoulTables | null {
    if (bits === 0) return null;
    const key = `${bits} ${tabinnerRsq} ${cutCoul} ${qqrd2e} ${order}`;
    if (key !== this.key) {
      // pair_modify tabinner keeps sqrt(2) as tabinner^2 = 2 (commands/forcefield.ts); native squares sqrt(2.0)
      const tabinner = tabinnerRsq === 2 ? Math.SQRT2 : Math.sqrt(tabinnerRsq);
      this.tab = initMsmTables(bits, tabinner, cutCoul, qqrd2e, order);
      this.key = key;
    }
    return this.tab;
  }
}

/** The Coulomb term of one pair: forcecoul (F r) and ecoul, as the compute loops of the coul/msm styles evaluate them. */
const coulMsm = (
  out: { f: number; e: number }, rsq: number, qtmp: number, qj: number, qqrd2e: number, cutCoul: number, cutCoulsq: number,
  order: number, fc: number, tab: CoulTables | null,
): void => {
  let forcecoul: number, ecoul: number;
  if (!tab || rsq <= tab.tabinnersq) {
    const r = Math.sqrt(rsq);
    const prefactor = (qqrd2e * qtmp * qj) / r;
    const egamma = 1.0 - (r / cutCoul) * msmGamma(order, r / cutCoul);
    const fgamma = 1.0 + (rsq / cutCoulsq) * msmDgamma(order, r / cutCoul);
    forcecoul = prefactor * fgamma;
    if (fc < 1.0) forcecoul -= (1.0 - fc) * prefactor;
    ecoul = prefactor * egamma;
    if (fc < 1.0) ecoul -= (1.0 - fc) * prefactor;
  } else {
    f32[0] = rsq;
    const itable = (i32[0] & tab.ncoulmask) >> tab.ncoulshiftbits;
    const fraction = (f32[0] - tab.rtable[itable]) * tab.drtable[itable];
    let table = tab.ftable[itable] + fraction * tab.dftable[itable];
    forcecoul = qtmp * qj * table;
    let prefactor = 0;
    if (fc < 1.0) {
      table = tab.ctable[itable] + fraction * tab.dctable[itable];
      prefactor = qtmp * qj * table;
      forcecoul -= (1.0 - fc) * prefactor;
    }
    table = tab.etable[itable] + fraction * tab.detable[itable];
    ecoul = qtmp * qj * table;
    if (fc < 1.0) ecoul -= (1.0 - fc) * prefactor;
  }
  out.f = forcecoul;
  out.e = ecoul;
};

const PERATOM_REFUSAL = "Must use 'kspace_modify pressure/scalar no' to obtain per-atom virial with kspace_style MSM";

/** The fdotr virial over owned and ghost atoms (Pair::virial_fdotr_compute), added to v. */
const fdotr = (pc: PairCompute, v: Float64Array): void => {
  const x = pc.x, f = pc.f;
  let v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;
  for (let k = 0; k < 3 * pc.nall; k += 3) {
    v0 += f[k] * x[k]; v1 += f[k + 1] * x[k + 1]; v2 += f[k + 2] * x[k + 2];
    v3 += f[k + 1] * x[k]; v4 += f[k + 2] * x[k]; v5 += f[k + 2] * x[k + 1];
  }
  v[0] += v0; v[1] += v1; v[2] += v2; v[3] += v3; v[4] += v4; v[5] += v5;
};

/** The short-range term of one pair: f = its F r before the special weight, e = its energy before the weight. */
interface ShortRange { f: number; e: number }

/** The parts of a coul/msm style the shared loop needs. */
interface MsmLoopStyle {
  readonly name: string;
  ntypes: number;
  cutCoul: number;
  table: number;
  tabinnerRsq: number;
  cutsq: Float64Array;
  readonly tables: TableCache;
}

interface LoopOptions {
  /** born/coul/msm and buck/coul/msm evaluate the Coulomb term directly whatever pair_modify table says. */
  tables: boolean;
  /** Squared outer cutoff of the pair loop (cutsq of the type pair; coul/msm and lj/charmm/coul/msm a global one). */
  outerSq: ((t: number) => number) | null;
  /** Sets sr for type slot t (itype * (ntypes + 1) + jtype); false when the pair is outside the short-range cutoff. */
  term: ((t: number, rsq: number, r2inv: number, sr: ShortRange) => boolean) | null;
  /** lj/spica/coul/msm sums the forces on i in fxtmp and adds them after its neighbors (native eval_msm). */
  sumI: boolean;
}

const sr0: ShortRange = { f: 0, e: 0 };
const coulTerm = { f: 0, e: 0 };
let ftmp = new Float64Array(0);

/** The compute() of the coul/msm styles (the shared shape of their native loops). */
const msmLoop = (style: MsmLoopStyle, pc: PairCompute, msm: KSpaceMSM, o: LoopOptions): void => {
  const scalar = msm.scalarPressure;
  if (scalar && pc.vatom) throw new StyleError(PERATOM_REFUSAL);
  const list = pc.half!;
  const { x, f, type, q } = pc;
  let ft = f;
  if (scalar) {
    if (ftmp.length < 3 * pc.nall) ftmp = new Float64Array(3 * pc.nall);
    else ftmp.fill(0, 0, 3 * pc.nall);
    ft = ftmp;
  }
  const nt = style.ntypes + 1;
  const cutCoul = style.cutCoul, cutCoulsq = cutCoul * cutCoul;
  const order = msm.order, qqrd2e = pc.qqrd2e;
  const tab = o.tables ? style.tables.get(style.table, style.tabinnerRsq, cutCoul, qqrd2e, order) : null;
  const sLJ = pc.specialLJ, sC = pc.specialCoul;
  const tally = pc.eatom !== null || pc.vatom !== null;
  const t2 = coulTerm, sr = sr0;
  const { term, outerSq } = o;
  const cutsq = style.cutsq;
  let evdwlSum = 0, ecoulSum = 0;
  const nb = list.neighbors;
  for (let i = list.ilo ?? 0; i < list.inum; i++) {
    const qtmp = q[i];
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const ti = type[i] * nt;
    let fxtmp = 0.0, fytmp = 0.0, fztmp = 0.0;
    for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
      const jj = nb[k];
      const factorLJ = sLJ[jj >>> SBBITS];
      const factorCoul = sC[jj >>> SBBITS];
      const j = jj & NEIGHMASK;
      const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const tt = ti + type[j];
      if (rsq >= (outerSq ? outerSq(tt) : cutsq[tt])) continue;
      const r2inv = 1.0 / rsq;
      let forcecoul = 0.0, ecoul = 0.0;
      if (rsq < cutCoulsq) {
        coulMsm(t2, rsq, qtmp, q[j], qqrd2e, cutCoul, cutCoulsq, order, factorCoul, tab);
        forcecoul = t2.f;
        ecoul = t2.e;
      }
      let forcelj = 0.0, evdwl = 0.0;
      if (term && term(tt, rsq, r2inv, sr)) {
        forcelj = sr.f;
        evdwl = factorLJ * sr.e;
      }
      let fpair: number;
      if (!scalar) {
        fpair = (forcecoul + factorLJ * forcelj) * r2inv;
        if (o.sumI) { fxtmp += dx * fpair; fytmp += dy * fpair; fztmp += dz * fpair; }
        else { f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair; }
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
      } else {
        // separate the short-range and Coulombic forces
        fpair = factorLJ * forcelj * r2inv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        const fcoul = forcecoul * r2inv;
        ft[3 * i] += dx * fcoul; ft[3 * i + 1] += dy * fcoul; ft[3 * i + 2] += dz * fcoul;
        ft[3 * j] -= dx * fcoul; ft[3 * j + 1] -= dy * fcoul; ft[3 * j + 2] -= dz * fcoul;
      }
      evdwlSum += evdwl;
      ecoulSum += ecoul;
      if (tally) tallyAtom(pc, i, j, evdwl + ecoul, fpair, dx, dy, dz);
    }
    if (o.sumI) { f[3 * i] += fxtmp; f[3 * i + 1] += fytmp; f[3 * i + 2] += fztmp; }
  }
  pc.acc.evdwl += evdwlSum;
  pc.acc.ecoul += ecoulSum;
  if (scalar) {
    // the fdotr virial of the short-range forces (none for coul/msm), then E_coul/3 on the diagonal, then the Coulomb
    // forces join f
    if (term) fdotr(pc, pc.acc.virial);
    pc.fdotrDone = true;
    for (let c = 0; c < 3; c++) pc.acc.virial[c] += ecoulSum / 3.0;
    for (let k = 0; k < 3 * pc.nall; k++) f[k] += ft[k];
  }
};

/** single() of the coul/msm styles: the Coulomb term plus the style's short-range term. */
const msmSingle = (
  style: MsmLoopStyle, msm: KSpaceMSM, o: LoopOptions, itype: number, jtype: number, rsq: number, factorCoul: number,
  factorLJ: number, qi: number, qj: number,
): { eng: number; fforce: number } => {
  const nt = style.ntypes + 1;
  const tt = itype * nt + jtype;
  const cutCoulsq = style.cutCoul * style.cutCoul;
  const r2inv = 1.0 / rsq;
  let forcecoul = 0, phicoul = 0, forcelj = 0, philj = 0;
  if (rsq < cutCoulsq) {
    const qqrd2e = msm.qqrd2eRun;
    const tab = o.tables ? style.tables.get(style.table, style.tabinnerRsq, style.cutCoul, qqrd2e, msm.order) : null;
    coulMsm(coulTerm, rsq, qi, qj, qqrd2e, style.cutCoul, cutCoulsq, msm.order, factorCoul, tab);
    forcecoul = coulTerm.f;
    phicoul = coulTerm.e;
  }
  if (o.term && o.term(tt, rsq, r2inv, sr0)) {
    forcelj = sr0.f;
    philj = factorLJ * sr0.e;
  }
  return { eng: phicoul + philj, fforce: (forcecoul + factorLJ * forcelj) * r2inv };
};

/** pair_style coul/msm cutoff (pair_coul.html: "coul/msm args = cutoff"). */
export class PairCoulMSM extends PairCoulLong implements MsmPair, MsmLoopStyle {
  override readonly name: string = 'coul/msm';
  readonly msmflag = true as const;
  readonly tables = new TableCache();
  private kspace: KSpace | null = null;
  private readonly opts: LoopOptions;

  constructor() {
    super();
    this.opts = { tables: true, outerSq: () => this.cutCoul * this.cutCoul, term: null, sumI: false };
  }

  override settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError(`Illegal pair_style ${this.name} command`);
    this.cutCoul = parseNum(args[0], 'cutoff');
  }

  linkKSpace(k: KSpace | null): void { this.kspace = k; }

  override compute(pc: PairCompute): void {
    msmLoop(this, pc, msmOf(this.kspace, this.name), this.opts);
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi = 0, qj = 0): { eng: number; fforce: number } {
    return msmSingle(this, msmOf(this.kspace, this.name), this.opts, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
  }
}

/** pair_style lj/cut/coul/msm cutoff (cutoff2) (pair_lj_cut_coul.html: "lj/cut/coul/msm args = cutoff (cutoff2)"). */
export class PairLJCutCoulMSM extends PairLJCutCoulLong implements MsmPair, MsmLoopStyle {
  override readonly name: string = 'lj/cut/coul/msm';
  readonly msmflag = true as const;
  readonly tables = new TableCache();
  private kspace: KSpace | null = null;
  private cutljsq = new Float64Array(0);
  private readonly opts: LoopOptions;

  constructor() {
    super();
    this.opts = {
      tables: true, outerSq: null, sumI: false,
      term: (t, rsq, r2inv, sr) => {
        if (rsq >= this.cutljsq[t]) return false;
        const r6inv = r2inv * r2inv * r2inv;
        sr.f = r6inv * (this.lj1[t] * r6inv - this.lj2[t]);
        sr.e = r6inv * (this.lj3[t] * r6inv - this.lj4[t]) - this.offset[t];
        return true;
      },
    };
  }

  override settings(args: string[]): void {
    if (args.length < 1 || args.length > 2) throw new StyleError(`Illegal pair_style ${this.name} command`);
    this.cutGlobal = parseNum(args[0], 'LJ cutoff');
    this.cutCoul = args.length === 2 ? parseNum(args[1], 'Coulomb cutoff') : this.cutGlobal;
  }

  linkKSpace(k: KSpace | null): void { this.kspace = k; }

  override init(ctx: StyleContext): void {
    super.init(ctx);
    const nt = this.ntypes + 1;
    this.cutljsq = new Float64Array(nt * nt);
    for (let a = 1; a < nt; a++) for (let b = 1; b < nt; b++) this.cutljsq[a * nt + b] = this.p.get('cut', a, b) ** 2;
  }

  override compute(pc: PairCompute): void {
    msmLoop(this, pc, msmOf(this.kspace, this.name), this.opts);
  }

  override single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi = 0, qj = 0): { eng: number; fforce: number } {
    return msmSingle(this, msmOf(this.kspace, this.name), this.opts, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
  }
}

/** Native coefficient arrays of the Born and Buckingham styles (init_one of born/coul/long and buck/coul/long). */
class ExpCoeffs {
  rhoinv = new Float64Array(0); c1 = new Float64Array(0); c2 = new Float64Array(0); c3 = new Float64Array(0);
  a = new Float64Array(0); sigma = new Float64Array(0); c = new Float64Array(0); d = new Float64Array(0);
  offset = new Float64Array(0); cutljsq = new Float64Array(0);
  ensure(nt: number): void {
    if (this.a.length === nt * nt) return;
    for (const k of ['rhoinv', 'c1', 'c2', 'c3', 'a', 'sigma', 'c', 'd', 'offset', 'cutljsq'] as const) this[k] = new Float64Array(nt * nt);
  }
}

/** pair_style born/coul/msm cutoff (cutoff2) (pair_born.html: "born/coul/msm args = cutoff (cutoff2)"). */
export class PairBornCoulMSM extends PairBornCoulLong implements MsmPair, MsmLoopStyle {
  override readonly name: string = 'born/coul/msm';
  readonly msmflag = true as const;
  readonly tables = new TableCache();
  private kspace: KSpace | null = null;
  private readonly k = new ExpCoeffs();
  private readonly opts: LoopOptions;

  constructor() {
    super();
    const k = this.k;
    this.opts = {
      tables: false, outerSq: null, sumI: false,
      term: (t, rsq, r2inv, sr) => {
        if (rsq >= k.cutljsq[t]) return false;
        const r = Math.sqrt(rsq);
        const r6inv = r2inv * r2inv * r2inv;
        const rexp = exp((k.sigma[t] - r) * k.rhoinv[t]);
        sr.f = k.c1[t] * r * rexp - k.c2[t] * r6inv + k.c3[t] * r2inv * r6inv;
        sr.e = k.a[t] * rexp - k.c[t] * r6inv + k.d[t] * r6inv * r2inv - k.offset[t];
        return true;
      },
    };
  }

  override settings(args: string[], ctx: StyleContext): void {
    if (args.length < 1 || args.length > 2) throw new StyleError(`Illegal pair_style ${this.name} command`);
    super.settings(args, ctx);
  }

  linkKSpace(kspace: KSpace | null): void { this.kspace = kspace; }

  override initOne(i: number, j: number): number {
    const cut = super.initOne(i, j);
    const nt = this.ntypes + 1, k = this.k;
    k.ensure(nt);
    const p = this.p;
    const a = p.get('A', i, j), rho = p.get('rho', i, j), sigma = p.get('sigma', i, j), c = p.get('C', i, j), d = p.get('D', i, j);
    const raw = p.get('cut', i, j);
    const cutLJ = Number.isNaN(raw) ? this.cutGlobal : raw;
    const rhoinv = 1.0 / rho;
    let offset = 0.0;
    if (this.shift && cutLJ > 0.0) {
      const rexp = exp((sigma - cutLJ) * rhoinv);
      offset = a * rexp - c / pow(cutLJ, 6.0) + d / pow(cutLJ, 8.0);
    }
    for (const t of [i * nt + j, j * nt + i]) {
      k.rhoinv[t] = rhoinv; k.c1[t] = a / rho; k.c2[t] = 6.0 * c; k.c3[t] = 8.0 * d;
      k.a[t] = a; k.sigma[t] = sigma; k.c[t] = c; k.d[t] = d; k.offset[t] = offset; k.cutljsq[t] = cutLJ * cutLJ;
    }
    return cut;
  }

  override compute(pc: PairCompute): void {
    msmLoop(this, pc, msmOf(this.kspace, this.name), this.opts);
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi = 0, qj = 0): { eng: number; fforce: number } {
    return msmSingle(this, msmOf(this.kspace, this.name), this.opts, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
  }
}

/** pair_style buck/coul/msm cutoff (cutoff2) (pair_buck.html: "buck/coul/msm args = cutoff (cutoff2)"). */
export class PairBuckCoulMSM extends PairBuckCoulLong implements MsmPair, MsmLoopStyle {
  override readonly name: string = 'buck/coul/msm';
  readonly msmflag = true as const;
  readonly tables = new TableCache();
  private kspace: KSpace | null = null;
  private readonly k = new ExpCoeffs();
  private readonly opts: LoopOptions;

  constructor() {
    super();
    const k = this.k;
    this.opts = {
      tables: false, outerSq: null, sumI: false,
      term: (t, rsq, r2inv, sr) => {
        if (rsq >= k.cutljsq[t]) return false;
        const r = Math.sqrt(rsq);
        const r6inv = r2inv * r2inv * r2inv;
        const rexp = exp(-r * k.rhoinv[t]);
        sr.f = k.c1[t] * r * rexp - k.c2[t] * r6inv;
        sr.e = k.a[t] * rexp - k.c[t] * r6inv - k.offset[t];
        return true;
      },
    };
  }

  override settings(args: string[], ctx: StyleContext): void {
    if (args.length < 1 || args.length > 2) throw new StyleError(`Illegal pair_style ${this.name} command`);
    super.settings(args, ctx);
  }

  linkKSpace(kspace: KSpace | null): void { this.kspace = kspace; }

  override initOne(i: number, j: number): number {
    const cut = super.initOne(i, j);
    const nt = this.ntypes + 1, k = this.k;
    k.ensure(nt);
    const p = this.p;
    const a = p.get('A', i, j), rho = p.get('rho', i, j), c = p.get('C', i, j);
    const raw = p.get('cut', i, j);
    const cutLJ = Number.isNaN(raw) ? this.cutGlobal : raw;
    let offset = 0.0;
    if (this.shift && cutLJ > 0.0) {
      const rexp = exp(-cutLJ / rho);
      offset = a * rexp - c / pow(cutLJ, 6.0);
    }
    for (const t of [i * nt + j, j * nt + i]) {
      k.rhoinv[t] = 1.0 / rho; k.c1[t] = a / rho; k.c2[t] = 6.0 * c;
      k.a[t] = a; k.c[t] = c; k.offset[t] = offset; k.cutljsq[t] = cutLJ * cutLJ;
    }
    return cut;
  }

  override compute(pc: PairCompute): void {
    msmLoop(this, pc, msmOf(this.kspace, this.name), this.opts);
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi = 0, qj = 0): { eng: number; fforce: number } {
    return msmSingle(this, msmOf(this.kspace, this.name), this.opts, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
  }
}

/** pair_style lj/charmm/coul/msm inner outer (cutoff) (pair_charmm.html: "lj/charmm/coul/msm args = inner outer (cutoff)"). */
export class PairLJCharmmCoulMSM extends PairLJCharmmCoulLong implements MsmPair, MsmLoopStyle {
  override readonly name: string = 'lj/charmm/coul/msm';
  readonly msmflag = true as const;
  override cutCoul = 0;
  readonly tables = new TableCache();
  private kspace: KSpace | null = null;
  private n1 = new Float64Array(0); private n2 = new Float64Array(0);
  private n3 = new Float64Array(0); private n4 = new Float64Array(0);
  private cutLjInnersq = 0; private cutLjsq = 0; private denomLjInv = 0;
  private readonly opts: LoopOptions;

  constructor() {
    super();
    this.opts = {
      tables: true, sumI: false,
      outerSq: () => Math.max(this.cutLjsq, this.cutCoul * this.cutCoul),
      term: (t, rsq, r2inv, sr) => {
        const cutLjsq = this.cutLjsq, inner = this.cutLjInnersq;
        if (rsq >= cutLjsq) return false;
        const r6inv = r2inv * r2inv * r2inv;
        let forcelj = r6inv * (this.n1[t] * r6inv - this.n2[t]);
        let evdwl = r6inv * (this.n3[t] * r6inv - this.n4[t]);
        if (rsq > inner) {
          const switch1 = (cutLjsq - rsq) * (cutLjsq - rsq) * (cutLjsq + 2.0 * rsq - 3.0 * inner) * this.denomLjInv;
          const switch2 = 12.0 * rsq * (cutLjsq - rsq) * (rsq - inner) * this.denomLjInv;
          const philj = r6inv * (this.n3[t] * r6inv - this.n4[t]);
          forcelj = forcelj * switch1 + philj * switch2;
          evdwl *= switch1;
        }
        sr.f = forcelj;
        sr.e = evdwl;
        return true;
      },
    };
  }

  override settings(args: string[]): void {
    if (args.length !== 2 && args.length !== 3) throw new StyleError(`Illegal pair_style ${this.name} command`);
    super.settings(args);
  }

  linkKSpace(k: KSpace | null): void { this.kspace = k; }

  override initOne(i: number, j: number): number {
    const cut = super.initOne(i, j);
    const nt = this.ntypes + 1;
    if (this.n1.length !== nt * nt) {
      this.n1 = new Float64Array(nt * nt); this.n2 = new Float64Array(nt * nt);
      this.n3 = new Float64Array(nt * nt); this.n4 = new Float64Array(nt * nt);
    }
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
    for (const t of [i * nt + j, j * nt + i]) {
      this.n1[t] = 48.0 * eps * pow(sig, 12.0);
      this.n2[t] = 24.0 * eps * pow(sig, 6.0);
      this.n3[t] = 4.0 * eps * pow(sig, 12.0);
      this.n4[t] = 4.0 * eps * pow(sig, 6.0);
    }
    return cut;
  }

  override init(ctx: StyleContext): void {
    super.init(ctx);
    // PairLJCharmmCoulLong::init_style
    this.cutLjInnersq = this.inner * this.inner;
    this.cutLjsq = this.outer * this.outer;
    const dl = this.cutLjsq - this.cutLjInnersq;
    this.denomLjInv = 1.0 / (dl * dl * dl);
  }

  override compute(pc: PairCompute): void {
    msmLoop(this, pc, msmOf(this.kspace, this.name), this.opts);
  }

  override single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi = 0, qj = 0): { eng: number; fforce: number } {
    return msmSingle(this, msmOf(this.kspace, this.name), this.opts, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
  }
}

/** lj_spica_common.h: the LJ types and their prefactors and powers. */
const SPICA_PREFACT: Record<string, { pref: number; p1: number; p2: number }> = {
  '9 6': { pref: 6.75, p1: 9.0, p2: 6.0 },
  '12 4': { pref: 2.59807621135332, p1: 12.0, p2: 4.0 },
  '12 6': { pref: 4.0, p1: 12.0, p2: 6.0 },
  '12 5': { pref: 3.20377984125109, p1: 12.0, p2: 5.0 },
};
const LJ9_6 = 1, LJ12_4 = 2, LJ12_6 = 3, LJ12_5 = 4;
const SPICA_TYPE: Record<string, number> = { '9 6': LJ9_6, '12 4': LJ12_4, '12 6': LJ12_6, '12 5': LJ12_5 };

/**
 * pair_style lj/spica/coul/msm and lj/sdk/coul/msm cutoff (cutoff2) (pair_spica.html: "lj/spica/coul/msm args = cutoff
 * (cutoff2)"; "The original implementation of the above styles are style lj/sdk, lj/sdk/coul/long, and lj/sdk/coul/msm").
 * native: "Must use 'kspace_modify pressure/scalar no' with Pair style".
 */
export class PairLJSpicaCoulMSM extends PairLJSpica implements MsmPair, MsmLoopStyle {
  override readonly name: string;
  readonly msmflag = true as const;
  readonly tables = new TableCache();
  private kspace: KSpace | null = null;
  private ljt = new Int32Array(0);
  private n1 = new Float64Array(0); private n2 = new Float64Array(0);
  private n3 = new Float64Array(0); private n4 = new Float64Array(0);
  private noff = new Float64Array(0); private cutljsq = new Float64Array(0);
  private readonly opts: LoopOptions;

  constructor(name: 'lj/spica/coul/msm' | 'lj/sdk/coul/msm') {
    super(name === 'lj/spica/coul/msm' ? 'lj/spica/coul/long' : 'lj/sdk/coul/long');
    this.name = name;
    this.opts = {
      tables: true, outerSq: null, sumI: true,
      term: (t, rsq, r2inv, sr) => {
        if (rsq >= this.cutljsq[t]) return false;
        const lt = this.ljt[t];
        const lj1 = this.n1[t], lj2 = this.n2[t], lj3 = this.n3[t], lj4 = this.n4[t], off = this.noff[t];
        if (lt === LJ12_4) {
          const r4inv = r2inv * r2inv;
          sr.f = r4inv * (lj1 * r4inv * r4inv - lj2);
          sr.e = r4inv * (lj3 * r4inv * r4inv - lj4) - off;
        } else if (lt === LJ9_6) {
          const r3inv = r2inv * Math.sqrt(r2inv);
          const r6inv = r3inv * r3inv;
          sr.f = r6inv * (lj1 * r3inv - lj2);
          sr.e = r6inv * (lj3 * r3inv - lj4) - off;
        } else if (lt === LJ12_6) {
          const r6inv = r2inv * r2inv * r2inv;
          sr.f = r6inv * (lj1 * r6inv - lj2);
          sr.e = r6inv * (lj3 * r6inv - lj4) - off;
        } else {
          const r5inv = r2inv * r2inv * Math.sqrt(r2inv);
          const r7inv = r5inv * r2inv;
          sr.f = r5inv * (lj1 * r7inv - lj2);
          sr.e = r5inv * (lj3 * r7inv - lj4) - off;
        }
        return true;
      },
    };
  }

  linkKSpace(k: KSpace | null): void { this.kspace = k; }

  override initOne(i: number, j: number): number {
    const cut = super.initOne(i, j);
    const nt = this.ntypes + 1;
    if (this.n1.length !== nt * nt) {
      this.ljt = new Int32Array(nt * nt);
      for (const k of ['n1', 'n2', 'n3', 'n4', 'noff', 'cutljsq'] as const) this[k] = new Float64Array(nt * nt);
    }
    const p = this.p;
    const key = `${p.get('cgn', i, j)} ${p.get('cgm', i, j)}`;
    const c = SPICA_PREFACT[key];
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), cutLJ = p.get('cut', i, j);
    let offset = 0.0;
    if (this.shift && cutLJ > 0.0) {
      const ratio = sig / cutLJ;
      offset = c.pref * eps * (pow(ratio, c.p1) - pow(ratio, c.p2));
    }
    for (const t of [i * nt + j, j * nt + i]) {
      this.ljt[t] = SPICA_TYPE[key];
      this.n1[t] = c.pref * c.p1 * eps * pow(sig, c.p1);
      this.n2[t] = c.pref * c.p2 * eps * pow(sig, c.p2);
      this.n3[t] = c.pref * eps * pow(sig, c.p1);
      this.n4[t] = c.pref * eps * pow(sig, c.p2);
      this.noff[t] = offset;
      this.cutljsq[t] = cutLJ * cutLJ;
    }
    return cut;
  }

  override compute(pc: PairCompute): void {
    const msm = msmOf(this.kspace, this.name);
    if (msm.scalarPressure) throw new StyleError("Must use 'kspace_modify pressure/scalar no' with Pair style");
    msmLoop(this, pc, msm, this.opts);
  }

  override single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi = 0, qj = 0): { eng: number; fforce: number } {
    return msmSingle(this, msmOf(this.kspace, this.name), this.opts, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
  }
}
