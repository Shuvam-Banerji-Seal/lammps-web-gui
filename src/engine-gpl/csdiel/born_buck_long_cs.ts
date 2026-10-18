/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/CORESHELL/pair_born_coul_long_cs.cpp and
 * .h, src/CORESHELL/pair_buck_coul_long_cs.cpp and .h (the core/shell variants, Hendrik Heenen's /cs series), and the base
 * styles src/KSPACE/pair_born_coul_long.cpp and pair_buck_coul_long.cpp (settings, coeff, init_one). The erfc table of the
 * tabulated branch is pair.cpp init_tables (table N > 0), as emulated by engine/force/erfc.ts (ErfcTableCache, shared with
 * the other /cs styles in this folder). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * pair_style born/coul/long/cs cutoff (cutoff2) and buck/coul/long/cs cutoff (cutoff2) (docs.lammps.org/pair_cs.html:
 * "All the styles are identical to the corresponding pair style without the "/cs" in the name ... except that they correctly
 * treat the special case where the distance between two charged core and shell atoms in the same core/shell pair approach
 * r = 0.0"). The native compute() adds EPSILON to rsq after the cutsq test; the Coulomb term is the real-space Ewald term
 * of pair_coul.html with the polynomial erfc (EWALD_P, B0..B5, EWALD_F, native constants), or erf() for an excluded pair
 * (factor_coul < 1), or the table of pair_modify table N for rsq above tabinner.
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { PairBorn, PairBuck } from '../../engine/force/pair/simple';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { ErfcTableCache, type ErfcTable } from '../../engine/force/erfc';
import { parseNum } from '../../engine/force/util';
import { erf as glibcErf } from '../libm/erf';
import { pow as libmPow } from '../libm/pow';
import { exp as glibcExp } from '../libm/exp';

/** EWALD_F, EWALD_P, B0..B5 and EPSILON of the native /cs compute() (born/buck long cs). */
const EWALD_F = 1.12837917;
const EWALD_P = 9.95473818e-1;
const B0 = -0.1335096380159268;
const B1 = -2.57839507e-1;
const B2 = -1.37203639e-1;
const B3 = -8.88822059e-3;
const B4 = -5.80844129e-3;
const B5 = 1.14652755e-1;
const EPSILON = 1.0e-20;
/** MathConst::MY_ISPI4 = 2/sqrt(pi) (the force term of an excluded pair). */
const MY_ISPI4 = 1.12837916709551257389;

/** Refusal shared by the /cs styles (native single_enable = 0). */
const noSingle = (name: string): never => {
  throw new StyleError(`pair style ${name}: single() (pair/local, pair_write) is not supported (native single_enable = 0)`);
};

/**
 * The Coulomb term of one pair of a born/buck long cs style (native compute(), both branches). Returns the force term
 * (force times r, before the 1/r^2 of fpair) and the energy. fc = special_coul weight, tab = the table or null (table 0 or
 * rsq <= tabinner, which the native code tests with rsq <= tabinnersq).
 */
const coulTerm = (
  rsq: number, qi: number, qj: number, fc: number, g: number, qqrd2e: number, tab: ErfcTable | null,
): { force: number; e: number } => {
  if (tab && rsq > tab.inner) {
    // table N > 0: the tabulated kernels carry qqrd2e / r; the special correction removes (1 - fc) times the bare term
    let force = qi * qj * (qqrd2e * tab.force(rsq));
    let e = qi * qj * (qqrd2e * tab.energy(rsq));
    if (fc < 1.0) {
      const prefactor = qi * qj * (qqrd2e * tab.coul(rsq));
      force -= (1.0 - fc) * prefactor;
      e -= (1.0 - fc) * prefactor;
    }
    return { force, e };
  }
  const r = Math.sqrt(rsq);
  let prefactor = qqrd2e * qi * qj;
  const grij = g * r;
  const expm2 = glibcExp(-grij * grij);
  if (fc < 1.0) {
    // for excluded pairs the correction nearly cancels the Ewald term: use erf(); erfc holds erfc - (1 - factor_coul)
    const erfc = fc - glibcErf(grij);
    prefactor /= r;
    return { force: prefactor * (erfc + MY_ISPI4 * grij * expm2), e: prefactor * erfc };
  }
  const t = 1.0 / (1.0 + EWALD_P * grij);
  const u = 1.0 - t;
  const erfc = t * (1.0 + u * (B0 + u * (B1 + u * (B2 + u * (B3 + u * (B4 + u * B5)))))) * expm2;
  prefactor /= r;
  return { force: prefactor * (erfc + EWALD_F * grij * expm2), e: prefactor * erfc };
};

/** Shared per-type storage of the Born and Buckingham arrays (flat, index i * (ntypes + 1) + j). */
class TypeArrays {
  rhoinv = new Float64Array(0);
  cutLJsq = new Float64Array(0);
  ensure(n: number): void {
    if (this.rhoinv.length === n) return;
    this.rhoinv = new Float64Array(n);
    this.cutLJsq = new Float64Array(n);
  }
}

/* ----------------------------------------------------------------------
   born/coul/long/cs
   ---------------------------------------------------------------------- */

/**
 * pair_style born/coul/long/cs cutoff (cutoff2). The Born term of pair_born.html with the native born1 = A/rho, born2 = 6C,
 * born3 = 8D (init_one), and the long-range Coulomb term above.
 */
export class PairBornCoulLongCS extends PairBorn {
  readonly name: string = 'born/coul/long/cs';
  coulLong = true;
  cutCoul = 0;
  private readonly ta = new TypeArrays();
  private born1 = new Float64Array(0);
  private born2 = new Float64Array(0);
  private born3 = new Float64Array(0);
  private bSig = new Float64Array(0);
  private bA = new Float64Array(0);
  private bC = new Float64Array(0);
  private bD = new Float64Array(0);
  private bOff = new Float64Array(0);
  private readonly erfcTables = new ErfcTableCache();

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length < 1 || args.length > 2) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.cutCoul = args.length === 1 ? this.cutGlobal : parseNum(args[1], 'cutoff2');
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 7 || args.length > 8) throw new StyleError('Incorrect args for pair coefficients');
    const a = parseNum(args[2], 'A');
    const rho = parseNum(args[3], 'rho');
    const sig = parseNum(args[4], 'sigma');
    if (rho <= 0) throw new StyleError('Incorrect args for pair coefficients');
    const c = parseNum(args[5], 'C');
    const d = parseNum(args[6], 'D');
    const cut = args.length === 8 ? parseNum(args[7], 'cutoff') : Number.NaN;
    this.p.setRange(args[0], args[1], [a, rho, sig, c, d, cut]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError('All pair coeffs are not set. Status');
    const nt = this.ntypes + 1;
    if (this.ta.rhoinv.length !== nt * nt) {
      this.born1 = new Float64Array(nt * nt); this.born2 = new Float64Array(nt * nt); this.born3 = new Float64Array(nt * nt);
      this.bSig = new Float64Array(nt * nt); this.bA = new Float64Array(nt * nt); this.bC = new Float64Array(nt * nt);
      this.bD = new Float64Array(nt * nt); this.bOff = new Float64Array(nt * nt);
    }
    this.ta.ensure(nt * nt);
    const a = p.get('A', i, j), rho = p.get('rho', i, j), sig = p.get('sigma', i, j);
    const c = p.get('C', i, j), d = p.get('D', i, j);
    const cutRaw = p.get('cut', i, j);
    const cutLJ = Number.isNaN(cutRaw) ? this.cutGlobal : cutRaw;
    const rhoinv = 1.0 / rho;
    let off = 0;
    if (this.shift && cutLJ > 0.0) {
      const rexp = glibcExp((sig - cutLJ) * rhoinv);
      off = a * rexp - c / libmPow(cutLJ, 6.0) + d / libmPow(cutLJ, 8.0);
    }
    const k1 = i * nt + j, k2 = j * nt + i;
    for (const k of [k1, k2]) {
      this.ta.rhoinv[k] = rhoinv;
      this.ta.cutLJsq[k] = cutLJ * cutLJ;
      this.born1[k] = a / rho;
      this.born2[k] = 6.0 * c;
      this.born3[k] = 8.0 * d;
      this.bSig[k] = sig; this.bA[k] = a; this.bC[k] = c; this.bD[k] = d; this.bOff[k] = off;
    }
    return Math.max(cutLJ, this.cutCoul);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const cutCoulSq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.erfcTables.get(this.table, g, cutCoulSq, this.tabinnerRsq);
    const { rhoinv, cutLJsq } = this.ta;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qtmp = q[i];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const factorLJ = sLJ[sb], factorCoul = sC[sb];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq0 = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (!(rsq0 < this.cutsq[t])) continue;
        // EPSILON for r = 0 (a core and its shell): the interaction is removed by the special bond
        const rsq = rsq0 + EPSILON;
        const r2inv = 1.0 / rsq;
        let forcecoul = 0, ec = 0;
        if (rsq < cutCoulSq) {
          const c = coulTerm(rsq, qtmp, q[j], factorCoul, g, qqrd2e, tab);
          forcecoul = c.force;
          ec = c.e;
        }
        let forceborn = 0, ev = 0;
        if (rsq < cutLJsq[t]) {
          const r = Math.sqrt(rsq);
          const r6inv = r2inv * r2inv * r2inv;
          const rexp = glibcExp((this.bSig[t] - r) * rhoinv[t]);
          forceborn = this.born1[t] * r * rexp - this.born2[t] * r6inv + this.born3[t] * r2inv * r6inv;
          ev = this.bA[t] * rexp - this.bC[t] * r6inv + this.bD[t] * r6inv * r2inv - this.bOff[t];
          ev *= factorLJ;
        }
        const fpair = (forcecoul + factorLJ * forceborn) * r2inv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        evdwl += ev;
        ecoul += ec;
        if (tally) tallyAtom(pc, i, j, ev + ec, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }

  single(): never { return noSingle(this.name); }
}

/* ----------------------------------------------------------------------
   buck/coul/long/cs
   ---------------------------------------------------------------------- */

/**
 * pair_style buck/coul/long/cs cutoff (cutoff2). The Buckingham term of pair_buck.html with the native buck1 = A/rho,
 * buck2 = 6C (init_one), and the same long-range Coulomb term as born/coul/long/cs.
 */
export class PairBuckCoulLongCS extends PairBuck {
  readonly name: string = 'buck/coul/long/cs';
  coulLong = true;
  cutCoul = 0;
  private readonly ta = new TypeArrays();
  private buck1 = new Float64Array(0);
  private buck2 = new Float64Array(0);
  private bA = new Float64Array(0);
  private bC = new Float64Array(0);
  private bOff = new Float64Array(0);
  private readonly erfcTables = new ErfcTableCache();

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length < 1 || args.length > 2) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.cutCoul = args.length === 1 ? this.cutGlobal : parseNum(args[1], 'cutoff2');
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 5 || args.length > 6) throw new StyleError('Incorrect args for pair coefficients');
    const a = parseNum(args[2], 'A');
    const rho = parseNum(args[3], 'rho');
    if (rho <= 0) throw new StyleError('Incorrect args for pair coefficients');
    const c = parseNum(args[4], 'C');
    const cut = args.length === 6 ? parseNum(args[5], 'cutoff') : Number.NaN;
    this.p.setRange(args[0], args[1], [a, rho, c, cut]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError('All pair coeffs are not set. Status');
    const nt = this.ntypes + 1;
    if (this.ta.rhoinv.length !== nt * nt) {
      this.buck1 = new Float64Array(nt * nt); this.buck2 = new Float64Array(nt * nt);
      this.bA = new Float64Array(nt * nt); this.bC = new Float64Array(nt * nt);
      this.bOff = new Float64Array(nt * nt);
    }
    this.ta.ensure(nt * nt);
    const a = p.get('A', i, j), rho = p.get('rho', i, j), c = p.get('C', i, j);
    const cutRaw = p.get('cut', i, j);
    const cutLJ = Number.isNaN(cutRaw) ? this.cutGlobal : cutRaw;
    const rhoinv = 1.0 / rho;
    let off = 0;
    if (this.shift && cutLJ > 0.0) {
      const rexp = glibcExp(-cutLJ / rho);
      off = a * rexp - c / libmPow(cutLJ, 6.0);
    }
    const k1 = i * nt + j, k2 = j * nt + i;
    for (const k of [k1, k2]) {
      this.ta.rhoinv[k] = rhoinv;
      this.ta.cutLJsq[k] = cutLJ * cutLJ;
      this.buck1[k] = a / rho;
      this.buck2[k] = 6.0 * c;
      this.bA[k] = a; this.bC[k] = c; this.bOff[k] = off;
    }
    return Math.max(cutLJ, this.cutCoul);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const cutCoulSq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.erfcTables.get(this.table, g, cutCoulSq, this.tabinnerRsq);
    const { rhoinv, cutLJsq } = this.ta;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qtmp = q[i];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const factorLJ = sLJ[sb], factorCoul = sC[sb];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq0 = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (!(rsq0 < this.cutsq[t])) continue;
        // EPSILON for r = 0 (a core and its shell): the interaction is removed by the special bond
        const rsq = rsq0 + EPSILON;
        const r2inv = 1.0 / rsq;
        let forcecoul = 0, ec = 0;
        if (rsq < cutCoulSq) {
          const c = coulTerm(rsq, qtmp, q[j], factorCoul, g, qqrd2e, tab);
          forcecoul = c.force;
          ec = c.e;
        }
        let forcebuck = 0, ev = 0;
        if (rsq < cutLJsq[t]) {
          const r = Math.sqrt(rsq);
          const r6inv = r2inv * r2inv * r2inv;
          const rexp = glibcExp(-r * rhoinv[t]);
          forcebuck = this.buck1[t] * r * rexp - this.buck2[t] * r6inv;
          ev = this.bA[t] * rexp - this.bC[t] * r6inv - this.bOff[t];
          ev *= factorLJ;
        }
        const fpair = (forcecoul + factorLJ * forcebuck) * r2inv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        evdwl += ev;
        ecoul += ec;
        if (tally) tallyAtom(pc, i, j, ev + ec, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }

  single(): never { return noSingle(this.name); }
}
