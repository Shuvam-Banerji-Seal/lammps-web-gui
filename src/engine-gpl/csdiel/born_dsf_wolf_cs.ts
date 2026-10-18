/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/CORESHELL/pair_born_coul_dsf_cs.cpp and
 * .h (contributing author Ariel Lozano, arielzn@gmail.com, after Fennell and Gezelter, JCP 124, 234104 (2006)),
 * src/CORESHELL/pair_born_coul_wolf_cs.cpp and .h, and the base styles src/EXTRA-PAIR/pair_born_coul_dsf.cpp and
 * pair_born_coul_wolf.cpp (settings, coeff, init_style, init_one). Copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style born/coul/dsf/cs and born/coul/wolf/cs alpha cutoff (cutoff2) (docs.lammps.org/pair_cs.html: "All the styles
 * are identical to the corresponding pair style without the "/cs" in the name ... except that they correctly treat the
 * special case where the distance between two charged core and shell atoms in the same core/shell pair approach
 * r = 0.0"). Each adds EPSILON to rsq (after the cutsq test, as native) so a core/shell pair at r = 0 has a finite Coulomb
 * term that the special-bond correction removes. The Born-Mayer-Huggins part is that of pair_born.html; the Coulomb part
 * is the damped shifted force (dsf) or the Wolf sum (wolf) of pair_coul.html.
 *
 * Native arithmetic: the dsf compute takes erfc(alpha r) as my_erfcx(arg) * expmsq(arg) (math_special.h, the Johnson erfcx
 * table and the exp2_x86 bit construction, ported in erfcx.ts); the wolf compute uses the C library erfc() and exp()
 * (glibc, the libm/ port). The shifts are the init_style (dsf) or compute (wolf) expressions of the base styles.
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { PairBorn } from '../../engine/force/pair/simple';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum } from '../../engine/force/util';
import { erfc as glibcErfc } from '../libm/erf';
import { exp as glibcExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
import { expmsq, myErfcx } from './erfcx';

/** sqrt(pi), MathConst::MY_PIS. */
const MY_PIS = 1.77245385090551602729;
/** EPSILON of the /cs styles (added to rsq for r = 0). */
const EPSILON = 1.0e-20;

/** Refusal shared by the /cs styles (native single_enable = 0). */
const noSingle = (name: string): never => {
  throw new StyleError(`pair style ${name}: single() (pair/local, pair_write) is not supported (native single_enable = 0)`);
};

/**
 * The shared body of born/coul/dsf/cs and born/coul/wolf/cs: the native settings, coeff and init_one of the base styles,
 * with the Born parameters stored as native (rhoinv = 1/rho, born1 = A/rho, born2 = 6C, born3 = 8D).
 */
abstract class BornCoulCSBase extends PairBorn {
  keepExcluded = true;
  /** alpha (native alpha / alf), the Coulomb cutoff cut_coul and cut_coul^2. */
  protected alpha = 0;
  protected cutCoul = 0;
  protected cutCoulSq = 0;
  protected cutGlobalLJ = 0;
  private bRhoinv = new Float64Array(0);
  private bBorn1 = new Float64Array(0);
  private bBorn2 = new Float64Array(0);
  private bBorn3 = new Float64Array(0);
  private bSig = new Float64Array(0);
  private bA = new Float64Array(0);
  private bC = new Float64Array(0);
  private bD = new Float64Array(0);
  private bOff = new Float64Array(0);
  private cutLJsq = new Float64Array(0);

  /** settings of both base styles: narg 2 or 3 (alpha cutoff [cutoff2]), as native. */
  protected baseSettings(args: string[]): void {
    if (args.length < 2 || args.length > 3) throw new StyleError('Illegal pair_style command');
    this.alpha = parseNum(args[0], 'alpha');
    this.cutGlobalLJ = parseNum(args[1], 'cutoff');
    this.cutGlobal = this.cutGlobalLJ;
    this.cutCoul = args.length === 2 ? this.cutGlobalLJ : parseNum(args[2], 'cutoff2');
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
    if (this.bRhoinv.length !== nt * nt) {
      this.bRhoinv = new Float64Array(nt * nt); this.bBorn1 = new Float64Array(nt * nt);
      this.bBorn2 = new Float64Array(nt * nt); this.bBorn3 = new Float64Array(nt * nt);
      this.bSig = new Float64Array(nt * nt); this.bA = new Float64Array(nt * nt);
      this.bC = new Float64Array(nt * nt); this.bD = new Float64Array(nt * nt);
      this.bOff = new Float64Array(nt * nt); this.cutLJsq = new Float64Array(nt * nt);
    }
    const a = p.get('A', i, j), rho = p.get('rho', i, j), sig = p.get('sigma', i, j);
    const c = p.get('C', i, j), d = p.get('D', i, j);
    const cutRaw = p.get('cut', i, j);
    const cutLJ = Number.isNaN(cutRaw) ? this.cutGlobalLJ : cutRaw;
    const rhoinv = 1.0 / rho;
    const born1 = a / rho;
    const born2 = 6.0 * c;
    const born3 = 8.0 * d;
    let off = 0;
    if (this.shift && cutLJ > 0.0) {
      const rexp = glibcExp((sig - cutLJ) * rhoinv);
      off = a * rexp - c / libmPow(cutLJ, 6.0) + d / libmPow(cutLJ, 8.0);
    }
    const k1 = i * nt + j, k2 = j * nt + i;
    for (const k of [k1, k2]) {
      this.bRhoinv[k] = rhoinv; this.bBorn1[k] = born1; this.bBorn2[k] = born2; this.bBorn3[k] = born3;
      this.bSig[k] = sig; this.bA[k] = a; this.bC[k] = c; this.bD[k] = d; this.bOff[k] = off;
      this.cutLJsq[k] = cutLJ * cutLJ;
    }
    return Math.max(cutLJ, this.cutCoul);
  }

  /** Born-Mayer-Huggins part of one pair (native compute()): returns the force over r^2 part and the energy, both scaled by factor_lj. */
  protected bornTerm(t: number, rsq: number, r2inv: number, factorLJ: number): { forceborn: number; evdwl: number; rexp: number; r6inv: number; r: number } {
    let forceborn = 0, evdwl = 0, rexp = 0, r6inv = 0, r = 0;
    if (rsq < this.cutLJsq[t]) {
      r6inv = r2inv * r2inv * r2inv;
      r = Math.sqrt(rsq);
      rexp = glibcExp((this.bSig[t] - r) * this.bRhoinv[t]);
      forceborn = this.bBorn1[t] * r * rexp - this.bBorn2[t] * r6inv + this.bBorn3[t] * r2inv * r6inv;
      evdwl = this.bA[t] * rexp - this.bC[t] * r6inv + this.bD[t] * r6inv * r2inv - this.bOff[t];
      evdwl *= factorLJ;
    }
    return { forceborn, evdwl, rexp, r6inv, r };
  }

  /** The Born part has no tail correction in these base styles (native init_one has no tail code). */
  tailSums(_count: Float64Array): { etail: number; ptail: number } { return { etail: 0, ptail: 0 }; }

  single(): never { return noSingle(this.name); }
  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }

}

/* ----------------------------------------------------------------------
   born/coul/dsf/cs
   ---------------------------------------------------------------------- */

/**
 * pair_style born/coul/dsf/cs alpha cutoff (cutoff2). Native init_style: f_shift = -(erfc(a rc)/rc^2 + 2 a exp(-a^2 rc^2)/(MY_PIS rc)),
 * e_shift = erfc(a rc)/rc - f_shift rc. Native compute(): self energy -(e_shift/2 + a/MY_PIS) q^2 qqrd2e per owned atom; per pair
 * the dsf Coulomb term with erfc(arg) = my_erfcx(arg) expmsq(arg), arg = alpha r, and the special correction.
 */
export class PairBornCoulDsfCS extends BornCoulCSBase {
  readonly name: string = 'born/coul/dsf/cs';
  private eShift = 0;
  private fShift = 0;

  settings(args: string[], _ctx: StyleContext): void {
    this.baseSettings(args);
  }

  initStyle(_ctx: StyleContext): void {
    this.cutCoulSq = this.cutCoul * this.cutCoul;
    const erfcc = glibcErfc(this.alpha * this.cutCoul);
    const erfcd = glibcExp(-this.alpha * this.alpha * this.cutCoul * this.cutCoul);
    this.fShift = -(erfcc / this.cutCoulSq + 2.0 / MY_PIS * this.alpha * erfcd / this.cutCoul);
    this.eShift = erfcc / this.cutCoul - this.fShift * this.cutCoul;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const alpha = this.alpha;
    const eShift = this.eShift, fShift = this.fShift;
    const cutCoulSq = this.cutCoulSq;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qtmp = q[i];
      const ti = type[i] * nt;
      // self coulombic energy (native ev_tally(i, i, ...) with the whole e_self on the owned atom)
      const eSelf = -(eShift / 2.0 + alpha / MY_PIS) * qtmp * qtmp * qqrd2e;
      ecoul += eSelf;
      if (pc.eatom) pc.eatom[i] += eSelf;
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
        let r = 0, prefactor = 0, erfcc = 0, forcecoul = 0, ec = 0;
        if (rsq < cutCoulSq) {
          r = Math.sqrt(rsq);
          prefactor = qqrd2e * qtmp * q[j] / r;
          const arg = alpha * r;
          const erfcd = expmsq(arg);
          erfcc = myErfcx(arg) * erfcd;
          forcecoul = prefactor * (erfcc / r + 2.0 * alpha / MY_PIS * erfcd + r * fShift) * r;
          if (factorCoul < 1.0) forcecoul -= (1.0 - factorCoul) * prefactor;
        }
        const lj = this.bornTerm(t, rsq, r2inv, factorLJ);
        const fpair = (forcecoul + factorLJ * lj.forceborn) * r2inv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (rsq < cutCoulSq) {
          ec = prefactor * (erfcc - r * eShift - rsq * fShift);
          if (factorCoul < 1.0) ec -= (1.0 - factorCoul) * prefactor;
        }
        evdwl += lj.evdwl;
        ecoul += ec;
        if (tally) tallyAtom(pc, i, j, lj.evdwl + ec, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }
}

/* ----------------------------------------------------------------------
   born/coul/wolf/cs
   ---------------------------------------------------------------------- */

/**
 * pair_style born/coul/wolf/cs alpha cutoff (cutoff2). Native compute(): e_shift = erfc(a rc)/rc,
 * f_shift = -(e_shift + 2 a/MY_PIS exp(-a^2 rc^2))/rc, self energy -(e_shift/2 + a/MY_PIS) q^2 qqrd2e, and per pair
 * v_sh = (erfc(a r) - e_shift r) prefactor, dvdrr = erfc/rsq + 2 a/MY_PIS erfcd/r + f_shift, forcecoul = dvdrr rsq prefactor.
 */
export class PairBornCoulWolfCS extends BornCoulCSBase {
  readonly name: string = 'born/coul/wolf/cs';

  settings(args: string[], _ctx: StyleContext): void {
    this.baseSettings(args);
  }

  initStyle(_ctx: StyleContext): void {
    this.cutCoulSq = this.cutCoul * this.cutCoul;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const alf = this.alpha;
    const cutCoul = this.cutCoul;
    const cutCoulSq = this.cutCoulSq;
    // self and shifted coulombic energy (native compute() start)
    const eShift = glibcErfc(alf * cutCoul) / cutCoul;
    const fShift = -(eShift + 2.0 * alf / MY_PIS * glibcExp(-alf * alf * cutCoul * cutCoul)) / cutCoul;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qtmp = q[i];
      const ti = type[i] * nt;
      const qisq = qtmp * qtmp;
      const eSelf = -(eShift / 2.0 + alf / MY_PIS) * qisq * qqrd2e;
      ecoul += eSelf;
      if (pc.eatom) pc.eatom[i] += eSelf;
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
        let r = 0, prefactor = 0, vSh = 0, forcecoul = 0, ec = 0;
        if (rsq < cutCoulSq) {
          r = Math.sqrt(rsq);
          prefactor = qqrd2e * qtmp * q[j] / r;
          const erfcc = glibcErfc(alf * r);
          const erfcd = glibcExp(-alf * alf * r * r);
          vSh = (erfcc - eShift * r) * prefactor;
          const dvdrr = (erfcc / rsq + 2.0 * alf / MY_PIS * erfcd / r) + fShift;
          forcecoul = dvdrr * rsq * prefactor;
          if (factorCoul < 1.0) forcecoul -= (1.0 - factorCoul) * prefactor;
        }
        const lj = this.bornTerm(t, rsq, r2inv, factorLJ);
        const fpair = (forcecoul + factorLJ * lj.forceborn) * r2inv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (rsq < cutCoulSq) {
          ec = vSh;
          if (factorCoul < 1.0) ec -= (1.0 - factorCoul) * prefactor;
        }
        evdwl += lj.evdwl;
        ecoul += ec;
        if (tally) tallyAtom(pc, i, j, lj.evdwl + ec, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }
}
