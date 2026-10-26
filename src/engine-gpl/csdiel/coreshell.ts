/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/CORESHELL/pair_coul_wolf_cs.cpp and
 * pair_coul_wolf_cs.h, src/CORESHELL/pair_lj_cut_coul_long_cs.cpp and .h (contributing author Hendrik Heenen, hendrik.heenen@mytum.de),
 * src/CORESHELL/pair_coul_long_cs.cpp and .h (contributing author Hendrik Heenen), and src/CORESHELL/pair_lj_class2_coul_long_cs.cpp
 * and .h (the native files of coul/wolf/cs and lj/class2/coul/long/cs carry no contributing-author line). The base styles are
 * src/pair_coul_wolf.cpp, src/KSPACE/pair_coul_long.cpp, src/KSPACE/pair_lj_cut_coul_long.cpp and
 * src/CLASS2/pair_lj_class2_coul_long.cpp at b332f28.
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style coul/wolf/cs, coul/long/cs, lj/cut/coul/long/cs and lj/class2/coul/long/cs: the core-shell variants. Each is its base style
 * with the Coulomb term of a core and its shell (a special pair whose 1-2 weight is 0) evaluated at r + EPSILON, so the
 * r -> 0 limit is finite (docs.lammps.org/pair_cs.html: "except that they correctly treat the special case where the
 * distance between two charged core and shell atoms in the same core/shell pair approach r = 0.0").
 * The loops keep the operation order of the native compute() functions (f[i] accumulated pair by pair, the native erfc
 * polynomial with EWALD_P and the B0..B5 fit, erf() from glibc for the excluded-pair branch, exp() from glibc).
 * Energies are tallied by the force field from pc.acc; the global virial is the force-field dot product (virialFdotr).
 */
import { StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { PairCoulLong, PairLJCutCoulLong } from '../../engine/force/pair/coul_long';
import { PairCoulWolf } from '../../engine/force/pair/coul';
import { PairLJClass2Coul } from '../../engine/force/pair/class2_spica51';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { ErfcTableCache, type ErfcTable } from '../../engine/force/erfc';
import { erf as glibcErf, erfc as glibcErfc } from '../libm/erf';
import { pow as libmPow } from '../libm/pow';
import { exp as glibcExp } from '../libm/exp';

/** EPSILON of the /cs computes: added to rsq so that r = 0 (a core and its shell) gives a finite Coulomb term. */
const EPSILON = 1.0e-20;
/** sqrt(pi), MathConst::MY_PIS. */
const MY_PIS = 1.77245385090551602729;
/** 1/sqrt(pi/4) = 2/sqrt(pi), MathConst::MY_ISPI4 (the force term of an excluded pair). */
const MY_ISPI4 = 1.12837916709551257389;
/** The Ewald force constant the native cs computes use (rounded, as native). */
const EWALD_F = 1.12837917;
/** Polynomial erfc of the native lj/cut/coul/long and lj/class2/coul/long cs computes. */
const EWALD_P = 9.95473818e-1;
const B0 = -0.1335096380159268;
const B1 = -2.57839507e-1;
const B2 = -1.37203639e-1;
const B3 = -8.88822059e-3;
const B4 = -5.80844129e-3;
const B5 = 1.14652755e-1;

/**
 * The Coulomb term of one pair of a long-range /cs style (native lj/cut/coul/long/cs compute, the same in
 * lj/class2/coul/long/cs): the force (times r, without the 1/r^2 of fpair) and the energy. fc = special_coul weight.
 * Without a table (pair_modify table 0, or rsq <= tabinner) the excluded pair (fc < 1) uses erf(); a included pair the
 * polynomial erfc. With a table the tabulated kernels are used, the special correction subtracting (1 - fc) times the
 * bare 1/r term (the tabulated qqrd2e/r).
 */
const csLongCoul = (
  rsq: number, qi: number, qj: number, fc: number, g: number, qqrd2e: number, tab: ErfcTable | null,
): { force: number; e: number } => {
  if (tab && rsq > tab.inner) {
    const qq = qqrd2e * qi * qj;
    let force = qq * tab.force(rsq);
    let e = qq * tab.energy(rsq);
    if (fc < 1.0) {
      const bare = qq * tab.coul(rsq);
      force -= (1.0 - fc) * bare;
      e -= (1.0 - fc) * bare;
    }
    return { force, e };
  }
  const r = Math.sqrt(rsq);
  const grij = g * r;
  const expm2 = glibcExp(-grij * grij);
  if (fc < 1.0) {
    // for excluded pairs the correction nearly cancels the Ewald term: use erf(); erfc holds erfc - (1 - factor_coul)
    const erfc = fc - glibcErf(grij);
    const prefactor = qqrd2e * qi * qj / r;
    return { force: prefactor * (erfc + MY_ISPI4 * grij * expm2), e: prefactor * erfc };
  }
  const t = 1.0 / (1.0 + EWALD_P * grij);
  const u = 1.0 - t;
  const erfc = t * (1.0 + u * (B0 + u * (B1 + u * (B2 + u * (B3 + u * (B4 + u * B5)))))) * expm2;
  const prefactor = qqrd2e * qi * qj / r;
  return { force: prefactor * (erfc + EWALD_F * grij * expm2), e: prefactor * erfc };
};

/** Shared refusal for single() (native single_enable = 0 in all three styles). */
const noSingle = (name: string): never => {
  throw new StyleError(`pair style ${name}: single() (pair/local, pair_write) is not supported (native single_enable = 0)`);
};

/* ----------------------------------------------------------------------
   coul/wolf/cs
   ---------------------------------------------------------------------- */

/**
 * pair_style coul/wolf/cs alpha cutoff (docs.lammps.org/pair_cs.html; pair_coul.html for coul/wolf). Native compute():
 * e_shift = erfc(alf rc)/rc, f_shift = -(e_shift + 2 alf/MY_PIS exp(-alf^2 rc^2))/rc, self energy
 * -(e_shift/2 + alf/MY_PIS) q^2 qqrd2e per owned atom, and for each pair within rc with rsq += EPSILON:
 * v_sh = (erfc(alf r) - e_shift r) qqrd2e q_i q_j / r, dvdrr = erfc/rsq + 2 alf/MY_PIS erfcd/r + f_shift, the special
 * correction -(1 - fc) qqrd2e q_i q_j / r.
 */
export class PairCoulWolfCS extends PairCoulWolf {
  readonly name: string = 'coul/wolf/cs';

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, q } = pc;
    const alf = this.alpha;
    const rc = this.cutGlobal;
    const cutCoulsq = rc * rc;
    const eShift = glibcErfc(alf * rc) / rc;
    const fShift = -(eShift + 2.0 * alf / MY_PIS * glibcExp(-alf * alf * rc * rc)) / rc;
    const sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qtmp = q[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qisq = qtmp * qtmp;
      // self energy of the owned atom (native ev_tally(i, i, ...) with the newton-off half weights: the whole of e_self)
      const eSelf = -(eShift / 2.0 + alf / MY_PIS) * qisq * qqrd2e;
      ecoul += eSelf;
      if (pc.eatom) pc.eatom[i] += eSelf;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const fc = sC[jj >>> SBBITS];
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        let rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutCoulsq) {
          rsq += EPSILON;
          // EPSILON is added for r = 0: the interaction is removed by the special bond
          const r = Math.sqrt(rsq);
          const prefactor = qqrd2e * qtmp * q[j] / r;
          const erfcc = glibcErfc(alf * r);
          const erfcd = glibcExp(-alf * alf * r * r);
          const vsh = (erfcc - eShift * r) * prefactor;
          const dvdrr = (erfcc / rsq + 2.0 * alf / MY_PIS * erfcd / r) + fShift;
          let forcecoul = dvdrr * rsq * prefactor;
          if (fc < 1.0) forcecoul -= (1.0 - fc) * prefactor;
          const fpair = forcecoul / rsq;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          let e = vsh;
          if (fc < 1.0) e -= (1.0 - fc) * prefactor;
          ecoul += e;
          if (tally) tallyAtom(pc, i, j, e, fpair, delx, dely, delz);
        }
      }
    }
    pc.acc.ecoul += ecoul;
  }

  single(): never { return noSingle(this.name); }
}

/* ----------------------------------------------------------------------
   coul/long/cs
   ---------------------------------------------------------------------- */

/**
 * pair_style coul/long/cs cutoff (docs.lammps.org/pair_cs.html): coul/long with the native PairCoulLongCS compute(). A pair
 * within the Coulomb cutoff (tested before EPSILON is added) gets the Coulomb term of csLongCoul; f[i] is accumulated pair
 * by pair, as native.
 */
export class PairCoulLongCS extends PairCoulLong {
  readonly name: string = 'coul/long/cs';
  private readonly csTables = new ErfcTableCache();

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, q } = pc;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.csTables.get(this.table, g, cutcsq, this.tabinnerRsq);
    const sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq0 = dx * dx + dy * dy + dz * dz;
        if (rsq0 >= cutcsq) continue;
        // EPSILON for r = 0 (a core and its shell): the interaction is removed by the special bond
        const rsq = rsq0 + EPSILON;
        const r2inv = 1.0 / rsq;
        const c = csLongCoul(rsq, qi, q[j], sC[jj >>> SBBITS], g, qqrd2e, tab);
        const fpair = c.force * r2inv;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        ecoul += c.e;
        if (tally) tallyAtom(pc, i, j, c.e, fpair, dx, dy, dz);
      }
    }
    pc.acc.ecoul += ecoul;
  }

  single(): never { return noSingle(this.name); }
}

/* ----------------------------------------------------------------------
   lj/cut/coul/long/cs
   ---------------------------------------------------------------------- */

/**
 * pair_style lj/cut/coul/long/cs cutoff (cutoff2) (docs.lammps.org/pair_cs.html). The LJ term is that of lj/cut/coul/long;
 * the Coulomb term is csLongCoul. The per-pair table selection and the special-bond weights follow native compute().
 */
export class PairLJCutCoulLongCS extends PairLJCutCoulLong {
  readonly name: string = 'lj/cut/coul/long/cs';
  private readonly csTables = new ErfcTableCache();

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { lj1, lj2, lj3, lj4, offset } = this;
    const cutljsq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) cutljsq[i * nt + j] = this.p.get('cut', i, j) ** 2;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.csTable(g);
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const factorLJ = sLJ[sb], factorCoul = sC[sb];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq0 = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq0 >= this.cutsq[t]) continue;
        // EPSILON for r = 0 (a core and its shell): the interaction is removed by the special bond
        const rsq = rsq0 + EPSILON;
        const r2inv = 1.0 / rsq;
        let forcecoul = 0, ec = 0;
        if (rsq < cutcsq) {
          const c = csLongCoul(rsq, qi, q[j], factorCoul, g, qqrd2e, tab);
          forcecoul = c.force;
          ec = c.e;
        }
        let forcelj = 0, ev = 0;
        if (rsq < cutljsq[t]) {
          const r6inv = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
          ev = factorLJ * (r6inv * (lj3[t] * r6inv - lj4[t]) - offset[t]);
        }
        const fpair = (forcecoul + factorLJ * forcelj) * r2inv;
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

  /** The erfc table of pair_modify table N (null for table 0), built from the G-ewald of the kspace style. */
  private csTable(g: number): ErfcTable | null {
    return this.csTables.get(this.table, g, this.cutCoul * this.cutCoul, this.tabinnerRsq);
  }

  single(): never { return noSingle(this.name); }
}

/* ----------------------------------------------------------------------
   lj/class2/coul/long/cs
   ---------------------------------------------------------------------- */

/**
 * pair_style lj/class2/coul/long/cs cutoff (cutoff2) (docs.lammps.org/pair_cs.html; pair_class2.html). The class 2 LJ term
 * in the native form of lj/class2/coul/long: lj1 = 18 eps sigma^9, lj2 = 18 eps sigma^6, lj3 = 2 eps sigma^9,
 * lj4 = 3 eps sigma^6, offset = eps (2 (sigma/rc)^9 - 3 (sigma/rc)^6) with pair_modify shift yes. Mixing and cutoffs are
 * those of the base style (sixthpower, pair_class2.html).
 */
export class PairLJClass2CoulLongCS extends PairLJClass2Coul {
  readonly name: string = 'lj/class2/coul/long/cs';

  constructor() {
    super('lj/class2/coul/long');
  }

  private c2lj1 = new Float64Array(0);
  private c2lj2 = new Float64Array(0);
  private c2lj3 = new Float64Array(0);
  private c2lj4 = new Float64Array(0);
  private c2off = new Float64Array(0);
  private readonly csTables = new ErfcTableCache();

  initOne(i: number, j: number): number {
    const cut = super.initOne(i, j);
    const nt = this.ntypes + 1;
    if (this.c2lj1.length !== nt * nt) {
      this.c2lj1 = new Float64Array(nt * nt); this.c2lj2 = new Float64Array(nt * nt);
      this.c2lj3 = new Float64Array(nt * nt); this.c2lj4 = new Float64Array(nt * nt); this.c2off = new Float64Array(nt * nt);
    }
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j), cutLJ = this.p.get('cut', i, j);
    // native pow() is the C library's (pair_lj_class2_coul_long.cpp init_one)
    const lj1 = 18.0 * eps * libmPow(sig, 9.0);
    const lj2 = 18.0 * eps * libmPow(sig, 6.0);
    const lj3 = 2.0 * eps * libmPow(sig, 9.0);
    const lj4 = 3.0 * eps * libmPow(sig, 6.0);
    let off = 0;
    if (this.shift && cutLJ > 0.0) {
      const ratio = sig / cutLJ;
      off = eps * (2.0 * libmPow(ratio, 9.0) - 3.0 * libmPow(ratio, 6.0));
    }
    for (const k of [i * nt + j, j * nt + i]) {
      this.c2lj1[k] = lj1; this.c2lj2[k] = lj2; this.c2lj3[k] = lj3; this.c2lj4[k] = lj4; this.c2off[k] = off;
    }
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const cutljsq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) cutljsq[i * nt + j] = this.p.get('cut', i, j) ** 2;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.csTables.get(this.table, g, cutcsq, this.tabinnerRsq);
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const c2lj1 = this.c2lj1, c2lj2 = this.c2lj2, c2lj3 = this.c2lj3, c2lj4 = this.c2lj4, c2off = this.c2off;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const factorLJ = sLJ[sb], factorCoul = sC[sb];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq0 = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq0 >= this.cutsq[t]) continue;
        const rsq = rsq0 + EPSILON;
        const r2inv = 1.0 / rsq;
        let forcecoul = 0, ec = 0;
        if (rsq < cutcsq) {
          const c = csLongCoul(rsq, qi, q[j], factorCoul, g, qqrd2e, tab);
          forcecoul = c.force;
          ec = c.e;
        }
        let forcelj = 0, ev = 0;
        if (rsq < cutljsq[t]) {
          const rinv = Math.sqrt(r2inv);
          const r3inv = r2inv * rinv;
          const r6inv = r3inv * r3inv;
          forcelj = r6inv * (c2lj1[t] * r3inv - c2lj2[t]);
          ev = factorLJ * (r6inv * (c2lj3[t] * r3inv - c2lj4[t]) - c2off[t]);
        }
        const fpair = (forcecoul + factorLJ * forcelj) * r2inv;
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

  single(): never { return noSingle(this.name); }
}
