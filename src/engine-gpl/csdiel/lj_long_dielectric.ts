/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/DIELECTRIC/pair_lj_long_coul_long_dielectric.cpp
 * and .h (contributing author Trung Nguyen, Northwestern). The base style is src/KSPACE/pair_lj_long_coul_long.cpp at b332f28
 * (contributing author Pieter J. in 't Veld, SNL; the dispersion tabulation was added by Wayne Mitchell). Copyright Sandia Corporation (GNU General Public License). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style lj/long/coul/long/dielectric flag_lj flag_coul cutoff (cutoff2) (docs.lammps.org/pair_dielectric.html): the
 * lj/long/coul/long real-space terms with the Coulomb charges scaled by the dielectric constants (the force on i carries
 * eps_i, the energy 0.5 (eps_i + eps_j)), a full neighbor list, and the electric field and potential of the pair per owned
 * atom. The dispersion real-space series and its table (pair_modify table/disp), the special-bond LJ weight of the dispersion
 * kernel (the t = rn (1 - f) terms of the native compute()), the Coulomb table and the A&S erfc follow the native compute().
 * The field of the pair term carries no qqrd2e (native prefactorE = q_j / r), unlike coul/long/dielectric. single() (compute
 * pair/local) is the native single() of the same file (pair_lj_long_coul_long_dielectric.cpp at b332f28).
 */
import { PairLJLongCoulLong, type DispTable } from '../../engine/force/pair/lj_long';
import { evTallyFull, StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { ErfcTableCache, erfcPoly, EWALD_F, type ErfcTable } from '../../engine/force/erfc';
import { makeDispTable } from '../pair_disp/dispTable';
import { dielArrays, type DielArrays } from '../dielectric/common';
import { exp as glibcExp } from '../libm/exp';
import { selfField, singleArrays, singleEps } from './dielectric';

/** EWALD_P and the A&S coefficients of the erfc polynomial in native ewald_const.h (the same as erfc.ts erfcPoly). */
const EWALD_P = 0.3275911, A1 = 0.254829592, A2 = -0.284496736, A3 = 1.421413741, A4 = -1.453152027, A5 = 1.061405429;

export class PairLJLongCoulLongDielectric extends PairLJLongCoulLong {
  readonly name: string = 'lj/long/coul/long/dielectric';
  needsFull = true;
  needsHalf = false;
  virialFdotr = false;
  explicitVirial = true;
  efield = new Float64Array(0);
  epot = new Float64Array(0);
  private readonly dieTables = new ErfcTableCache();
  /** The Coulomb table of single(): native's double rsq lookup (lj/long/coul/long takes the double fraction). */
  private readonly singleTables = new ErfcTableCache();
  /** The per-atom arrays and Coulomb prefactor of the last compute() (single() reads them). */
  private dielSingle: DielArrays | null = null;
  private dielQqrd2e = 1;
  private dieDispTab: DispTable | null = null;
  private dieDispKey: number[] = [];

  /** The dispersion table of pair_modify table/disp N (native Pair::init_tables_disp), rebuilt when its inputs change. */
  private dispTableOf(gd: number): DispTable | null {
    if (!this.ljLong || this.tableDisp <= 0) return null;
    // init_tables_disp(cut_lj_global) of init_style: the global LJ cutoff of the run's init (dispCut)
    const key = [this.tableDisp, gd, this.dispCut, this.tabinnerDisp];
    if (this.dieDispTab === null || key.some((v, i) => v !== this.dieDispKey[i])) {
      this.dieDispTab = makeDispTable(this.tableDisp, gd, this.dispCut, this.tabinnerDisp);
      this.dieDispKey = key;
    }
    return this.dieDispTab;
  }

  compute(pc: PairCompute): void {
    const d = dielArrays(pc.s, 'pair lj/long/coul/long/dielectric');
    this.dielSingle = d;
    this.dielQqrd2e = pc.qqrd2e;
    const list = pc.full;
    if (!list) throw new StyleError('pair lj/long/coul/long/dielectric needs a full neighbor list');
    const nl = pc.nlocal;
    if (this.efield.length < 3 * nl) { this.efield = new Float64Array(3 * nl); this.epot = new Float64Array(nl); }
    const E = this.efield, P = this.epot;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nt = this.ntypes + 1;
    const qqrd2e = pc.qqrd2e;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const order1 = this.coulLongFlag, order6 = this.ljLong;
    const g = this.gEwald, gd = this.gEwaldDisp;
    const g2 = gd * gd, g6 = g2 * g2 * g2, g8 = g6 * g2;
    // cut_coulsq of the run's init, and cutsq as init_one last set it (src/KSPACE/pair_lj_long_coul_long.cpp)
    const cutcoulsq = order1 ? this.coulSqInit : 0;
    const outerSq = this.outerSq;
    const coulTab: ErfcTable | null = order1 ? this.dieTables.get(this.table, g, cutcoulsq, this.tabinnerRsq) : null;
    const dispTab = order6 ? this.dispTableOf(gd) : null;
    const { lj1, lj2, lj3, lj4, offset } = this;
    const cutljsq = new Float64Array(nt * nt);
    for (let a = 1; a <= this.ntypes; a++) for (let b = 1; b <= this.ntypes; b++) cutljsq[a * nt + b] = this.p.get('cut', a, b) ** 2;
    const nb = list.neighbors;
    const qs = d.qs, eps = d.epsilon;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = qs[i], ei = eps[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      selfField(d, i, E);
      P[i] = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const ni = jj >>> SBBITS;
        const factorCoul = sC[ni];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= outerSq[t]) continue;
        const oj = owner[j];
        const qj = qs[oj], ej = eps[oj];
        const r2inv = 1.0 / rsq;
        const r = Math.sqrt(rsq);

        // Coulomb: force, field, energy (native order; the table branch holds qqrd2e in its kernels)
        let forceCoul = 0, efieldI = 0, ecoul = 0;
        if (order1 && rsq < cutcoulsq) {
          if (!coulTab || rsq <= coulTab.inner) {
            const grij = g * r;
            const expm2 = glibcExp(-grij * grij);
            const erfc = erfcPoly(grij, expm2);
            const prefactor = qqrd2e * qi * qj / r;
            forceCoul = prefactor * (erfc + EWALD_F * grij * expm2);
            const prefactorE = qj / r;
            efieldI = prefactorE * (erfc + EWALD_F * grij * expm2);
            ecoul = prefactor * 0.5 * (ei + ej) * erfc;
            if (factorCoul < 1.0) {
              forceCoul -= (1.0 - factorCoul) * prefactor;
              efieldI -= (1.0 - factorCoul) * prefactorE;
              ecoul -= (1.0 - factorCoul) * prefactor;
            }
          } else {
            const qq = qqrd2e * qi * qj;
            forceCoul = qq * coulTab.force(rsq);
            efieldI = qj * coulTab.force(rsq);
            ecoul = qq * 0.5 * (ei + ej) * coulTab.energy(rsq);
            if (factorCoul < 1.0) {
              const bare = qq * coulTab.coul(rsq);
              forceCoul -= (1.0 - factorCoul) * bare;
              efieldI -= (1.0 - factorCoul) * qj * coulTab.coul(rsq);
              ecoul -= (1.0 - factorCoul) * bare;
            }
          }
        }

        // LJ or dispersion: native branches of compute()
        let forceLJ = 0, evdwl = 0;
        if (rsq < cutljsq[t]) {
          const rn0 = r2inv * r2inv * r2inv;
          const A = lj3[t], C6 = lj4[t];
          if (order6) {
            if (!dispTab || rsq <= dispTab.inner) {
              // series real space (native: x2 = g^2 r^2, a2 = 1 / x2, the damped kernel times lj4)
              const x2raw = g2 * rsq;
              const a2 = 1.0 / x2raw;
              const x2 = a2 * glibcExp(-x2raw) * C6;
              const series = g8 * (((6.0 * a2 + 6.0) * a2 + 3.0) * a2 + 1.0) * x2 * rsq;
              const eser = g6 * ((a2 + 1.0) * a2 + 0.5) * x2;
              if (ni === 0) {
                const rn = rn0 * rn0;
                forceLJ = rn * lj1[t] - series;
                evdwl = rn * A - eser;
              } else {
                const fs = sLJ[ni], tt = rn0 * (1.0 - fs);
                const rn = rn0 * rn0;
                forceLJ = fs * rn * lj1[t] - series + tt * lj2[t];
                evdwl = fs * rn * A - eser + tt * C6;
              }
            } else {
              // table real space: the dispersion table gives the kernels without lj4
              const fdisp = dispTab.force(rsq), edisp = dispTab.energy(rsq);
              if (ni === 0) {
                const rn = rn0 * rn0;
                forceLJ = rn * lj1[t] - fdisp * C6;
                evdwl = rn * A - edisp * C6;
              } else {
                const fs = sLJ[ni], tt = rn0 * (1.0 - fs);
                const rn = rn0 * rn0;
                forceLJ = fs * rn * lj1[t] - fdisp * C6 + tt * lj2[t];
                evdwl = fs * rn * A - edisp * C6 + tt * C6;
              }
            }
          } else if (ni === 0) {
            forceLJ = rn0 * (rn0 * lj1[t] - lj2[t]);
            evdwl = rn0 * (rn0 * A - C6) - offset[t];
          } else {
            const fs = sLJ[ni];
            forceLJ = fs * rn0 * (rn0 * lj1[t] - lj2[t]);
            evdwl = fs * (rn0 * (rn0 * A - C6) - offset[t]);
          }
        }

        const fpairI = (forceCoul * ei + forceLJ) * r2inv;
        f[3 * i] += dx * fpairI; f[3 * i + 1] += dy * fpairI; f[3 * i + 2] += dz * fpairI;
        P[i] += efieldI;
        efieldI *= ei * r2inv;
        E[3 * i] += dx * efieldI; E[3 * i + 1] += dy * efieldI; E[3 * i + 2] += dz * efieldI;
        // Pair::ev_tally_full: half of the energies and of the virial of each ordered pair, to atom i
        evTallyFull(pc, i, evdwl, ecoul, fpairI, dx, dy, dz);
      }
    }
  }

  /**
   * As native LAMMPS does (src/DIELECTRIC/pair_lj_long_coul_long_dielectric.cpp single at b332f28): the Coulomb term inside
   * cut_coulsq, the series real space (the special term r = qq (1 - factor_coul) / r, the force on i with eps_i) or
   * the table real space (rsq above the table's tabinner; the special term rounded to float as native's union), the LJ
   * or dispersion term of the native single (its series with the factor_lj weight, no dispersion table), and ei = 0 for
   * eps == 1. The field and the potential are compute()'s (not used here). The charges are the arrays of the last compute.
   */
  single(i: number, j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, _qi = 0, _qj = 0): { eng: number; fforce: number } {
    const d = singleArrays(this.dielSingle, 'pair lj/long/coul/long/dielectric');
    const qqrd2e = this.dielQqrd2e;
    const q = d.qs, eps = d.epsilon;
    const nt = this.ntypes + 1;
    const t = itype * nt + jtype;
    const order1 = this.coulLongFlag, order6 = this.ljLong;
    const g = this.gEwald, gd = this.gEwaldDisp;
    const g2 = gd * gd, g6 = g2 * g2 * g2, g8 = g6 * g2;
    const cutCoulSq = this.coulSqInit;
    const cutLJSq = this.p.get('cut', itype, jtype) ** 2;
    const { lj1, lj2, lj3, lj4, offset } = this;
    const ei = singleEps(eps[i]), ej = singleEps(eps[j]);
    const r2inv = 1.0 / rsq;
    let eng = 0.0, forceCoul = 0.0;
    if (order1 && rsq < cutCoulSq) {
      const coulTab = this.singleTables.get(this.table, g, cutCoulSq, this.tabinnerRsq, true);
      const qiqj = q[i] * q[j];
      if (!coulTab || rsq <= coulTab.inner) {
        const r = Math.sqrt(rsq), x = g * r;
        let s = qqrd2e * q[i] * q[j];
        const tp = 1.0 / (1.0 + EWALD_P * x);
        const special = s * (1.0 - factorCoul) / r;
        s *= g * glibcExp(-x * x);
        const poly = ((((tp * A5 + A4) * tp + A3) * tp + A2) * tp + A1);
        const tn = tp * (poly * s / x);
        forceCoul = tn + EWALD_F * s - special;
        eng += (tn - special) * (ei + ej) * 0.5;
      } else {
        // native: the table holds qqrd2e; the special term is a float (t.f = (1 - factor_coul) ctable, rounded once)
        const bare = Math.fround((1.0 - factorCoul) * (qqrd2e * coulTab.coul(rsq)));
        forceCoul = qiqj * (qqrd2e * coulTab.force(rsq) - bare);
        eng += qiqj * (qqrd2e * coulTab.energy(rsq) - bare) * (ei + ej) * 0.5;
      }
    }

    let forceLJ = 0.0;
    if (rsq < cutLJSq) {
      let r6inv = r2inv * r2inv * r2inv;
      if (order6) {
        // native single: the dispersion series (x2 = g^2 r^2, a2 = 1 / x2), with "+ a2" in the force polynomial
        const x2raw = g2 * rsq;
        const a2 = 1.0 / x2raw;
        const tt = r6inv * (1.0 - factorLJ);
        const x2 = a2 * glibcExp(-x2raw) * lj4[t];
        r6inv *= r6inv;
        forceLJ = factorLJ * r6inv * lj1[t] - g8 * (((6.0 * a2 + 6.0) * a2 + 3.0) * a2 + a2) * x2 * rsq + tt * lj2[t];
        eng += factorLJ * r6inv * lj3[t] - g6 * ((a2 + 1.0) * a2 + 0.5) * x2 + tt * lj4[t];
      } else {
        forceLJ = factorLJ * r6inv * (lj1[t] * r6inv - lj2[t]);
        eng += factorLJ * (r6inv * (r6inv * lj3[t] - lj4[t]) - offset[t]);
      }
    }
    const fforce = (forceCoul * eps[i] + forceLJ) * r2inv;
    return { eng, fforce };
  }
}
