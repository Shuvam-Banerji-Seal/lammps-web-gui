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
 * The field of the pair term carries no qqrd2e (native prefactorE = q_j / r), unlike coul/long/dielectric.
 */
import { PairLJLongCoulLong, type DispTable } from '../../engine/force/pair/lj_long';
import { evTally, StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { ErfcTableCache, erfcPoly, EWALD_F, type ErfcTable } from '../../engine/force/erfc';
import { makeDispTable } from '../pair_disp/dispTable';
import { dielArrays } from '../dielectric/common';
import { exp as glibcExp } from '../libm/exp';
import { selfField } from './dielectric';

export class PairLJLongCoulLongDielectric extends PairLJLongCoulLong {
  readonly name: string = 'lj/long/coul/long/dielectric';
  needsFull = true;
  needsHalf = false;
  virialFdotr = false;
  explicitVirial = true;
  efield = new Float64Array(0);
  epot = new Float64Array(0);
  private readonly dieTables = new ErfcTableCache();
  private dieDispTab: DispTable | null = null;
  private dieDispKey: number[] = [];

  /** The dispersion table of pair_modify table/disp N (native Pair::init_tables_disp), rebuilt when its inputs change. */
  private dispTableOf(gd: number): DispTable | null {
    if (!this.ljLong || this.tableDisp <= 0) return null;
    const key = [this.tableDisp, gd, this.cutGlobal, this.tabinnerDisp];
    if (this.dieDispTab === null || key.some((v, i) => v !== this.dieDispKey[i])) {
      this.dieDispTab = makeDispTable(this.tableDisp, gd, this.cutGlobal, this.tabinnerDisp);
      this.dieDispKey = key;
    }
    return this.dieDispTab;
  }

  compute(pc: PairCompute): void {
    const d = dielArrays(pc.s, 'pair lj/long/coul/long/dielectric');
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
    const cutcoulsq = order1 ? this.cutCoul * this.cutCoul : 0;
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
        if (rsq >= this.cutsq[t]) continue;
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
        // ev_tally_full: each ordered pair carries half of the energies and of the virial
        evTally(pc, i, j, 0.5 * evdwl, 0.5 * ecoul, 0.5 * fpairI, dx, dy, dz);
      }
    }
  }

  single(): never {
    throw new StyleError('pair style lj/long/coul/long/dielectric: single() (pair/local, pair_write) is not supported by the browser engine');
  }
}
