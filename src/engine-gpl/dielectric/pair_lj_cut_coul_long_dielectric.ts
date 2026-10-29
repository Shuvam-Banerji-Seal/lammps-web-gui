/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/pair_lj_cut_coul_long_dielectric.cpp and pair_lj_cut_coul_long_dielectric.h at commit b332f28
 * (Trung Nguyen and Monica Olvera de la Cruz, Northwestern University, and the DIELECTRIC package authors). Copyright Sandia Corporation (GNU
 * General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style lj/cut/coul/long/dielectric: the real-space part of the Ewald/PPPM Coulomb sum with the charges scaled by
 * the per-atom dielectric constant (see lj/cut/coul/cut/dielectric). Full neighbor list; the Coulomb energy of a pair is
 * qq 0.5 (eps_i + eps_j) erfc(g r)/r, the Coulomb force on i carries eps_i; the special-bond correction keeps its
 * original form (-(1 - w) qq / r, not scaled by the dielectric constants). The electric field of the pair (efield) and
 * the potential (epot) at each owned atom are stored for the polarization fixes. The erfc is the polynomial fit of
 * LAMMPS (pair_modify table 0), or its interpolation table (default table 12) for r^2 >= 2, as the core coul/long pair.
 * single() (compute pair/local) is the native single() of pair_lj_cut_coul_long_dielectric.cpp at b332f28.
 */
import { PairLJCutCoulLong } from '../../engine/force/pair/coul_long';
import { evTallyFull, StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { ErfcTableCache, EWALD_F, erfcPoly, TABLE_INNER_RSQ, type ErfcTable } from '../../engine/force/erfc';
import { dielArrays, type DielArrays } from './common';
import { exp as libmExp } from '../libm/exp';
import { singleArrays, singleEps } from '../csdiel/dielectric';

/** Within this distance squared the Coulomb term is left out (pair_lj_cut_coul_long_dielectric.cpp EPSILON). */
const EPSILON = 1.0e-6;
/** sqrt(pi), as LAMMPS MathConst::MY_PIS. */
const MY_PIS = 1.77245385090551602729; // math_const.h (sqrt(pi) rounded once; Math.sqrt(Math.PI) is one ulp below)

/**
 * Coulomb kernels of one pair at distance r (per unit charge product, without qqrd2e): fo = force / r of the damped
 * term, eo = erfc(g r) / r (energy), co = 1 / r (the bare term of the special-bond correction).
 */
const kernels = (rsq: number, g: number, table: ErfcTable | null): { fo: number; eo: number; co: number } => {
  if (table && rsq > TABLE_INNER_RSQ) return { fo: table.force(rsq), eo: table.energy(rsq), co: table.coul(rsq) };
  const r = Math.sqrt(rsq);
  const grij = g * r;
  const ex = libmExp(-grij * grij);
  const erfc = erfcPoly(grij, ex);
  return { fo: (erfc + EWALD_F * grij * ex) / r, eo: erfc / r, co: 1 / r };
};

export class PairLJCutCoulLongDielectric extends PairLJCutCoulLong {
  readonly name: string = 'lj/cut/coul/long/dielectric';
  /** native sets no_virial_fdotr_compute: the virial is the sum of the ev_tally_full tallies (forces on i only) */
  override virialFdotr = false;
  needsFull = true;
  needsHalf = false;
  /** Electric field and potential at each owned atom from this pair (set by compute()). */
  efield = new Float64Array(0);
  epot = new Float64Array(0);
  private readonly dieTables = new ErfcTableCache();
  /** The table of single() (its own cache: the key includes tabinner, which compute() leaves at the default). */
  private readonly singleTables = new ErfcTableCache();
  /** The per-atom arrays and Coulomb prefactor of the last compute() (single() reads them). */
  private dielSingle: DielArrays | null = null;
  private qqrd2eSingle = 1;

  compute(pc: PairCompute): void {
    const d = dielArrays(pc.s, 'pair lj/cut/coul/long/dielectric');
    this.dielSingle = d;
    this.qqrd2eSingle = pc.qqrd2e;
    const list = pc.full;
    if (!list) throw new StyleError('pair lj/cut/coul/long/dielectric needs a full neighbor list');
    const nl = pc.nlocal;
    if (this.efield.length < 3 * nl) {
      this.efield = new Float64Array(3 * nl);
      this.epot = new Float64Array(nl);
    }
    const nt = this.ntypes + 1;
    const cutljsq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) cutljsq[i * nt + j] = this.p.get('cut', i, j) ** 2;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.dieTables.get(this.table, g, cutcsq);
    const { lj1, lj2, lj3, lj4, offset } = this;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const qs = d.qs, eps = d.epsilon, area = d.area, curv = d.curvature, mu = d.mu;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const E = this.efield, P = this.epot;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = qs[i], ei = eps[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      const threshold = Math.sqrt(area[i]);
      if (curv[i] < threshold) {
        const sf = curv[i] / (4.0 * MY_PIS * threshold) * area[i] * qi;
        E[3 * i] = sf * mu[4 * i]; E[3 * i + 1] = sf * mu[4 * i + 1]; E[3 * i + 2] = sf * mu[4 * i + 2];
      } else {
        E[3 * i] = E[3 * i + 1] = E[3 * i + 2] = 0;
      }
      P[i] = 0;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const factorLJ = sLJ[sb], factorCoul = sC[sb];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= this.cutsq[t]) continue;
        const oj = owner[j];
        const qj = qs[oj], ej = eps[oj];
        const r2inv = 1.0 / rsq;
        let forcecoul = 0, efieldI = 0, epotI = 0, ecoul = 0;
        if (rsq < cutcsq && rsq > EPSILON) {
          const kr = kernels(rsq, g, tab);
          const qq = qqrd2e * qi * qj;
          const qE = qqrd2e * qj;
          forcecoul = qq * kr.fo;
          efieldI = qE * kr.fo;
          if (factorCoul < 1.0) {
            forcecoul -= (1.0 - factorCoul) * qq * kr.co;
            efieldI -= (1.0 - factorCoul) * qE * kr.co;
          }
          epotI = efieldI;
          ecoul = qq * 0.5 * (ei + ej) * kr.eo;
          if (factorCoul < 1.0) ecoul -= (1.0 - factorCoul) * qq * kr.co;
        }
        let forcelj = 0, evdwl = 0;
        if (rsq < cutljsq[t]) {
          const r6inv = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
          evdwl = factorLJ * (r6inv * (lj3[t] * r6inv - lj4[t]) - offset[t]);
        }
        const fpairI = (forcecoul * ei + factorLJ * forcelj) * r2inv;
        fxi += dx * fpairI; fyi += dy * fpairI; fzi += dz * fpairI;
        const ef = efieldI * ei * r2inv;
        E[3 * i] += dx * ef; E[3 * i + 1] += dy * ef; E[3 * i + 2] += dz * ef;
        P[i] += epotI;
        // Pair::ev_tally_full: half of the energy and of the virial of each ordered pair, to atom i
        evTallyFull(pc, i, evdwl, ecoul, fpairI, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
  }

  /**
   * As native LAMMPS does (src/DIELECTRIC/pair_lj_cut_coul_long_dielectric.cpp single at b332f28): inside cut_coul the
   * direct A&S erfc (no table, or rsq <= tabinner) or the table of pair_modify table N (its ftable, ctable and etable hold
   * qqrd2e), the special-bond correction -(1 - factor_coul) prefactor (not scaled by eps) in the energy too, the force on i
   * with eps_i (fforce = (forcecoul eps_i + factor_lj forcelj) / r^2), and ei = 0 for eps == 1.
   */
  single(i: number, j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, _qi = 0, _qj = 0): { eng: number; fforce: number } {
    const d = singleArrays(this.dielSingle, 'pair lj/cut/coul/long/dielectric');
    const qqrd2e = this.qqrd2eSingle;
    const q = d.qs, eps = d.epsilon;
    const nt = this.ntypes + 1;
    const t = itype * nt + jtype;
    const cutCoulSq = this.cutCoul * this.cutCoul;
    const cutLJSq = this.p.get('cut', itype, jtype) ** 2;
    const g = this.gEwald;
    const r2inv = 1.0 / rsq;
    const ei = singleEps(eps[i]), ej = singleEps(eps[j]);
    let forcecoul = 0, eng = 0;
    if (rsq < cutCoulSq) {
      const tab = this.singleTables.get(this.table, g, cutCoulSq, this.tabinnerRsq);
      if (!tab || rsq <= tab.inner) {
        const r = Math.sqrt(rsq);
        const grij = g * r;
        const expm2 = libmExp(-grij * grij);
        const erfc = erfcPoly(grij, expm2);
        const prefactor = qqrd2e * q[i] * q[j] / r;
        forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
        let phicoul = prefactor * 0.5 * (ei + ej) * erfc;
        if (factorCoul < 1.0) {
          forcecoul -= (1.0 - factorCoul) * prefactor;
          phicoul -= (1.0 - factorCoul) * prefactor;
        }
        eng += phicoul;
      } else {
        // the table holds qqrd2e in native (ftable, ctable, etable); the engine's table is per unit product
        const qq = qqrd2e * q[i] * q[j];
        forcecoul = qq * tab.force(rsq);
        let phicoul = qq * 0.5 * (ei + ej) * tab.energy(rsq);
        if (factorCoul < 1.0) {
          const prefactor = qq * tab.coul(rsq);
          forcecoul -= (1.0 - factorCoul) * prefactor;
          phicoul -= (1.0 - factorCoul) * prefactor;
        }
        eng += phicoul;
      }
    }
    let forcelj = 0;
    if (rsq < cutLJSq) {
      const r6inv = r2inv * r2inv * r2inv;
      forcelj = r6inv * (this.lj1[t] * r6inv - this.lj2[t]);
      eng += factorLJ * (r6inv * (this.lj3[t] * r6inv - this.lj4[t]) - this.offset[t]);
    }
    const fforce = (forcecoul * eps[i] + factorLJ * forcelj) * r2inv;
    return { eng, fforce };
  }
}
