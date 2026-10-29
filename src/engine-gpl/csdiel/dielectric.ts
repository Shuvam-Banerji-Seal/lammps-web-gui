/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/DIELECTRIC/pair_coul_cut_dielectric.cpp
 * and .h, pair_coul_long_dielectric.cpp and .h, and pair_lj_cut_coul_debye_dielectric.cpp and .h (contributing author Trung Nguyen,
 * Northwestern). The base styles are src/pair_coul_cut.cpp, src/pair_coul_long.cpp and src/pair_lj_cut_coul_debye.cpp at b332f28.
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style coul/cut/dielectric, coul/long/dielectric and lj/cut/coul/debye/dielectric (docs.lammps.org/pair_dielectric.html):
 * the base Coulomb term with the charges scaled by the per-atom dielectric constant (q_scaled, q / epsilon), a full neighbor
 * list, the force on i carrying eps_i, and the electric field (efield) and potential (epot) at each owned atom from the pair
 * term, which the polarization fixes read. Energies are tallied per ordered pair with ev_tally_full (half of each pair term
 * per visit); the virial is tallied pair by pair (no_virial_fdotr_compute in all three styles).
 * The arithmetic follows the native compute() order (efield self term of curvature and area, EPSILON = 1.0e-6 cut on
 * the Coulomb term of coul/cut and debye, the A&S polynomial erfc of coul/long, the table branch of pair_modify table N).
 * single() (compute pair/local) of coul/cut/dielectric and lj/cut/coul/debye/dielectric is ported from the same files.
 */
import { PairCoulCut } from '../../engine/force/pair/coul';
import { PairCoulLong } from '../../engine/force/pair/coul_long';
import { PairLJCutCoulDebye } from '../../engine/force/pair/lj_coul';
import { evTallyFull, StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { ErfcTableCache, erfcPoly, EWALD_F } from '../../engine/force/erfc';
import { dielArrays, type DielArrays } from '../dielectric/common';
import { exp as glibcExp } from '../libm/exp';

/** Within this distance squared the Coulomb term of coul/cut and debye is left out (native EPSILON). */
const EPSILON = 1.0e-6;
/** sqrt(pi), MathConst::MY_PIS. */
const MY_PIS = 1.77245385090551602729;

/**
 * The per-atom dielectric arrays of the last compute() (the styles' single() reads q_scaled and epsilon from them, as
 * native does); a StyleError when no force evaluation has set them yet.
 */
export const singleArrays = (d: DielArrays | null, who: string): DielArrays => {
  if (!d) throw new StyleError(`${who}: single() needs a force evaluation first (its per-atom dielectric arrays)`);
  return d;
};

/**
 * The dielectric constant as the native single() of the DIELECTRIC styles reads it: 0 when eps == 1 (native sets ei = 0
 * for an unscreened atom in the energy), else eps.
 */
export const singleEps = (e: number): number => (e === 1 ? 0 : e);

/** The self term of the field at owned atom i (Eq. (55) of Barros et al., as in every native dielectric compute). */
export const selfField = (
  d: ReturnType<typeof dielArrays>, i: number, efield: Float64Array,
): void => {
  const threshold = Math.sqrt(d.area[i]);
  if (d.curvature[i] < threshold) {
    const sf = d.curvature[i] / (4.0 * MY_PIS * threshold) * d.area[i] * d.qs[i];
    efield[3 * i] = sf * d.mu[4 * i];
    efield[3 * i + 1] = sf * d.mu[4 * i + 1];
    efield[3 * i + 2] = sf * d.mu[4 * i + 2];
  } else {
    efield[3 * i] = efield[3 * i + 1] = efield[3 * i + 2] = 0;
  }
};

/* ----------------------------------------------------------------------
   coul/cut/dielectric
   ---------------------------------------------------------------------- */

/**
 * pair_style coul/cut/dielectric cutoff. Native compute(): for each ordered pair with rsq < cutsq and rsq > EPSILON,
 * efield_i = qqrd2e scale q_j / r, forcecoul = q_i efield_i, fpair = factor_coul eps_i forcecoul / r^2, E = factor_coul
 * qqrd2e scale q_i q_j 0.5 (eps_i + eps_j) / r. The energy and the virial are tallied with ev_tally_full.
 */
export class PairCoulCutDielectric extends PairCoulCut {
  readonly name: string = 'coul/cut/dielectric';
  needsFull = true;
  needsHalf = false;
  virialFdotr = false;
  explicitVirial = true;
  /** Electric field at each owned atom from this pair (3 per atom; set by compute()). */
  efield = new Float64Array(0);
  /** The per-atom arrays and Coulomb prefactor of the last compute() (single() reads them, as native reads q_scaled and force->qqrd2e). */
  private dielSingle: DielArrays | null = null;
  private qqrd2eSingle = 1;

  compute(pc: PairCompute): void {
    const d = dielArrays(pc.s, 'pair coul/cut/dielectric');
    this.dielSingle = d;
    this.qqrd2eSingle = pc.qqrd2e;
    const list = pc.full;
    if (!list) throw new StyleError('pair coul/cut/dielectric needs a full neighbor list');
    const nl = pc.nlocal;
    if (this.efield.length < 3 * nl) this.efield = new Float64Array(3 * nl);
    const E = this.efield;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nt = this.ntypes + 1;
    const qqrd2e = pc.qqrd2e;
    const sC = pc.specialCoul;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = d.qs[i], ei = d.epsilon[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      selfField(d, i, E);
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const factorCoul = sC[jj >>> SBBITS];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (!(rsq < this.cutsq[t] && rsq > EPSILON)) continue;
        const oj = owner[j];
        const qj = d.qs[oj], ej = d.epsilon[oj];
        const r2inv = 1.0 / rsq;
        const rinv = Math.sqrt(r2inv);
        let efieldI = qqrd2e * this.scale[t] * qj * rinv;
        const forcecoul = qi * efieldI;
        const fpairI = factorCoul * ei * forcecoul * r2inv;
        f[3 * i] += dx * fpairI; f[3 * i + 1] += dy * fpairI; f[3 * i + 2] += dz * fpairI;
        efieldI *= factorCoul * ei * r2inv;
        E[3 * i] += dx * efieldI; E[3 * i + 1] += dy * efieldI; E[3 * i + 2] += dz * efieldI;
        const ecoul = factorCoul * qqrd2e * this.scale[t] * qi * qj * 0.5 * (ei + ej) * rinv;
        evTallyFull(pc, i, 0, ecoul, fpairI, dx, dy, dz);
      }
    }
  }

  /**
   * As native LAMMPS does (src/DIELECTRIC/pair_coul_cut_dielectric.cpp single at b332f28): no cutoff test; fforce carries
   * eps_i but not factor_coul, and the energy is factor_coul qqrd2e q_i q_j (1/r) 0.5 (ei + ej), with q_scaled (the
   * arrays of the last compute) and ei = 0 for eps == 1. The charges passed in are the unscaled ones and are not used.
   */
  single(i: number, j: number, _itype: number, _jtype: number, rsq: number, factorCoul: number, _factorLJ: number, _qi = 0, _qj = 0): { eng: number; fforce: number } {
    const d = singleArrays(this.dielSingle, 'pair coul/cut/dielectric');
    const qqrd2e = this.qqrd2eSingle;
    const q = d.qs, eps = d.epsilon;
    const r2inv = 1.0 / rsq;
    const fforce = qqrd2e * q[i] * q[j] * Math.sqrt(r2inv) * eps[i];
    const ei = singleEps(eps[i]), ej = singleEps(eps[j]);
    let phicoul = qqrd2e * q[i] * q[j] * Math.sqrt(r2inv);
    phicoul *= 0.5 * (ei + ej);
    return { eng: factorCoul * phicoul, fforce };
  }
}

/* ----------------------------------------------------------------------
   coul/long/dielectric
   ---------------------------------------------------------------------- */

/**
 * pair_style coul/long/dielectric cutoff (the real-space Ewald term with the dielectric scaling of coul/cut/dielectric).
 * Native compute(): direct branch with the A&S polynomial erfc (pair_modify table 0 and rsq <= tabinner), table branch
 * otherwise; the force on i carries eps_i, the energy 0.5 (eps_i + eps_j), the special correction -(1 - w) prefactor
 * (not scaled by eps). The table branch uses the tabulated kernels of pair_modify table N, which include qqrd2e in native.
 */
export class PairCoulLongDielectric extends PairCoulLong {
  readonly name: string = 'coul/long/dielectric';
  needsFull = true;
  needsHalf = false;
  virialFdotr = false;
  explicitVirial = true;
  efield = new Float64Array(0);
  private readonly dieTables = new ErfcTableCache();

  compute(pc: PairCompute): void {
    const d = dielArrays(pc.s, 'pair coul/long/dielectric');
    const list = pc.full;
    if (!list) throw new StyleError('pair coul/long/dielectric needs a full neighbor list');
    const nl = pc.nlocal;
    if (this.efield.length < 3 * nl) this.efield = new Float64Array(3 * nl);
    const E = this.efield;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const scale = this.set.p('scale');
    const owner = pc.nb.owner;
    const qqrd2e = pc.qqrd2e;
    const g = this.gEwald;
    const cutsq = this.cutCoul * this.cutCoul;
    const tab = this.dieTables.get(this.table, g, cutsq, this.tabinnerRsq);
    const sC = pc.specialCoul;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = d.qs[i], ei = d.epsilon[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      selfField(d, i, E);
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const factorCoul = sC[jj >>> SBBITS];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (!(rsq < cutsq)) continue;
        const oj = owner[j];
        const qj = d.qs[oj], ej = d.epsilon[oj];
        const r2inv = 1.0 / rsq;
        // fix adapt's scale[itype][jtype] (native pair_coul_long_dielectric.cpp: in prefactor, prefactorE and the tables)
        const sc = scale[type[i] * nt + type[j]];
        let forcecoul: number, efieldI: number, ecoul: number;
        if (!tab || rsq <= tab.inner) {
          const r = Math.sqrt(rsq);
          const grij = g * r;
          const expm2 = glibcExp(-grij * grij);
          const erfc = erfcPoly(grij, expm2);
          const prefactor = qqrd2e * sc * qi * qj / r;
          forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
          if (factorCoul < 1.0) forcecoul -= (1.0 - factorCoul) * prefactor;
          const prefactorE = qqrd2e * sc * qj / r;
          efieldI = prefactorE * (erfc + EWALD_F * grij * expm2);
          if (factorCoul < 1.0) efieldI -= (1.0 - factorCoul) * prefactorE;
          ecoul = prefactor * 0.5 * (ei + ej) * erfc;
          if (factorCoul < 1.0) ecoul -= (1.0 - factorCoul) * prefactor;
        } else {
          // table branch: the tables hold qqrd2e (native ftable, etable, ctable); the field of the table branch is
          // scaled by qqrd2e as the direct branch's prefactorE
          const qq = qqrd2e * sc * qi * qj;
          forcecoul = qq * tab.force(rsq);
          efieldI = qqrd2e * sc * qj * tab.force(rsq);
          if (factorCoul < 1.0) {
            const bare = qq * tab.coul(rsq);
            forcecoul -= (1.0 - factorCoul) * bare;
            efieldI -= (1.0 - factorCoul) * qqrd2e * sc * qj * tab.coul(rsq);
          }
          ecoul = qq * 0.5 * (ei + ej) * tab.energy(rsq);
          if (factorCoul < 1.0) ecoul -= (1.0 - factorCoul) * qq * tab.coul(rsq);
        }
        const fpairI = ei * forcecoul * r2inv;
        f[3 * i] += dx * fpairI; f[3 * i + 1] += dy * fpairI; f[3 * i + 2] += dz * fpairI;
        efieldI *= ei * r2inv;
        E[3 * i] += dx * efieldI; E[3 * i + 1] += dy * efieldI; E[3 * i + 2] += dz * efieldI;
        evTallyFull(pc, i, 0, ecoul, fpairI, dx, dy, dz);
      }
    }
  }

  /** Native single_enable = 0 for this style (b332f28): compute pair/local refuses it as native does. */
  protected override singleOff = true;
  single(): never {
    throw new StyleError('pair style coul/long/dielectric: single() (pair/local, pair_write) is not supported by the browser engine');
  }
}

/* ----------------------------------------------------------------------
   lj/cut/coul/debye/dielectric
   ---------------------------------------------------------------------- */

/**
 * pair_style lj/cut/coul/debye/dielectric kappa cutoff (cutoff2). The Debye-screened Coulomb term of lj/cut/coul/debye with
 * the dielectric scaling: E = factor_coul qqrd2e q_i q_j 0.5 (eps_i + eps_j) exp(-kappa r)/r, force carrying eps_i, the
 * field and the potential of the pair stored per owned atom; the LJ term is that of lj/cut.
 */
export class PairLJCutCoulDebyeDielectric extends PairLJCutCoulDebye {
  readonly name: string = 'lj/cut/coul/debye/dielectric';
  needsFull = true;
  needsHalf = false;
  virialFdotr = false;
  explicitVirial = true;
  efield = new Float64Array(0);
  epot = new Float64Array(0);
  /** The per-atom arrays and Coulomb prefactor of the last compute() (single() reads them). */
  private dielSingle: DielArrays | null = null;
  private qqrd2eSingle = 1;

  compute(pc: PairCompute): void {
    const d = dielArrays(pc.s, 'pair lj/cut/coul/debye/dielectric');
    this.dielSingle = d;
    this.qqrd2eSingle = pc.qqrd2e;
    const list = pc.full;
    if (!list) throw new StyleError('pair lj/cut/coul/debye/dielectric needs a full neighbor list');
    const nl = pc.nlocal;
    if (this.efield.length < 3 * nl) { this.efield = new Float64Array(3 * nl); this.epot = new Float64Array(nl); }
    const E = this.efield, P = this.epot;
    const nt = this.ntypes + 1;
    const { lj1, lj2, lj3, lj4, offset } = this;
    const cutljsq = new Float64Array(nt * nt);
    const cutcoulsq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        cutljsq[i * nt + j] = this.p.get('cut', i, j) ** 2;
        // inside cutsq (the run's init) as well: native's compute tests rsq < cutsq first (src/DIELECTRIC/pair_lj_cut_coul_debye_dielectric.cpp)
        cutcoulsq[i * nt + j] = Math.min(this.cutsq[i * nt + j], this.p.get('cut_coul', i, j) ** 2);
      }
    }
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const qqrd2e = pc.qqrd2e;
    const kappa = this.kappa;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = d.qs[i], ei = d.epsilon[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      selfField(d, i, E);
      P[i] = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const factorLJ = sLJ[sb], factorCoul = sC[sb];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (!(rsq < this.cutsq[t])) continue;
        const oj = owner[j];
        const qj = d.qs[oj], ej = d.epsilon[oj];
        const r2inv = 1.0 / rsq;
        let efieldI = 0, epotI = 0, forcecoul = 0, ecoul = 0;
        let rinv = 0, screening = 0;
        const coulOn = rsq < cutcoulsq[t] && rsq > EPSILON;
        if (coulOn) {
          const r = Math.sqrt(rsq);
          rinv = 1.0 / r;
          screening = glibcExp(-kappa * r);
          efieldI = qqrd2e * qj * screening * (kappa + rinv);
          forcecoul = qi * efieldI;
          epotI = efieldI;
        }
        let forcelj = 0, evdwl = 0;
        if (rsq < cutljsq[t]) {
          const r6inv = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
          evdwl = factorLJ * (r6inv * (lj3[t] * r6inv - lj4[t]) - offset[t]);
        }
        if (coulOn) ecoul = factorCoul * qqrd2e * qi * qj * 0.5 * (ei + ej) * rinv * screening;
        const fpairI = (factorCoul * ei * forcecoul + factorLJ * forcelj) * r2inv;
        f[3 * i] += dx * fpairI; f[3 * i + 1] += dy * fpairI; f[3 * i + 2] += dz * fpairI;
        efieldI *= factorCoul * ei * r2inv;
        E[3 * i] += dx * efieldI; E[3 * i + 1] += dy * efieldI; E[3 * i + 2] += dz * efieldI;
        P[i] += epotI;
        evTallyFull(pc, i, evdwl, ecoul, fpairI, dx, dy, dz);
      }
    }
  }

  /**
   * As native LAMMPS does (src/DIELECTRIC/pair_lj_cut_coul_debye_dielectric.cpp single at b332f28): the screened Coulomb
   * force (eps_i in it, no EPSILON test) inside cut_coul, the LJ force inside cut, fforce = (factor_coul forcecoul +
   * factor_lj forcelj) / r^2, and the energy with ei = 0 for eps == 1 and the cutoffs of cut_coulsq and cut_ljsq.
   */
  single(i: number, j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, _qi = 0, _qj = 0): { eng: number; fforce: number } {
    const d = singleArrays(this.dielSingle, 'pair lj/cut/coul/debye/dielectric');
    const qqrd2e = this.qqrd2eSingle;
    const q = d.qs, eps = d.epsilon;
    const nt = this.ntypes + 1;
    const t = itype * nt + jtype;
    const cutCoulSq = this.p.get('cut_coul', itype, jtype) ** 2;
    const cutLJSq = this.p.get('cut', itype, jtype) ** 2;
    const r2inv = 1.0 / rsq;
    let forcecoul = 0, rinv = 0, screening = 0;
    if (rsq < cutCoulSq) {
      const r = Math.sqrt(rsq);
      rinv = 1.0 / r;
      screening = glibcExp(-this.kappa * r);
      forcecoul = qqrd2e * q[i] * q[j] * screening * (this.kappa + rinv) * eps[i];
    }
    let forcelj = 0, r6inv = 0;
    if (rsq < cutLJSq) {
      r6inv = r2inv * r2inv * r2inv;
      forcelj = r6inv * (this.lj1[t] * r6inv - this.lj2[t]);
    }
    const fforce = (factorCoul * forcecoul + factorLJ * forcelj) * r2inv;
    const ei = singleEps(eps[i]), ej = singleEps(eps[j]);
    let eng = 0;
    if (rsq < cutCoulSq) {
      let phicoul = qqrd2e * q[i] * q[j] * rinv * screening;
      phicoul *= 0.5 * (ei + ej);
      eng += factorCoul * phicoul;
    }
    if (rsq < cutLJSq) {
      const philj = r6inv * (this.lj3[t] * r6inv - this.lj4[t]) - this.offset[t];
      eng += factorLJ * philj;
    }
    return { eng, fforce };
  }
}
