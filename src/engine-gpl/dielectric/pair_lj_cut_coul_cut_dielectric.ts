/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/pair_lj_cut_coul_cut_dielectric.cpp and pair_lj_cut_coul_cut_dielectric.h at commit b332f28
 * (Trung Nguyen and Monica Olvera de la Cruz, Northwestern University, and the DIELECTRIC package authors). Copyright Sandia Corporation (GNU
 * General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style lj/cut/coul/cut/dielectric: the lj/cut/coul/cut pair with the Coulomb charges scaled by the per-atom
 * dielectric constant (atom_style dielectric). Full neighbor list, the force on owned atom i only (LAMMPS
 * ev_tally_full halves each pair's energy and virial). It also stores the electric field at each owned atom (efield),
 * which the polarization fixes read. Documented on docs.lammps.org/pair_lj_cut_coul.html and dielectric.html.
 * single() (compute pair/local) is the native single() of pair_lj_cut_coul_cut_dielectric.cpp at b332f28.
 */
import { PairLJCutCoulCut } from '../../engine/force/pair/lj_coul';
import { evTallyFull, StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { dielArrays, type DielArrays } from './common';
import { singleArrays, singleEps } from '../csdiel/dielectric';

/** Within this distance squared the Coulomb term is left out (pair_coul_cut_dielectric.cpp EPSILON). */
const EPSILON = 1.0e-6;
/** sqrt(pi), as LAMMPS MathConst::MY_PIS (the self term of the field uses it). */
const MY_PIS = 1.77245385090551602729; // math_const.h (sqrt(pi) rounded once; Math.sqrt(Math.PI) is one ulp below)

export class PairLJCutCoulCutDielectric extends PairLJCutCoulCut {
  readonly name: string = 'lj/cut/coul/cut/dielectric';
  /** native sets no_virial_fdotr_compute: the virial is the sum of the ev_tally_full tallies (forces on i only) */
  override virialFdotr = false;
  needsFull = true;
  needsHalf = false;
  /** Electric field at each owned atom from this pair (3 per atom), set by compute(). */
  efield = new Float64Array(0);
  /** Electric potential at each owned atom from this pair (pair_lj_cut_coul_cut_dielectric.cpp epot). */
  epot = new Float64Array(0);
  /** The per-atom arrays and Coulomb prefactor of the last compute() (single() reads them). */
  private dielSingle: DielArrays | null = null;
  private qqrd2eSingle = 1;

  compute(pc: PairCompute): void {
    const d = dielArrays(pc.s, 'pair lj/cut/coul/cut/dielectric');
    this.dielSingle = d;
    this.qqrd2eSingle = pc.qqrd2e;
    const list = pc.full;
    if (!list) throw new StyleError('pair lj/cut/coul/cut/dielectric needs a full neighbor list');
    const nl = pc.nlocal;
    if (this.efield.length < 3 * nl) {
      this.efield = new Float64Array(3 * nl);
      this.epot = new Float64Array(nl);
    }
    const nt = this.ntypes + 1;
    const cutljsq = new Float64Array(nt * nt);
    const cutcoulsq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        cutljsq[i * nt + j] = this.p.get('cut', i, j) ** 2;
        // inside cutsq (the run's init) as well: native's compute tests rsq < cutsq first (src/DIELECTRIC/pair_lj_cut_coul_cut_dielectric.cpp)
        cutcoulsq[i * nt + j] = Math.min(this.cutsq[i * nt + j], this.p.get('cut_coul', i, j) ** 2);
      }
    }
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
      // self term of the field (LAMMPS: curvature below sqrt(area) contributes sf times the normal)
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
        if (rsq >= cutljsq[t] && rsq >= cutcoulsq[t]) continue;
        // owned or ghost j: its scaled charge and dielectric constant are those of its owner
        const oj = owner[j];
        const qj = qs[oj], ej = eps[oj];
        const r2inv = 1.0 / rsq;
        const rinv = Math.sqrt(r2inv);
        let efieldI: number, forcecoul: number, epotI: number;
        if (rsq < cutcoulsq[t] && rsq > EPSILON) {
          efieldI = qqrd2e * qj * rinv;
          forcecoul = qi * efieldI;
          epotI = efieldI;
        } else {
          efieldI = forcecoul = epotI = 0;
        }
        let r6inv = 0, forcelj = 0;
        if (rsq < cutljsq[t]) {
          r6inv = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
        }
        const fpairI = (factorCoul * ei * forcecoul + factorLJ * forcelj) * r2inv;
        fxi += dx * fpairI; fyi += dy * fpairI; fzi += dz * fpairI;
        efieldI *= factorCoul * ei * r2inv;
        E[3 * i] += dx * efieldI; E[3 * i + 1] += dy * efieldI; E[3 * i + 2] += dz * efieldI;
        P[i] += epotI;
        let ecoul = 0, evdwl = 0;
        if (rsq < cutcoulsq[t]) ecoul = factorCoul * qqrd2e * qi * qj * 0.5 * (ei + ej) * rinv;
        if (rsq < cutljsq[t]) evdwl = factorLJ * (r6inv * (lj3[t] * r6inv - lj4[t]) - offset[t]);
        // Pair::ev_tally_full: half of the energy and of the virial of each ordered pair, to atom i
        evTallyFull(pc, i, evdwl, ecoul, fpairI, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
  }

  /**
   * As native LAMMPS does (src/DIELECTRIC/pair_lj_cut_coul_cut_dielectric.cpp single at b332f28): the Coulomb force inside
   * cut_coul (with eps_i, no EPSILON test), the LJ force inside cut, fforce = (factor_coul forcecoul + factor_lj forcelj) / r^2,
   * and the energy with ei = 0 for eps == 1. The arrays are those of the last compute (q_scaled and epsilon).
   */
  single(i: number, j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, _qi = 0, _qj = 0): { eng: number; fforce: number } {
    const d = singleArrays(this.dielSingle, 'pair lj/cut/coul/cut/dielectric');
    const qqrd2e = this.qqrd2eSingle;
    const q = d.qs, eps = d.epsilon;
    const nt = this.ntypes + 1;
    const t = itype * nt + jtype;
    const cutCoulSq = this.p.get('cut_coul', itype, jtype) ** 2;
    const cutLJSq = this.p.get('cut', itype, jtype) ** 2;
    const r2inv = 1.0 / rsq;
    const forcecoul = rsq < cutCoulSq ? qqrd2e * q[i] * q[j] * Math.sqrt(r2inv) * eps[i] : 0;
    let forcelj = 0, r6inv = 0;
    if (rsq < cutLJSq) {
      r6inv = r2inv * r2inv * r2inv;
      forcelj = r6inv * (this.lj1[t] * r6inv - this.lj2[t]);
    }
    const fforce = (factorCoul * forcecoul + factorLJ * forcelj) * r2inv;
    const ei = singleEps(eps[i]), ej = singleEps(eps[j]);
    let eng = 0;
    if (rsq < cutCoulSq) {
      let phicoul = qqrd2e * q[i] * q[j] * Math.sqrt(r2inv);
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
