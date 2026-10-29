/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/pair_lj_cut_coul_msm_dielectric.cpp and pair_lj_cut_coul_msm_dielectric.h at commit b332f28
 * (Trung Nguyen and Monica Olvera de la Cruz, Northwestern University, and the DIELECTRIC package authors), with the
 * MSM kernel and tables of ../msm/pair_coul_msm.ts. Copyright Sandia Corporation (GNU General Public License). This file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style lj/cut/coul/msm/dielectric: the MSM short-range Coulomb term with the scaled charges of atom style
 * dielectric, on a full neighbor list. The Coulomb force on i carries epsilon_i, the energy of a pair is
 * qq 0.5 (eps_i + eps_j) (1 - (r/a) gamma(r/a)) / r and its special-bond correction keeps the bare form. The field at
 * each owned atom (efield, with the curvature self term of Barros et al.) is kept for the polarization fixes. The
 * virial is tallied pair by pair (Pair::ev_tally_full; native sets no_virial_fdotr_compute): with pressure/scalar yes
 * the tally carries the Coulomb part of the force, the LJ forces go to f at once and the Coulomb ones after the loop,
 * and E_coul/3 is added to the diagonal, as natively.
 */
import { StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { msmGamma, msmDgamma } from '../msm/tables';
import { PairLJCutCoulMSM, msmOf } from '../msm/pair_coul_msm';
import { dielArrays } from './common';

/** pair_lj_cut_coul_msm_dielectric.cpp EPSILON: within this distance squared the Coulomb term is left out. */
const EPSILON = 1.0e-6;
/** MathConst::MY_PIS (sqrt(pi) rounded once). */
const MY_PIS = 1.77245385090551602729;

const f32 = new Float32Array(1);
const i32 = new Int32Array(f32.buffer);

export class PairLJCutCoulMSMDielectric extends PairLJCutCoulMSM {
  override readonly name: string = 'lj/cut/coul/msm/dielectric';
  override needsFull = true;
  override needsHalf = false;
  override virialFdotr = false;
  /** Electric field at each owned atom from this pair (3 per atom), read by the polarization fixes. */
  efield = new Float64Array(0);
  private ftmp = new Float64Array(0);

  override compute(pc: PairCompute): void {
    const msm = msmOf(this.kspace, this.name);
    const scalar = msm.scalarPressure;
    if (scalar && pc.vatom) throw new StyleError("Must use 'kspace_modify pressure/scalar no' to obtain per-atom virial with kspace_style MSM");
    const d = dielArrays(pc.s, 'Pair lj/cut/coul/msm/dielectric');
    const list = pc.full;
    if (!list) throw new StyleError('pair lj/cut/coul/msm/dielectric needs a full neighbor list');
    const nl = pc.nlocal;
    if (this.efield.length < 3 * nl) this.efield = new Float64Array(3 * nl);
    if (scalar) {
      if (this.ftmp.length < 3 * pc.nall) this.ftmp = new Float64Array(3 * pc.nall);
      else this.ftmp.fill(0, 0, 3 * pc.nall);
    }
    const ft = this.ftmp;
    const nt = this.ntypes + 1;
    const { lj1, lj2, lj3, lj4, offset, cutsq, cutljsq } = this;
    // cut_coulsq and the tables of the run's init, the gamma argument r / cut_coul as cut_coul is now (fix adapt)
    const cutCoul = this.cutCoul, cutCoulsq = this.cutCoulInit * this.cutCoulInit;
    const order = msm.order, qqrd2e = pc.qqrd2e;
    const tab = this.tables.get(this.table, this.tabinnerRsq, this.cutCoulInit, qqrd2e, order);
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const q = d.qs, eps = d.epsilon, area = d.area, curvature = d.curvature, norm = d.mu;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const E = this.efield;
    const acc = pc.acc, eatom = pc.eatom, vatom = pc.vatom;
    let engCoul = 0.0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qtmp = q[i], etmp = eps[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const ti = type[i] * nt;
      // self term Eq. (55) for I_{ii} and Eq. (52) and in Barros et al
      const curvatureThreshold = Math.sqrt(area[i]);
      if (curvature[i] < curvatureThreshold) {
        const sf = (curvature[i] / (4.0 * MY_PIS * curvatureThreshold)) * area[i] * q[i];
        E[3 * i] = sf * norm[4 * i]; E[3 * i + 1] = sf * norm[4 * i + 1]; E[3 * i + 2] = sf * norm[4 * i + 2];
      } else {
        E[3 * i] = E[3 * i + 1] = E[3 * i + 2] = 0;
      }
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const factorCoul = sC[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const oj = owner[j];
        const qj = q[oj], epsj = eps[oj];
        const r2inv = 1.0 / rsq;
        let forcecoul = 0.0, efieldI = 0.0, prefactor = 0.0, egamma = 0.0;
        let direct = true, fraction = 0.0, itable = 0;
        const coul = rsq < cutCoulsq && rsq > EPSILON;
        if (coul) {
          if (!tab || rsq <= tab.tabinnersq) {
            const r = Math.sqrt(rsq);
            prefactor = (qqrd2e * qtmp * qj) / r;
            egamma = 1.0 - (r / cutCoul) * msmGamma(order, r / cutCoul);
            const fgamma = 1.0 + (rsq / cutCoulsq) * msmDgamma(order, r / cutCoul);
            forcecoul = prefactor * fgamma;
            if (factorCoul < 1.0) forcecoul -= (1.0 - factorCoul) * prefactor;
            const prefactorE = (qqrd2e * qj) / r;
            efieldI = prefactorE * fgamma;
            if (factorCoul < 1.0) efieldI -= (1.0 - factorCoul) * prefactorE;
          } else {
            direct = false;
            f32[0] = rsq;
            itable = (i32[0] & tab.ncoulmask) >> tab.ncoulshiftbits;
            fraction = (f32[0] - tab.rtable[itable]) * tab.drtable[itable];
            let table = tab.ftable[itable] + fraction * tab.dftable[itable];
            forcecoul = qtmp * qj * table;
            efieldI = qj * table;
            if (factorCoul < 1.0) {
              table = tab.ctable[itable] + fraction * tab.dctable[itable];
              prefactor = qtmp * qj * table;
              forcecoul -= (1.0 - factorCoul) * prefactor;
              const prefactorE = qj * table;
              efieldI -= (1.0 - factorCoul) * prefactorE;
            }
          }
        }
        let forcelj = 0.0, r6inv = 0.0;
        const lj = rsq < cutljsq[t];
        if (lj) {
          r6inv = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
        }
        let fpairI: number;
        if (!scalar) {
          fpairI = (forcecoul * etmp + factorLJ * forcelj) * r2inv;
          f[3 * i] += delx * fpairI; f[3 * i + 1] += dely * fpairI; f[3 * i + 2] += delz * fpairI;
        } else {
          // separate LJ and Coulombic forces
          const fpair = factorLJ * forcelj * r2inv;
          f[3 * i] += delx * fpair; f[3 * i + 1] += dely * fpair; f[3 * i + 2] += delz * fpair;
          fpairI = forcecoul * etmp * r2inv;
          ft[3 * i] += delx * fpairI; ft[3 * i + 1] += dely * fpairI; ft[3 * i + 2] += delz * fpairI;
        }
        efieldI *= etmp * r2inv;
        E[3 * i] += delx * efieldI; E[3 * i + 1] += dely * efieldI; E[3 * i + 2] += delz * efieldI;
        let ecoul = 0.0;
        if (coul) {
          if (direct) ecoul = prefactor * 0.5 * (etmp + epsj) * egamma;
          else {
            const table = tab!.etable[itable] + fraction * tab!.detable[itable];
            ecoul = qtmp * qj * 0.5 * (etmp + epsj) * table;
          }
          if (factorCoul < 1.0) ecoul -= (1.0 - factorCoul) * prefactor;
        }
        let evdwl = 0.0;
        if (lj) evdwl = factorLJ * (r6inv * (lj3[t] * r6inv - lj4[t]) - offset[t]);
        // Pair::ev_tally_full
        acc.evdwl += 0.5 * evdwl;
        acc.ecoul += 0.5 * ecoul;
        engCoul += 0.5 * ecoul;
        if (eatom) eatom[i] += 0.5 * (evdwl + ecoul);
        const v0 = 0.5 * delx * delx * fpairI, v1 = 0.5 * dely * dely * fpairI, v2 = 0.5 * delz * delz * fpairI;
        const v3 = 0.5 * delx * dely * fpairI, v4 = 0.5 * delx * delz * fpairI, v5 = 0.5 * dely * delz * fpairI;
        const vir = acc.virial;
        vir[0] += v0; vir[1] += v1; vir[2] += v2; vir[3] += v3; vir[4] += v4; vir[5] += v5;
        if (vatom) {
          vatom[6 * i] += v0; vatom[6 * i + 1] += v1; vatom[6 * i + 2] += v2;
          vatom[6 * i + 3] += v3; vatom[6 * i + 4] += v4; vatom[6 * i + 5] += v5;
        }
      }
    }
    if (scalar) {
      // "virial[i] += force->pair->eng_coul / 3.0", then the Coulomb forces join f
      for (let c = 0; c < 3; c++) acc.virial[c] += engCoul / 3.0;
      for (let k = 0; k < 3 * pc.nall; k++) f[k] += ft[k];
    }
  }

  /** Native has a single() for this style that this engine does not have yet (the parent's would be wrong): a named gap. */
  protected override singleMissing = true;
  override single(): { eng: number; fforce: number } {
    throw new StyleError('pair lj/cut/coul/msm/dielectric: single() (pair/local, pair_write) is not supported by the browser engine');
  }
}
