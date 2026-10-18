/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/pair_lj_cut_coul_esp.cpp
 * and pair_lj_cut_coul_esp.h at b332f28 (PairLJCutCoulEsp::settings, init_one, compute, extract). The Lennard-Jones
 * part is the lj/cut term of the engine (PairLJCut); the Coulomb real-space kernels are the esp polynomials
 * (kspace_esp.ts), read from the kspace object at each force evaluation.
 * Copyright of the LAMMPS sources: Sandia Corporation, GPL-2.0. This file is GPL-2.0 too (see src/engine-gpl/LICENSE).
 *
 * Both the polynomial path (pair_modify table 0) and the native lookup table (default table 12, espTables.ts).
 */
import { StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { PairLJCutCoulLong } from '../../engine/force/pair/coul_long';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import type { KSpaceESP } from './kspace_esp';
import { espPolyValues } from './pair_coul_esp';
import { EspTableCache, floatBits, bitsToFloat } from './espTables';

export class PairLJCutCoulEsp extends PairLJCutCoulLong {
  readonly name: string = 'lj/cut/coul/esp';
  /** The kspace style esp that supplies the splitting polynomials (linked by KSpaceESP.linkPair). */
  kspace: KSpaceESP | null = null;

  private readonly espTable = new EspTableCache();

  compute(pc: PairCompute): void {
    const ks = this.kspace;
    if (!ks) throw new StyleError('Pair style requires a KSpace style');
    const list = pc.half!;
    const { x, f, q, type } = pc;
    const nt = this.ntypes + 1;
    const { lj1, lj2, lj3, lj4, offset } = this;
    // cut_ljsq and cutsq of init_one: cut = MAX(cut_lj, cut_coul + 2 qdist) with qdist = 0
    const cutljsq = new Float64Array(nt * nt);
    const cutsq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const cl = this.p.get('cut', i, j);
        cutljsq[i * nt + j] = cl * cl;
        const c = Math.max(cl, this.cutCoul + 0.0);
        cutsq[i * nt + j] = c * c;
      }
    }
    const cutCoul = this.cutCoul;
    const cutCoulsq = cutCoul * cutCoul;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const qqrd2e = pc.qqrd2e;
    const fpoly = ks.forcePolyCoeff, epoly = ks.energyPolyCoeff;
    const tab = this.table !== 0 ? this.espTable.get(cutCoul, this.tabinnerRsq, this.table, qqrd2e, fpoly, epoly) : null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qtmp = q[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const factorLJ = sLJ[sb];
        const factorCoul = sC[sb];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = ti + type[j];
        if (rsq < cutsq[t]) {
          const r2inv = 1.0 / rsq;
          let forcecoul = 0.0;
          let prefactor = 0.0;
          let r = 0.0;
          const usePoly = !tab || rsq <= tab.tabinnersq;
          let itable = 0, fraction = 0;
          if (rsq < cutCoulsq) {
            if (usePoly) {
              r = Math.sqrt(rsq);
              const vals = espPolyValues(r, cutCoul, fpoly, epoly);
              prefactor = qqrd2e * qtmp * q[j] / r;
              forcecoul = prefactor * vals.force;
              if (factorCoul < 1.0) forcecoul -= (1.0 - factorCoul) * prefactor;
            } else {
              itable = (floatBits(rsq) & tab!.ncoulmask) >>> tab!.ncoulshiftbits;
              fraction = (bitsToFloat(floatBits(rsq)) - tab!.rtable[itable]) * tab!.drtable[itable];
              let table = tab!.ftable[itable] + fraction * tab!.dftable[itable];
              forcecoul = qtmp * q[j] * table;
              if (factorCoul < 1.0) {
                table = tab!.ctable[itable] + fraction * tab!.dctable[itable];
                prefactor = qtmp * q[j] * table;
                forcecoul -= (1.0 - factorCoul) * prefactor;
              }
            }
          }
          let forcelj = 0.0;
          let r6inv = 0.0;
          if (rsq < cutljsq[t]) {
            r6inv = r2inv * r2inv * r2inv;
            forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
          }
          const fpair = (forcecoul + factorLJ * forcelj) * r2inv;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          let e = 0.0, eLJ = 0.0;
          if (rsq < cutCoulsq) {
            if (usePoly) {
              const vals = espPolyValues(r, cutCoul, fpoly, epoly);
              e = prefactor * vals.energy;
            } else {
              const table = tab!.etable[itable] + fraction * tab!.detable[itable];
              e = qtmp * q[j] * table;
            }
            if (factorCoul < 1.0) e -= (1.0 - factorCoul) * prefactor;
          }
          if (rsq < cutljsq[t]) {
            eLJ = r6inv * (lj3[t] * r6inv - lj4[t]) - offset[t];
            eLJ *= factorLJ;
          }
          evdwl += eLJ;
          ecoul += e;
          if (tally) tallyAtom(pc, i, j, eLJ + e, fpair, delx, dely, delz);
        }
      }
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }
}
