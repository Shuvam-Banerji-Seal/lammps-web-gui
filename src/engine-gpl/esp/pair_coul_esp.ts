/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/pair_coul_esp.cpp and
 * pair_coul_esp.h at b332f28 (PairCoulEsp::settings, coeff, init_style, init_one, compute, single, extract). The
 * real-space kernels are the polynomials of the kspace style esp (kspace_esp.ts: force_poly_coeff and
 * energy_poly_coeff), read from the kspace object at each force evaluation as init_style reads them once.
 * Copyright of the LAMMPS sources: Sandia Corporation, GPL-2.0. This file is GPL-2.0 too (see src/engine-gpl/LICENSE).
 *
 * Both the polynomial path (pair_modify table 0) and the native lookup table (default pair_modify table 12, built by
 * espTables.ts as Pair::init_tables does with espflag) are ported.
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { PairCoulLong } from '../../engine/force/pair/coul_long';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum } from '../../engine/force/util';
import type { KSpaceESP } from './kspace_esp';
import { EspTableCache, floatBits, bitsToFloat } from './espTables';

/** The real-space force and energy kernels of esp at distance r (pair_coul_esp.cpp compute, polynomial branch). */
export const espPolyValues = (r: number, cut: number, fpoly: number[], epoly: number[]):
  { force: number; energy: number } => {
  let forceAppx = fpoly[0];
  let forceR = 1.0;
  const rScal = 2.0 * r / cut - 1.0;
  for (let index = 1; index < fpoly.length; index++) {
    forceR *= rScal;
    forceAppx += fpoly[index] * forceR;
  }
  let energyAppx = epoly[0];
  let energyR = 1.0;
  for (let index = 1; index < epoly.length; index++) {
    energyR *= rScal;
    energyAppx = energyAppx + epoly[index] * energyR;
  }
  return { force: forceAppx, energy: energyAppx };
};

export class PairCoulEsp extends PairCoulLong {
  readonly name: string = 'coul/esp';
  /** The kspace style esp that supplies the splitting polynomials (linked by KSpaceESP.linkPair). */
  kspace: KSpaceESP | null = null;

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutCoul = parseNum(args[0], 'cutoff');
    if (!(this.cutCoul > 0)) throw new StyleError('coul/esp cutoff must be > 0');
  }

  // The kspace style is linked and initialised after the pair style's init (forcefield.ts), so the kspace
  // polynomials are read at each force evaluation (compute), not here (native reads them in init_style).

  private readonly espTable = new EspTableCache();

  /**
   * cut_coul at the run's init. As native LAMMPS does (src/KSPACE/pair_coul_esp.cpp at b332f28): init_style sets cut_coulsq and the
   * tables from cut_coul; the force loop's r_scal = 2 r / cut_coul - 1 reads cut_coul as it is then (fix adapt).
   */
  private cutCoulInit = 0;

  initStyle(ctx: StyleContext): void {
    super.initStyle(ctx);
    this.cutCoulInit = this.cutCoul;
  }

  /**
   * PairCoulEsp::single: the polynomial or table term with the special weight, no cutoff test and no fix adapt scale;
   * r_scal reads cut_coul as it is now, the table is the one of the run's init.
   */
  single(_i: number, _j: number, _it: number, _jt: number, rsq: number, factorCoul: number, _fl: number, qi = 0, qj = 0): { eng: number; fforce: number } {
    const ks = this.kspace;
    if (!ks) throw new StyleError('Pair style requires a KSpace style');
    const qqrd2e = this.qqrd2eLast;
    const fpoly = ks.forcePolyCoeff, epoly = ks.energyPolyCoeff;
    const tab = this.table !== 0 ? this.espTable.get(this.cutCoulInit, this.tabinnerRsq, this.table, qqrd2e, fpoly, epoly) : null;
    const r2inv = 1.0 / rsq;
    let forcecoul: number, phicoul: number, prefactor = 0.0;
    if (!tab || rsq <= tab.tabinnersq) {
      const r = Math.sqrt(rsq);
      const vals = espPolyValues(r, this.cutCoul, fpoly, epoly);
      prefactor = qqrd2e * qi * qj / r;
      forcecoul = prefactor * vals.force;
      if (factorCoul < 1.0) forcecoul -= (1.0 - factorCoul) * prefactor;
      phicoul = prefactor * vals.energy;
    } else {
      const F = bitsToFloat(floatBits(rsq));
      const itable = (floatBits(rsq) & tab.ncoulmask) >>> tab.ncoulshiftbits;
      const fraction = (F - tab.rtable[itable]) * tab.drtable[itable];
      let table = tab.ftable[itable] + fraction * tab.dftable[itable];
      forcecoul = qi * qj * table;
      if (factorCoul < 1.0) {
        table = tab.ctable[itable] + fraction * tab.dctable[itable];
        prefactor = qi * qj * table;
        forcecoul -= (1.0 - factorCoul) * prefactor;
      }
      table = tab.etable[itable] + fraction * tab.detable[itable];
      phicoul = qi * qj * table;
    }
    if (factorCoul < 1.0) phicoul -= (1.0 - factorCoul) * prefactor;
    return { eng: phicoul, fforce: forcecoul * r2inv };
  }

  compute(pc: PairCompute): void {
    const ks = this.kspace;
    if (!ks) throw new StyleError('Pair style requires a KSpace style');
    this.qqrd2eLast = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, q, type } = pc;
    const nt = this.ntypes + 1;
    const scale = this.set.p('scale');
    const cutCoul = this.cutCoul;
    const cutCoulsq = this.cutCoulInit * this.cutCoulInit;
    const sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const qqrd2e = pc.qqrd2e;
    const fpoly = ks.forcePolyCoeff, epoly = ks.energyPolyCoeff;
    const tab = this.table !== 0 ? this.espTable.get(this.cutCoulInit, this.tabinnerRsq, this.table, qqrd2e, fpoly, epoly) : null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qtmp = q[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const factorCoul = sC[jj >>> SBBITS];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutCoulsq) {
          const r2inv = 1.0 / rsq;
          let forcecoul: number, e: number;
          if (!tab || rsq <= tab.tabinnersq) {
            const r = Math.sqrt(rsq);
            const vals = espPolyValues(r, cutCoul, fpoly, epoly);
            const prefactor = qqrd2e * qtmp * q[j] / r;
            forcecoul = prefactor * vals.force;
            if (factorCoul < 1.0) forcecoul -= (1.0 - factorCoul) * prefactor;
            e = prefactor * vals.energy;
            if (factorCoul < 1.0) e -= (1.0 - factorCoul) * prefactor;
          } else {
            // native table branch: rsq is stored as a float and its bits index the bins
            const F = bitsToFloat(floatBits(rsq));
            const itable = (floatBits(rsq) & tab.ncoulmask) >>> tab.ncoulshiftbits;
            const fraction = (F - tab.rtable[itable]) * tab.drtable[itable];
            let table = tab.ftable[itable] + fraction * tab.dftable[itable];
            forcecoul = qtmp * q[j] * table;
            let prefactor = 0.0;
            if (factorCoul < 1.0) {
              table = tab.ctable[itable] + fraction * tab.dctable[itable];
              prefactor = qtmp * q[j] * table;
              forcecoul -= (1.0 - factorCoul) * prefactor;
            }
            table = tab.etable[itable] + fraction * tab.detable[itable];
            // native: only this energy carries fix adapt's scale[itype][jtype] (pair_coul_esp.cpp compute)
            e = scale[type[i] * nt + type[j]] * qtmp * q[j] * table;
            if (factorCoul < 1.0) e -= (1.0 - factorCoul) * prefactor;
          }
          const fpair = forcecoul * r2inv;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          ecoul += e;
          if (tally) tallyAtom(pc, i, j, e, fpair, delx, dely, delz);
        }
      }
    }
    pc.acc.ecoul += ecoul;
  }

  extract(name: string): unknown {
    return name === 'cut_coul' ? this.cutCoul : undefined;
  }
}
