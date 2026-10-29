/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/YAFF/pair_mm3_switch3_coulgauss_long.cpp and pair_mm3_switch3_coulgauss_long.h at commit b332f28
 * (contributing author in the C++ file: Steven Vandenbrande); copyright (2003) Sandia Corporation (GNU GPL).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { Pair, PairParams, StyleError, mixDistance, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { erfcPoly, EWALD_F, ErfcTableCache, type ErfcTable } from '../../engine/force/erfc';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
import { erfc as libmErfc } from '../libm/erf';
import { tallyAtom } from '../../engine/force/pair/lj_cut';

/**
 * Pair style mm3/switch3/coulgauss/long (pair_mm3_switch3_coulgauss_long.cpp): the MM3 van der Waals term with a
 * switched-off cutoff (truncw), the Gaussian-smeared Coulomb correction of the bond-charge term (lj2 = 1 / sqrt(gamma_i^2
 * + gamma_j^2)) and the real-space part of the Ewald / PPPM Coulomb sum, through pair_modify table N as in coul/long.
 * Only the pair style is ported: the ylz-like torque and the lj/switch3/coulgauss/long variant are not.
 */
export class PairMM3Switch3CoulGaussLong extends Pair {
  readonly name = 'mm3/switch3/coulgauss/long';
  coulLong = true;
  virialFdotr = true;
  private p = new PairParams(0, ['epsilon', 'sigma', 'gamma', 'cut_lj']);
  private cutLJGlobal = 0;
  private cutCoul = 0;
  private cutCoulsq = 0;
  private truncw = 0;
  private truncwi = 0;
  private allocatedTypes = false;
  private lj1 = new Float64Array(0);
  private lj2 = new Float64Array(0);
  private lj3 = new Float64Array(0);
  private lj4 = new Float64Array(0);
  private offset = new Float64Array(0);
  private cutLJ = new Float64Array(0);
  private cutLJsq = new Float64Array(0);
  private readonly erfcTables = new ErfcTableCache();

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'gamma', 'cut_lj']);
    this.allocatedTypes = true;
  }

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length < 2 || args.length > 3) throw new StyleError('Illegal pair_style mm3/switch3/coulgauss/long command');
    this.cutLJGlobal = parseNum(args[0], 'cutoff');
    if (args.length === 2) {
      this.cutCoul = this.cutLJGlobal;
      this.truncw = parseNum(args[1], 'truncw');
    } else {
      this.cutCoul = parseNum(args[1], 'coulomb cutoff');
      this.truncw = parseNum(args[2], 'truncw');
    }
  }

  override coeff(args: string[], _ctx: StyleContext): void {
    if (!this.allocatedTypes) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 5 || args.length > 6) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const gam = parseNum(args[4], 'gamma');
    const cutLJ = args.length === 6 ? parseNum(args[5], 'cutoff') : this.cutLJGlobal;
    const count = this.p.setRange(args[0], args[1], [eps, sig, gam, cutLJ]);
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  override initStyle(ctx: StyleContext): void {
    if (!ctx.s) throw new StyleError('pair_style mm3/switch3/coulgauss/long needs a box');
    this.cutCoulsq = this.cutCoul * this.cutCoul;
    this.truncwi = this.truncw > 0 ? 1.0 / this.truncw : 0;
    const nt = this.ntypes + 1;
    this.lj1 = new Float64Array(nt * nt);
    this.lj2 = new Float64Array(nt * nt);
    this.lj3 = new Float64Array(nt * nt);
    this.lj4 = new Float64Array(nt * nt);
    this.offset = new Float64Array(nt * nt);
    this.cutLJ = new Float64Array(nt * nt);
    this.cutLJsq = new Float64Array(nt * nt);
  }

  override initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) {
      if (!this.p.isSet(i, i) || !this.p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair mm3/switch3/coulgauss/long, types ${i} ${j})`);
      const ei = this.p.get('epsilon', i, i), ej = this.p.get('epsilon', j, j);
      const si = this.p.get('sigma', i, i), sj = this.p.get('sigma', j, j);
      const gi = this.p.get('gamma', i, i), gj = this.p.get('gamma', j, j);
      const ci = this.p.get('cut_lj', i, i), cj = this.p.get('cut_lj', j, j);
      // pair_mm3_switch3_coulgauss_long.cpp init_one: epsilon = sqrt(e_i e_j), sigma = (s_i + s_j) / 2,
      // gamma = 1 / sqrt(g_i^2 + g_j^2), cutoff by the mix rule
      this.p.setMixed(i, j, 'epsilon', Math.sqrt(ei * ej));
      this.p.setMixed(i, j, 'sigma', 0.5 * (si + sj));
      this.p.setMixed(i, j, 'gamma', 1.0 / Math.sqrt(gi * gi + gj * gj));
      this.p.setMixed(i, j, 'cut_lj', mixDistance(this.mix, ci, cj));
    }
    const cutLJ = this.p.get('cut_lj', i, j);
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
    // qdist = 0 (no TIP4P site): the pair cutoff is the larger of the LJ and Coulomb cutoffs
    const cut = Math.max(cutLJ, this.cutCoul);
    const nt = this.ntypes + 1;
    const q = i * nt + j, qr = j * nt + i;
    const cutLJsq = cutLJ * cutLJ;
    const lj1 = 12.0 / (2.0 * sig);
    const gi = this.p.get('gamma', i, i), gj = this.p.get('gamma', j, j);
    const lj2 = gi === 0 && gj === 0 ? 0 : 1.0 / Math.sqrt(gi * gi + gj * gj);
    const lj3 = 1.84e5 * eps;
    const lj4 = 2.25 * eps * libmPow(2.0 * sig, 6.0);
    let offset = 0;
    if (this.shift && cutLJ > 0.0 && this.truncw === 0.0) {
      // the offset makes the energy zero at the cutoff, as in init_one (truncw == 0)
      const r = cutLJ;
      const r2inv = 1.0 / (r * r);
      const r6inv = r2inv * r2inv * r2inv;
      const expb = lj3 * libmExp(-lj1 * r);
      offset = expb - lj4 * r6inv;
    }
    for (const k of [q, qr]) {
      this.cutLJ[k] = cutLJ;
      this.cutLJsq[k] = cutLJsq;
      this.lj1[k] = lj1;
      this.lj2[k] = lj2;
      this.lj3[k] = lj3;
      this.lj4[k] = lj4;
      this.offset[k] = offset;
    }
    return cut;
  }

  /** The Coulomb cutoff the kspace style reads (pair_mm3_switch3_coulgauss_long.cpp extract "cut_coul"). */
  override extract(name: string): unknown {
    // pair_mm3_switch3_coulgauss_long.cpp extract: "epsilon", "sigma", "gamma" (dim 2) besides "cut_coul"
    if (name === 'epsilon' || name === 'sigma' || name === 'gamma') return this.p.p(name);
    return name === 'cut_coul' ? this.cutCoul : undefined;
  }

  override compute(pc: PairCompute): void {
    if (!pc.q) throw new StyleError('Pair style mm3/switch3/coulgauss/long requires atom attribute q');
    const list = pc.half!;
    const { x, f, q, type } = pc;
    const nt = this.ntypes + 1;
    const g = this.gEwald;
    const tab: ErfcTable | null = this.erfcTables.get(this.table, g, this.cutCoul * this.cutCoul, this.tabinnerRsq);
    const qqrd2e = pc.qqrd2e;
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const cutsq = this.cutsq;
    const cutCoulsq = this.cutCoulsq;
    const truncw = this.truncw, truncwi = this.truncwi;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwlSum = 0, ecoulSum = 0;
    for (let ii = 0; ii < list.inum; ii++) {
      const i = ii;
      const qtmp = q[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = list.neighbors[k];
        const j = jj & NEIGHMASK;
        const factorLJ = sL[jj >>> SBBITS];
        const factorCoul = sC[jj >>> SBBITS];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jtype = type[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        let forcecoul = 0, forcecoul2 = 0, forcelj = 0, ecoul = 0, evdwl = 0;
        const r2inv = 1.0 / rsq;
        if (rsq < cutCoulsq) {
          if (!tab || rsq <= this.tabinnerRsq) {
            // direct branch: the polynomial erfc of coul/long (EWALD_P, A1..A5 in ewald_const.h)
            const r = Math.sqrt(rsq);
            const grij = g * r;
            const expm2 = libmExp(-grij * grij);
            const erfc1 = erfcPoly(grij, expm2);
            const prefactor = qqrd2e * qtmp * q[j] / r;
            forcecoul = prefactor * (erfc1 + EWALD_F * grij * expm2);
            ecoul = prefactor * erfc1;
            if (factorCoul < 1.0) {
              forcecoul -= (1.0 - factorCoul) * prefactor;
              ecoul -= (1.0 - factorCoul) * prefactor;
            }
          } else {
            // table branch (pair_modify table N): the tables carry qqrd2e, as in the C++ table lookup
            const qqr = qqrd2e * qtmp * q[j];
            forcecoul = qqr * tab.force(rsq);
            ecoul = qqr * tab.energy(rsq);
            if (factorCoul < 1.0) {
              const bare = qqr * tab.coul(rsq);
              forcecoul -= (1.0 - factorCoul) * bare;
              ecoul -= (1.0 - factorCoul) * bare;
            }
          }
        }
        const lt = itype * nt + jtype;
        const cutLJsq = this.cutLJsq[lt];
        let prefactor2 = 0, erfc2 = 0;
        let expb = 0, r6inv = 0, r = 0;
        if (rsq < cutLJsq) {
          r = Math.sqrt(rsq);
          expb = this.lj3[lt] * libmExp(-this.lj1[lt] * r);
          forcelj = expb * this.lj1[lt] * r;
          r6inv = r2inv * r2inv * r2inv;
          forcelj -= 6.0 * this.lj4[lt] * r6inv;
          if (this.lj2[lt] !== 0.0) {
            // the Gaussian-smeared correction: erfc(lj2 r) with the C library's erfc (../libm/erf.ts)
            const rrij = this.lj2[lt] * r;
            const expn2 = libmExp(-rrij * rrij);
            erfc2 = libmErfc(rrij);
            prefactor2 = -qqrd2e * qtmp * q[j] / r;
            forcecoul2 = prefactor2 * (erfc2 + EWALD_F * rrij * expn2);
          }
          ecoul += prefactor2 * erfc2 * factorCoul;
          evdwl = expb - this.lj4[lt] * r6inv - this.offset[lt];
        }
        if (truncw > 0 && rsq < cutLJsq) {
          const cutLJ = this.cutLJ[lt];
          if (r > cutLJ - truncw) {
            // switch3 truncation of the LJ term over the last truncw of the cutoff
            const trx = (cutLJ - r) * truncwi;
            const tr = trx * trx * (3.0 - 2.0 * trx);
            const ftr = 6.0 * trx * (1.0 - trx) * r * truncwi;
            forcelj = forcelj * tr + evdwl * ftr;
            evdwl *= tr;
          }
        }
        const fpair = (forcecoul + factorCoul * forcecoul2 + factorLJ * forcelj) * r2inv;
        evdwl *= factorLJ;
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
        evdwlSum += evdwl;
        ecoulSum += ecoul;
        // ev_tally(i, j, nlocal, newton_pair, evdwl, ecoul, fpair, del): half of the energy and of the virial per atom
        if (tally) tallyAtom(pc, i, j, evdwl + ecoul, fpair, delx, dely, delz);
      }
    }
    pc.acc.evdwl += evdwlSum;
    pc.acc.ecoul += ecoulSum;
  }
}
