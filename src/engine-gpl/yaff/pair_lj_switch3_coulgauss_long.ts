/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/YAFF/pair_lj_switch3_coulgauss_long.cpp and src/YAFF/pair_lj_switch3_coulgauss_long.h at commit
 * b332f28 (YAFF package; the switching-function truncation and the Gaussian-charge Coulomb correction follow the
 * YAFF code of Vandenbrande, Ghysels and co-workers). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * pair_style lj/switch3/coulgauss/long — docs.lammps.org/pair_lj_switch3_coulgauss_long.html. The doc gives the
 * syntax "lj/switch3/coulgauss/long args = cutoff (cutoff2) width" with "width = width parameter of the smoothing
 * function (distance units)", and the vdW term "E = 4\epsilon [ (\sigma/r)^{12} - (\sigma/r)^6 ]" which "goes
 * smoothly to zero at the cutoff r_c as defined by the switching function S_3(r) ... 3x^2 - 2x^3 ... with
 * x = (r_c - r)/w". The Coulomb term is "E = q_i q_j erf(r/\sqrt{\gamma_1^2+\gamma_2^2}) / (\epsilon r_{ij})", which
 * the code evaluates as the long-range Ewald real-space term plus the Gaussian correction
 * -q_i q_j erfc(r/\sqrt{\gamma_i^2+\gamma_j^2})/r (the 1/r of the k-space part minus its erfc part). The doc says
 * "These pair styles support the pair_modify table and options since they can tabulate the short-range portion of
 * the long-range Coulombic interactions", and "These pair styles do not support the pair_modify tail option".
 * Coefficients (pair_coeff): "\epsilon (energy)", "\sigma (distance)", "\gamma (distance)"; the optional cutoff.
 *
 * Measured/ported details (native pair_lj_switch3_coulgauss_long.cpp):
 *  - lj2 (the inverse Gaussian width) is 1/sqrt(gamma_ii^2 + gamma_jj^2) from the DIAGONAL gammas; it is 0 when both
 *    are 0 (then no Gaussian correction).
 *  - the Gaussian correction and the LJ term are evaluated only for r < cut_lj; the Coulomb real-space term for
 *    r < cut_coul; the Gaussian term is weighted by the Coulomb special weight.
 *  - the erfc of the Gaussian correction is the exact erfc() (erfc.ts erfcExact); the Ewald term uses the
 *    pair_modify table kernels of coul/long (erfc.ts), as native does with ncoultablebits.
 *  - shift: the offset is zero when the switching function is active (width > 0), as native init_one.
 *
 * Not ported (StyleError): pair_modify tail yes (native has a closed-form tail for the switched LJ term, which is
 * not reproduced), single() (pairwise energies for compute pair/local), and the mm3/switch3/coulgauss/long style
 * (its MM3 vdW term is not registered here).
 */
import { PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { PairLJCutCoulLong, coulLongPair } from '../../engine/force/pair/coul_long';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { erfcExact, EWALD_F, ErfcTableCache } from '../../engine/force/erfc';
import { fmtCoeff, parseNum } from '../../engine/force/util';

// LAMMPS: ewald_const.h EWALD_F; the engine's erfc.ts EWALD_F is the same rounded constant used by coul/long.

export class PairLJSwitch3CoulGaussLong extends PairLJCutCoulLong {
  readonly name: string = 'lj/switch3/coulgauss/long';
  /** Width of the switching function (pair_style third argument, or the second with one cutoff). */
  truncw = 0;
  private truncwi = 0;
  /** 1/sqrt(gamma_ii^2 + gamma_jj^2) per type pair (0 when both gammas are 0). */
  private gaussInv = new Float64Array(0);
  private ljA = new Float64Array(0);
  private ljB = new Float64Array(0);
  private offS = new Float64Array(0);
  private readonly erfcTabs = new ErfcTableCache();

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'gamma', 'cut']);
  }

  settings(args: string[]): void {
    if (args.length !== 2 && args.length !== 3) {
      throw new StyleError('usage: pair_style lj/switch3/coulgauss/long cutoff (cutoff2) width');
    }
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (args.length === 2) {
      this.cutCoul = this.cutGlobal;
      this.truncw = parseNum(args[1], 'width');
    } else {
      this.cutCoul = parseNum(args[1], 'Coulomb cutoff');
      this.truncw = parseNum(args[2], 'width');
    }
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError('pair_style lj/switch3/coulgauss/long: cutoffs must be > 0');
    if (!(this.truncw >= 0)) throw new StyleError('pair_style lj/switch3/coulgauss/long: width must be >= 0');
    this.truncwi = this.truncw > 0 ? 1 / this.truncw : 0;
    // native settings(): explicitly set LJ cutoffs follow the new global cutoff
    if (this.p) {
      const nt = this.ntypes;
      for (let i = 1; i <= nt; i++) {
        for (let j = i; j <= nt; j++) {
          if (this.p.isSet(i, j)) this.p.set(i, j, [this.p.get('epsilon', i, j), this.p.get('sigma', i, j), this.p.get('gamma', i, j), this.cutGlobal]);
        }
      }
    }
  }

  coeff(args: string[]): void {
    if (args.length < 5 || args.length > 6) throw new StyleError('usage: pair_coeff I J epsilon sigma gamma [cutoff]');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const gam = parseNum(args[4], 'gamma');
    const cut = args[5] !== undefined ? parseNum(args[5], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, gam, cut]);
  }

  initOne(i: number, j: number): number {
    if (this.tail) throw new StyleError('pair_modify tail yes is not supported for pair style lj/switch3/coulgauss/long');
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      const eps = mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j));
      const sig = mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j));
      const cut = mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j));
      p.setMixed(i, j, 'epsilon', eps);
      p.setMixed(i, j, 'sigma', sig);
      p.setMixed(i, j, 'cut', cut);
    }
    const nt = this.ntypes + 1;
    if (this.ljA.length !== nt * nt) {
      this.gaussInv = new Float64Array(nt * nt);
      this.ljA = new Float64Array(nt * nt);
      this.ljB = new Float64Array(nt * nt);
      this.offS = new Float64Array(nt * nt);
    }
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), cutLJ = p.get('cut', i, j);
    const gii = p.get('gamma', i, i), gjj = p.get('gamma', j, j);
    const gInv = gii === 0 && gjj === 0 ? 0 : 1 / Math.sqrt(gii * gii + gjj * gjj);
    const s6 = sig ** 6, s12 = s6 * s6;
    const lj3 = 4 * eps * s12, lj4 = 4 * eps * s6;
    let off = 0;
    if (this.shift && cutLJ > 0 && this.truncw === 0) {
      const r2inv = 1 / (cutLJ * cutLJ), r6inv = r2inv * r2inv * r2inv;
      off = lj3 * r6inv * r6inv - lj4 * r6inv;
    }
    const k1 = i * nt + j, k2 = j * nt + i;
    this.gaussInv[k1] = this.gaussInv[k2] = gInv;
    this.ljA[k1] = this.ljA[k2] = lj3;
    this.ljB[k1] = this.ljB[k2] = lj4;
    this.offS[k1] = this.offS[k2] = off;
    return Math.max(cutLJ, this.cutCoul);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const cutLJ = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) cutLJ[i * nt + j] = this.p.get('cut', i, j);
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.erfcTabs.get(this.table, g, cutcsq, this.tabinnerRsq);
    const poly = this.table === 0;
    const { gaussInv, truncw, truncwi } = this;
    const { ljA: lj3, ljB: lj4, offS: off } = this;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const qqrd2e = pc.qqrd2e;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const qj = q[j];
        const fc = sC[sb], fl = sLJ[sb];
        let fpair = 0, e = 0;
        // Ewald real-space Coulomb (native forcecoul, prefactor, erfc1), inside the Coulomb cutoff
        if (rsq < cutcsq && qi !== 0 && qj !== 0) {
          const c = coulLongPair(rsq, qi, qj, g, qqrd2e, fc, poly, tab);
          fpair += c.f;
          ecoul += c.e;
          e += c.e;
        }
        const gi = gaussInv[t];
        // Gaussian-charge correction (native erfc2, prefactor2, forcecoul2), inside the LJ cutoff
        if (rsq < cutLJ[t] * cutLJ[t] && gi !== 0 && qi !== 0 && qj !== 0) {
          const r = Math.sqrt(rsq);
          const rrij = gi * r;
          const expn2 = Math.exp(-rrij * rrij);
          const erfc2 = erfcExact(rrij);
          const prefactor2 = (-qqrd2e * qi * qj) / r;
          const forcecoul2 = prefactor2 * (erfc2 + EWALD_F * rrij * expn2);
          fpair += (fc * forcecoul2) / rsq;
          const ec2 = prefactor2 * erfc2 * fc;
          ecoul += ec2;
          e += ec2;
        }
        // switched Lennard-Jones (native forcelj, evdwl, then truncation by the switching function S_3)
        if (rsq < cutLJ[t] * cutLJ[t]) {
          const r = Math.sqrt(rsq);
          const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
          let forcelj = r6inv * (12 * lj3[t] * r6inv - 6 * lj4[t]);
          let ev = r6inv * (lj3[t] * r6inv - lj4[t]) - off[t];
          const cl = cutLJ[t];
          if (truncw > 0 && r > cl - truncw) {
            const trx = (cl - r) * truncwi;
            const tr = trx * trx * (3 - 2 * trx);
            const ftr = (6 * trx * (1 - trx) * r * truncwi);
            forcelj = forcelj * tr + ev * ftr;
            ev *= tr;
          }
          fpair += fl * forcelj * r2inv;
          ev *= fl;
          evdwl += ev;
          e += ev;
        }
        if (fpair === 0 && e === 0) continue;
        fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  /** Pairwise energies (compute pair/local, hbond/local) are not reproduced for this style: refused by name. */
  single(): { eng: number; fforce: number } {
    throw new StyleError('pairwise energies (compute pair/local, hbond/local) are not supported for pair style lj/switch3/coulgauss/long');
  }

  /** write_data: "i epsilon sigma gamma" per type (native write_data). */
  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))} ${fmtCoeff(this.p.get('gamma', i, i))}`);
    }
    return out;
  }
}
