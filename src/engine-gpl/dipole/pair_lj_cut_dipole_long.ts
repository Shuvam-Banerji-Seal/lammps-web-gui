/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/DIPOLE/pair_lj_cut_dipole_long.cpp
 * and pair_lj_cut_dipole_long.h (PairLJCutDipoleLong: pair_style lj/cut/dipole/long, compute, allocate, settings, coeff,
 * init_one, init_style, extract), with the erfc polynomial and its constants of src/ewald_const.h (EWALD_P, A1..A5) and
 * the constant MY_PIS (sqrt(pi)) of src/math_const.h, at the same commit. Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The pair term is written in the operation order of the source (charges g0, g1, g2 and the dipole coefficients b0..b3
 * for the force and d0..d3 for the factor_coul < 1 correction), so the forces, torques and tallies agree with native
 * LAMMPS to round-off. The engine's own atom indexing is kept: the moments of a ghost are those of its owner.
 *
 * Not ported: pair_modify shift (refused by the base class).
 */
import { StyleError, mixDistance, mixEpsilon, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { PairDipoleBase } from '../../engine/force/pair/dipole';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

/** ewald_const.h: the erfc approximation (Abramowitz and Stegun 7.1.26) of LAMMPS, b332f28. */
const EWALD_P = 0.3275911;
const A1 = 0.254829592;
const A2 = -0.284496736;
const A3 = 1.421413741;
const A4 = -1.453152027;
const A5 = 1.061405429;
/** math_const.h: MY_PIS = sqrt(pi). */
const MY_PIS = 1.77245385090551602729;

export class PairLJCutDipoleLong extends PairDipoleBase {
  readonly name = 'lj/cut/dipole/long';
  /** Long-range Coulomb (the kspace style supplies g-ewald; excluded pairs are kept). */
  coulLong = true;
  /** Per type pair (i * nt + j), as init_one: lj1 = 48 eps s^12, lj2 = 24 eps s^6, lj3 = 4 eps s^12, lj4 = 4 eps s^6. */
  private lj1 = new Float64Array(0);
  private lj2 = new Float64Array(0);
  private lj3 = new Float64Array(0);
  private lj4 = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length < 1 || args.length > 2) throw new StyleError('usage: pair_style lj/cut/dipole/long cutoff (cutoff2)');
    this.cutLJGlobal = parseNum(args[0], 'cutoff');
    this.cutCGlobal = args.length === 2 ? parseNum(args[1], 'cutoff2') : this.cutLJGlobal;
    if (!(this.cutLJGlobal > 0) || !(this.cutCGlobal > 0)) throw new StyleError('pair_style cutoffs must be > 0');
  }

  coeff(args: string[]): void {
    if (args.length < 4 || args.length > 6) throw new StyleError(`usage: pair_coeff I J epsilon sigma [cutoff1] (${this.name})`);
    if (args.length === 6) {
      throw new StyleError(`pair_coeff for ${this.name} accepts only the LJ cutoff: the Coulomb cutoff is global with a long-range solver`);
    }
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const cutLJ = args.length === 5 ? parseNum(args[4], 'cutoff') : this.cutLJGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, cutLJ, this.cutCGlobal, 1]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      p.setMixed(i, j, 'epsilon', mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'sigma', mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'cutlj', mixDistance(this.mix, p.get('cutlj', i, i), p.get('cutlj', j, j)));
      p.setMixed(i, j, 'cutc', this.cutCGlobal);
      p.setMixed(i, j, 'scale', 1);
    }
    const nt = this.ntypes + 1;
    if (this.cutLJsq.length !== nt * nt) {
      this.cutLJsq = new Float64Array(nt * nt);
      this.cutCsq = new Float64Array(nt * nt);
      this.lj1 = new Float64Array(nt * nt);
      this.lj2 = new Float64Array(nt * nt);
      this.lj3 = new Float64Array(nt * nt);
      this.lj4 = new Float64Array(nt * nt);
    }
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j);
    const cl = p.get('cutlj', i, j), cc = this.cutCGlobal;
    const ij = i * nt + j, ji = j * nt + i;
    this.cutLJsq[ij] = this.cutLJsq[ji] = cl * cl;
    this.cutCsq[ij] = this.cutCsq[ji] = cc * cc;
    this.lj1[ij] = this.lj1[ji] = 48.0 * eps * libmPow(sig, 12.0);
    this.lj2[ij] = this.lj2[ji] = 24.0 * eps * libmPow(sig, 6.0);
    this.lj3[ij] = this.lj3[ji] = 4.0 * eps * libmPow(sig, 12.0);
    this.lj4[ij] = this.lj4[ji] = 4.0 * eps * libmPow(sig, 6.0);
    // native: cut = MAX(cut_lj, cut_coul)
    return Math.max(cl, cc);
  }

  extract(name: string): unknown {
    // pair_lj_cut_dipole_long.cpp extract: "epsilon", "sigma" (dim 2) besides "cut_coul"
    if (name === 'epsilon' || name === 'sigma') return this.p.p(name);
    return name === 'cut_coul' ? this.cutCGlobal : undefined;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.mu) throw new StyleError(`pair_style ${this.name} requires atom_style dipole (or hybrid with dipole)`);
    if (!s.torque) throw new StyleError(`pair_style ${this.name} requires atom attribute torque (use atom_style hybrid sphere dipole)`);
    if (!(this.gEwald > 0)) throw new StyleError(`pair_style ${this.name} needs kspace_style ewald/dipole (g-ewald is not set)`);
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const mu = s.mu;
    const nt = this.ntypes + 1;
    const tq = new Float64Array(3 * pc.nall);
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const qqrd2e = pc.qqrd2e;
    const g = this.gEwald;
    const cutsq = this.cutsq;
    const cutCsqG = this.cutCGlobal * this.cutCGlobal;
    const cutLJsq = this.cutLJsq, lj1 = this.lj1, lj2 = this.lj2, lj3 = this.lj3, lj4 = this.lj4;
    const pre1 = 2.0 * g / MY_PIS;
    const pre2 = 4.0 * libmPow(g, 3.0) / MY_PIS;
    const pre3 = 8.0 * libmPow(g, 5.0) / MY_PIS;
    let evdwl = 0, ecoul = 0;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const qtmp = q[i];
      const mix = mu[4 * oi], miy = mu[4 * oi + 1], miz = mu[4 * oi + 2];
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = list.neighbors[k];
        const factorLJ = sL[jj >>> SBBITS];
        const factorCoul = sC[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jtype = type[j];
        const t = itype * nt + jtype;
        if (!(rsq < cutsq[t])) continue;
        const r2inv = 1.0 / rsq;
        const rinv = Math.sqrt(r2inv);
        const qj = q[j];
        const mjx = mu[4 * oj], mjy = mu[4 * oj + 1], mjz = mu[4 * oj + 2];
        let forcecoulx = 0, forcecouly = 0, forcecoulz = 0;
        let tixcoul = 0, tiycoul = 0, tizcoul = 0;
        let tjxcoul = 0, tjycoul = 0, tjzcoul = 0;
        let b0 = 0, b1 = 0, b2 = 0, d0 = 0, d1 = 0, d2 = 0;
        let g0 = 0, g1 = 0, g2 = 0;
        let r6inv = 0;
        if (rsq < cutCsqG) {
          const r = Math.sqrt(rsq);
          const grij = g * r;
          const expm2 = libmExp(-grij * grij);
          const tp = 1.0 / (1.0 + EWALD_P * grij);
          const erfc = tp * (A1 + tp * (A2 + tp * (A3 + tp * (A4 + tp * A5)))) * expm2;
          const pdotp = mix * mjx + miy * mjy + miz * mjz;
          const pidotr = mix * delx + miy * dely + miz * delz;
          const pjdotr = mjx * delx + mjy * dely + mjz * delz;
          g0 = qtmp * qj;
          g1 = qtmp * pjdotr - qj * pidotr + pdotp;
          g2 = -pidotr * pjdotr;
          let fdx = 0, fdy = 0, fdz = 0;
          let zdix = 0, zdiy = 0, zdiz = 0, zdjx = 0, zdjy = 0, zdjz = 0;
          if (factorCoul > 0.0) {
            b0 = erfc * rinv;
            b1 = (b0 + pre1 * expm2) * r2inv;
            b2 = (3.0 * b1 + pre2 * expm2) * r2inv;
            const b3 = (5.0 * b2 + pre3 * expm2) * r2inv;
            const g0b1_g1b2_g2b3 = g0 * b1 + g1 * b2 + g2 * b3;
            fdx = delx * g0b1_g1b2_g2b3 - b1 * (qtmp * mjx - qj * mix) + b2 * (pjdotr * mix + pidotr * mjx);
            fdy = dely * g0b1_g1b2_g2b3 - b1 * (qtmp * mjy - qj * miy) + b2 * (pjdotr * miy + pidotr * mjy);
            fdz = delz * g0b1_g1b2_g2b3 - b1 * (qtmp * mjz - qj * miz) + b2 * (pjdotr * miz + pidotr * mjz);
            zdix = delx * (qj * b1 + b2 * pjdotr) - b1 * mjx;
            zdiy = dely * (qj * b1 + b2 * pjdotr) - b1 * mjy;
            zdiz = delz * (qj * b1 + b2 * pjdotr) - b1 * mjz;
            zdjx = delx * (-qtmp * b1 + b2 * pidotr) - b1 * mix;
            zdjy = dely * (-qtmp * b1 + b2 * pidotr) - b1 * miy;
            zdjz = delz * (-qtmp * b1 + b2 * pidotr) - b1 * miz;
            if (factorCoul < 1.0) {
              fdx *= factorCoul; fdy *= factorCoul; fdz *= factorCoul;
              zdix *= factorCoul; zdiy *= factorCoul; zdiz *= factorCoul;
              zdjx *= factorCoul; zdjy *= factorCoul; zdjz *= factorCoul;
            }
          }
          let fax = 0, fay = 0, faz = 0;
          let zaix = 0, zaiy = 0, zaiz = 0, zajx = 0, zajy = 0, zajz = 0;
          if (factorCoul < 1.0) {
            // the bare (erfc - 1)/r terms: the part of the interaction the special factor removes
            d0 = (erfc - 1.0) * rinv;
            d1 = (d0 + pre1 * expm2) * r2inv;
            d2 = (3.0 * d1 + pre2 * expm2) * r2inv;
            const d3 = (5.0 * d2 + pre3 * expm2) * r2inv;
            const g0d1_g1d2_g2d3 = g0 * d1 + g1 * d2 + g2 * d3;
            fax = delx * g0d1_g1d2_g2d3 - d1 * (qtmp * mjx - qj * mix) + d2 * (pjdotr * mix + pidotr * mjx);
            fay = dely * g0d1_g1d2_g2d3 - d1 * (qtmp * mjy - qj * miy) + d2 * (pjdotr * miy + pidotr * mjy);
            faz = delz * g0d1_g1d2_g2d3 - d1 * (qtmp * mjz - qj * miz) + d2 * (pjdotr * miz + pidotr * mjz);
            zaix = delx * (qj * d1 + d2 * pjdotr) - d1 * mjx;
            zaiy = dely * (qj * d1 + d2 * pjdotr) - d1 * mjy;
            zaiz = delz * (qj * d1 + d2 * pjdotr) - d1 * mjz;
            zajx = delx * (-qtmp * d1 + d2 * pidotr) - d1 * mix;
            zajy = dely * (-qtmp * d1 + d2 * pidotr) - d1 * miy;
            zajz = delz * (-qtmp * d1 + d2 * pidotr) - d1 * miz;
            if (factorCoul > 0.0) {
              const facm1 = 1.0 - factorCoul;
              fax *= facm1; fay *= facm1; faz *= facm1;
              zaix *= facm1; zaiy *= facm1; zaiz *= facm1;
              zajx *= facm1; zajy *= facm1; zajz *= facm1;
            }
          }
          forcecoulx = fdx + fax;
          forcecouly = fdy + fay;
          forcecoulz = fdz + faz;
          tixcoul = miy * (zdiz + zaiz) - miz * (zdiy + zaiy);
          tiycoul = miz * (zdix + zaix) - mix * (zdiz + zaiz);
          tizcoul = mix * (zdiy + zaiy) - miy * (zdix + zaix);
          tjxcoul = mjy * (zdjz + zajz) - mjz * (zdjy + zajy);
          tjycoul = mjz * (zdjx + zajx) - mjx * (zdjz + zajz);
          tjzcoul = mjx * (zdjy + zajy) - mjy * (zdjx + zajx);
        }
        // LJ interaction
        let fforce = 0;
        if (rsq < cutLJsq[t]) {
          r6inv = r2inv * r2inv * r2inv;
          const forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
          fforce = factorLJ * forcelj * r2inv;
        }
        // total force
        const fx = qqrd2e * forcecoulx + delx * fforce;
        const fy = qqrd2e * forcecouly + dely * fforce;
        const fz = qqrd2e * forcecoulz + delz * fforce;
        // force and torque accumulation
        f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
        tq[3 * i] += qqrd2e * tixcoul;
        tq[3 * i + 1] += qqrd2e * tiycoul;
        tq[3 * i + 2] += qqrd2e * tizcoul;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        tq[3 * j] += qqrd2e * tjxcoul;
        tq[3 * j + 1] += qqrd2e * tjycoul;
        tq[3 * j + 2] += qqrd2e * tjzcoul;
        // energy: the Coulomb part is zero when factor_coul is 0 (native: "rsq < cut_coulsq && factor_coul > 0.0")
        let eLJ = 0, eC = 0;
        if (rsq < cutCsqG && factorCoul > 0.0) {
          eC = qqrd2e * (b0 * g0 + b1 * g1 + b2 * g2);
          if (factorCoul < 1.0) {
            eC *= factorCoul;
            eC += (1 - factorCoul) * qqrd2e * (d0 * g0 + d1 * g1 + d2 * g2);
          }
        }
        if (rsq < cutLJsq[t]) {
          eLJ = r6inv * (lj3[t] * r6inv - lj4[t]);
          eLJ *= factorLJ;
        }
        evdwl += eLJ;
        ecoul += eC;
        if (pc.eatom) {
          const h = 0.5 * (eLJ + eC);
          pc.eatom[i] += h; pc.eatom[j] += h;
        }
      }
    }
    nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }
}
