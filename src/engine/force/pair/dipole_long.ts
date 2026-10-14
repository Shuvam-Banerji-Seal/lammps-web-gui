import { StyleError, mixDistance, mixEpsilon, type PairCompute } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { parseNum } from '../util';
import { erfcPoly } from '../erfc';
import { PairDipoleBase } from './dipole';

/*
 * pair_style lj/cut/dipole/long — docs.lammps.org/pair_dipole.html: "Style *lj/cut/dipole/long*
 * computes the short-range portion of point-dipole interactions as discussed in (Toukmaji).
 * Dipole-dipole, dipole-charge, and charge-charge interactions are all supported, along with the
 * standard 12/6 Lennard-Jones interactions, which are computed with a cutoff. A kspace_style must be
 * defined to use this pair style." Its arguments: "pair_style lj/cut/dipole/long cutoff (cutoff2)"
 * with "cutoff = global cutoff LJ (and Coulombic if only 1 arg) (distance units)".
 * The page also says: "When using a long-rang Coulomb solver, only a global Coulomb cutoff may be
 * used and only the LJ cutoff may be changed with the pair_coeff command." This engine refuses a
 * second (Coulomb) cutoff in pair_coeff.
 *
 * Dipole-dipole real-space energy with the Ewald splitting parameter g (Wang and Holm, arXiv
 * cond-mat/0107064, eqs 3, 7, 8, 12; r = r_i - r_j, x = g r):
 *   U_pp  = C [ (mu_i . mu_j) B(r) - (mu_i . r)(mu_j . r) C(r) ]
 *   B(r)  = [erfc(x) + (2x/sqrt(pi)) exp(-x^2)] / r^3
 *   C(r)  = [3 erfc(x) + (2x/sqrt(pi)) (3 + 2x^2) exp(-x^2)] / r^5
 *   D(r)  = [15 erfc(x) + (2x/sqrt(pi)) (15 + 10x^2 + 4x^4) exp(-x^2)] / r^7
 *   F_i   = C [ (mu_i . mu_j) C r + mu_i (mu_j . r) C + mu_j (mu_i . r) C - (mu_i . r)(mu_j . r) D r ]
 *   tau_i = C [ -B (mu_i x mu_j) + C (mu_i x r)(mu_j . r) ],  tau_j = C [ -B (mu_j x mu_i) + C (mu_j x r)(mu_i . r) ]
 * (the pair energy is written once per pair; the paper's double sum with 1/2 gives the same).
 * Measured with native LAMMPS (black box): the erfc is the Abramowitz and Stegun 7.1.26 polynomial
 * of pair_modify table 0 and the 2/sqrt(pi) factor is exact (a 4 atom system at a fixed G-ewald matches
 * to 2e-15 relative with those choices; the exact erfc differs by 7e-8, and a 2/sqrt(pi) rounded to 8
 * digits differs by 5e-9). Measured: pair_modify table 0 and
 * the default table give identical dipole energies, so no table is emulated here.
 *
 * Special bonds (measured with native LAMMPS, see the oracle case w44dl_special): the factor f of the
 * special pair removes (1 - f) of the bare dipole-dipole term, i.e. B -> B - (1-f)/r^3 and
 * C -> C - 3(1-f)/r^5 (the same subtraction the coul/long styles make for charges).
 *
 * The LJ term is the plain 12/6 cut (no shift), with the special LJ factor, as in lj/cut/dipole/cut.
 * Charges: the charge-charge and charge-dipole real-space terms of native pair_lj_cut_dipole_long.cpp (the block under
 * "if (rsq < cut_coulsq)" with g0 = qi qj and g1 = qi (pj.r) - qj (pi.r) + pi.pj) are not ported. A pair with a
 * nonzero charge inside the Coulomb cutoff is refused by name; a charged system whose pairs all lie outside the
 * Coulomb cutoff (no real-space Coulomb term at all) runs with the kspace charge terms of pppm/dipole.
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

export class PairLJCutDipoleLong extends PairDipoleBase {
  readonly name = 'lj/cut/dipole/long';
  /** Long-range Coulomb (the kspace style supplies g-ewald; excluded pairs are kept). */
  coulLong = true;

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
    }
    const cl = p.get('cutlj', i, j), cc = this.cutCGlobal;
    this.cutLJsq[i * nt + j] = this.cutLJsq[j * nt + i] = cl * cl;
    this.cutCsq[i * nt + j] = this.cutCsq[j * nt + i] = cc * cc;
    return Math.max(cl, cc);
  }

  extract(name: string): unknown {
    return name === 'cut_coul' ? this.cutCGlobal : undefined;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.mu) throw new StyleError(`pair_style ${this.name} requires atom_style dipole (or hybrid with dipole)`);
    if (!s.torque) throw new StyleError(`pair_style ${this.name} requires atom attribute torque (use atom_style hybrid sphere dipole)`);
    if (!(this.gEwald > 0)) throw new StyleError(`pair_style ${this.name} needs kspace_style ewald/dipole (g-ewald is not set)`);
    const list = pc.half!;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const mu = s.mu;
    const nt = this.ntypes + 1;
    const tq = new Float64Array(3 * pc.nall);
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const qqrd2e = pc.qqrd2e;
    const q = pc.q;
    const g = this.gEwald;
    const cutsq = this.cutsq;
    const cutCsqG = this.cutCGlobal * this.cutCGlobal;
    let evdwl = 0, ecoul = 0;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const pix = mu[4 * oi], piy = mu[4 * oi + 1], piz = mu[4 * oi + 2];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = list.neighbors[k];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        if (rsq < cutCsqG && (q[i] !== 0 || q[j] !== 0)) {
          throw new StyleError(`pair_style ${this.name} with point charges inside the Coulomb cutoff is not supported by the browser engine (native LAMMPS adds the charge-charge and charge-dipole real-space terms; only the dipole terms are ported)`);
        }
        const fl = sL[jj >>> SBBITS], fc = sC[jj >>> SBBITS];
        const ti_ = type[i], tj_ = type[j];
        const pjx = mu[4 * oj], pjy = mu[4 * oj + 1], pjz = mu[4 * oj + 2];
        const r = Math.sqrt(rsq), rinv = 1 / r, rinv2 = rinv * rinv;
        let eLJ = 0, e = 0;
        let fx = 0, fy = 0, fz = 0;

        // Lennard-Jones 12/6 with the special LJ factor (as lj/cut/dipole/cut)
        if (rsq < this.cutLJsq[t]) {
          const eps = this.p.get('epsilon', ti_, tj_), sig = this.p.get('sigma', ti_, tj_);
          const sr2 = sig * sig * rinv2, sr6 = sr2 * sr2 * sr2, sr12 = sr6 * sr6;
          const elj = 4 * eps * (sr12 - sr6);
          const fljr = (48 * eps * sr12 - 24 * eps * sr6) * rinv2;
          eLJ = fl * elj;
          evdwl += eLJ;
          fx += fl * fljr * dx; fy += fl * fljr * dy; fz += fl * fljr * dz;
        }

        // dipole-dipole real-space Ewald term (rsq < cutCsqG is the global Coulomb cutoff)
        if (rsq < cutCsqG && (pix !== 0 || piy !== 0 || piz !== 0) && (pjx !== 0 || pjy !== 0 || pjz !== 0)) {
          const xg = g * r, ex = Math.exp(-xg * xg), ec = erfcPoly(xg, ex);
          const x2 = xg * xg;
          const r3 = rinv2 * rinv, r5 = r3 * rinv2, r7 = r5 * rinv2;
          const om = 1 - fc;
          // special-bond subtraction of the bare terms (1/r^3, 3/r^5, 15/r^7) scaled by (1 - f)
          const Bf = (ec + TWO_OVER_SQRTPI * xg * ex) * r3 - om * r3;
          const Cf = (3 * ec + TWO_OVER_SQRTPI * xg * (3 + 2 * x2) * ex) * r5 - 3 * om * r5;
          const Df = (15 * ec + TWO_OVER_SQRTPI * xg * (15 + 10 * x2 + 4 * x2 * x2) * ex) * r7 - 15 * om * r7;
          const pipj = pix * pjx + piy * pjy + piz * pjz;
          const pir = pix * dx + piy * dy + piz * dz;
          const pjr = pjx * dx + pjy * dy + pjz * dz;
          e = qqrd2e * (pipj * Bf - pir * pjr * Cf);
          const cfx = qqrd2e * ((pipj * Cf - pir * pjr * Df) * dx + Cf * (pix * pjr + pjx * pir));
          const cfy = qqrd2e * ((pipj * Cf - pir * pjr * Df) * dy + Cf * (piy * pjr + pjy * pir));
          const cfz = qqrd2e * ((pipj * Cf - pir * pjr * Df) * dz + Cf * (piz * pjr + pjz * pir));
          fx += cfx; fy += cfy; fz += cfz;
          // torques: tau_i = qqr2e [ -Bf (mu_i x mu_j) + Cf (mu_i x d)(mu_j . d) ], tau_j with i and j swapped
          const cxx = piy * pjz - piz * pjy, cxy = piz * pjx - pix * pjz, cxz = pix * pjy - piy * pjx;
          const ix = piy * dz - piz * dy, iy = piz * dx - pix * dz, iz = pix * dy - piy * dx;
          const jx = pjy * dz - pjz * dy, jy = pjz * dx - pjx * dz, jz = pjx * dy - pjy * dx;
          const tI = qqrd2e * Cf * pjr, tJ = qqrd2e * Cf * pir;
          tq[3 * i] += -qqrd2e * Bf * cxx + tI * ix;
          tq[3 * i + 1] += -qqrd2e * Bf * cxy + tI * iy;
          tq[3 * i + 2] += -qqrd2e * Bf * cxz + tI * iz;
          tq[3 * j] += qqrd2e * Bf * cxx + tJ * jx;
          tq[3 * j + 1] += qqrd2e * Bf * cxy + tJ * jy;
          tq[3 * j + 2] += qqrd2e * Bf * cxz + tJ * jz;
        }
        ecoul += e;
        f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        if (pc.eatom) {
          const h = 0.5 * (eLJ + e);
          pc.eatom[i] += h; pc.eatom[j] += h;
        }
      }
    }
    nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }
}
