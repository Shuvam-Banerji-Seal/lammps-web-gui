import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { parseNum } from '../util';

/*
 * pair_style lj/cut/dipole/cut and lj/sf/dipole/sf — docs.lammps.org/pair_dipole.html:
 *   pair_style lj/cut/dipole/cut cutoff (cutoff2)
 *   pair_style lj/sf/dipole/sf cutoff (cutoff2)
 * "Style lj/cut/dipole/cut computes interactions between pairs of particles that each have a
 * charge and/or a point dipole moment." The charge-charge, charge-dipole and dipole-dipole energies
 * are the page's E_qq, E_qp and E_pp (no C/eps factor is written in the formulas; the page says that
 * "all of these terms (except Elj) have a" C/epsilon prefactor, C/epsilon being pc.qqrd2e).
 * "the vector r = Ri - Rj is the separation vector between the two particles."
 * Shifted force (lj/sf/dipole/sf): the page gives E_qq with the factor (1 - r/r_c)^2, the charge-dipole
 * term with S(r) = 1 - 3(r/r_c)^2 + 2(r/r_c)^3 and the dipole-dipole term with
 * T(r) = 1 - 4(r/r_c)^3 + 3(r/r_c)^4 (its F and T expressions are the gradients of these energies;
 * tests/engineDipole.test.ts checks them against finite differences).
 * Torque on a point dipole is p x E, written here as -p x dU/dp.
 * "If one cutoff is specified in the pair_style command, it is used for both the LJ and Coulombic (q,p)
 * terms. If two cutoffs are specified, they are used as cutoffs for the LJ and Coulombic (q,p) terms
 * respectively." "This pair style also supports an optional *scale* keyword as part of a pair_coeff
 * statement, where the interactions can be scaled according to this factor."
 * (Measured with native LAMMPS, black box: the scale keyword is rejected by lj/cut/dipole/cut, and for
 * lj/sf/dipole/sf it scales only the Coulomb terms; an I J pair without its own pair_coeff gets scale 1.)
 * Torques need per-atom torque, e.g. "atom_style hybrid sphere dipole" (page text, Description).
 * lj/cut/dipole/long and lj/long/dipole/long are not implemented (they need kspace ewald/dipole).
 */

abstract class PairDipoleBase extends Pair {
  virialFdotr = true;
  cutLJGlobal = 0;
  cutCGlobal = 0;
  p!: PairParams;
  cutLJsq = new Float64Array(0);
  cutCsq = new Float64Array(0);
  /** lj/sf/dipole/sf: shifted-force forms. */
  protected sf = false;

  settings(args: string[]): void {
    if (args.length < 1 || args.length > 2) throw new StyleError(`usage: pair_style ${this.name} cutoff (cutoff2)`);
    this.cutLJGlobal = parseNum(args[0], 'cutoff');
    this.cutCGlobal = args.length === 2 ? parseNum(args[1], 'cutoff2') : this.cutLJGlobal;
    if (!(this.cutLJGlobal > 0) || !(this.cutCGlobal > 0)) throw new StyleError('pair_style cutoffs must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cutlj', 'cutc', 'scale']);
  }

  coeff(args: string[]): void {
    const words = [...args];
    let scale = 1;
    const k = words.indexOf('scale');
    if (k >= 0) {
      // Measured with native LAMMPS (black box): lj/cut/dipole/cut rejects the keyword (it reads scale as a number
      // and stops); lj/sf/dipole/sf accepts it (see the docs quote above).
      if (!this.sf) throw new StyleError(`pair_coeff keyword 'scale' is not supported for pair style ${this.name}`);
      if (words[k + 1] === undefined) throw new StyleError('pair_coeff scale needs a value');
      scale = parseNum(words[k + 1], 'scale');
      words.splice(k, 2);
    }
    if (words.length < 4 || words.length > 6) throw new StyleError(`usage: pair_coeff I J epsilon sigma [cutoff1 [cutoff2]] (${this.name})`);
    const eps = parseNum(words[2], 'epsilon');
    const sig = parseNum(words[3], 'sigma');
    let cutLJ = this.cutLJGlobal, cutC = this.cutCGlobal;
    if (words.length === 5) cutLJ = cutC = parseNum(words[4], 'cutoff');
    if (words.length === 6) { cutLJ = parseNum(words[4], 'cutoff'); cutC = parseNum(words[5], 'cutoff2'); }
    this.p.setRange(words[0], words[1], [eps, sig, cutLJ, cutC, scale]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      p.setMixed(i, j, 'epsilon', mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'sigma', mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'cutlj', mixDistance(this.mix, p.get('cutlj', i, i), p.get('cutlj', j, j)));
      p.setMixed(i, j, 'cutc', mixDistance(this.mix, p.get('cutc', i, i), p.get('cutc', j, j)));
      // Measured with native LAMMPS (black box): an unset I J pair gets scale 1 whatever the I I and J J scales.
      p.setMixed(i, j, 'scale', 1);
    }
    const nt = this.ntypes + 1;
    if (this.cutLJsq.length !== nt * nt) {
      this.cutLJsq = new Float64Array(nt * nt);
      this.cutCsq = new Float64Array(nt * nt);
    }
    const cl = p.get('cutlj', i, j), cc = p.get('cutc', i, j);
    this.cutLJsq[i * nt + j] = this.cutLJsq[j * nt + i] = cl * cl;
    this.cutCsq[i * nt + j] = this.cutCsq[j * nt + i] = cc * cc;
    return Math.max(cl, cc);
  }

  init(ctx: StyleContext): void {
    this.cutLJsq = new Float64Array(0);
    this.cutCsq = new Float64Array(0);
    super.init(ctx);
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.mu) throw new StyleError(`pair_style ${this.name} requires atom_style dipole (or hybrid with dipole)`);
    if (!s.torque) throw new StyleError(`pair_style ${this.name} requires atom attribute torque (use atom_style hybrid sphere dipole)`);
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const mu = s.mu;
    const nt = this.ntypes + 1;
    const tq = new Float64Array(3 * pc.nall);
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const qqrd2e = pc.qqrd2e;
    const sf = this.sf;
    const cutsq = this.cutsq;
    let evdwl = 0, ecoul = 0;
    for (let i = 0; i < list.inum; i++) {
      const oi = owner[i];
      const qi = q[i];
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
        const fl = sL[jj >>> SBBITS], fc = sC[jj >>> SBBITS];
        const ti_ = type[i], tj_ = type[j];
        const scale = this.p.get('scale', ti_, tj_);
        const qj = q[j];
        const pjx = mu[4 * oj], pjy = mu[4 * oj + 1], pjz = mu[4 * oj + 2];
        const r = Math.sqrt(rsq), rinv = 1 / r, rinv2 = rinv * rinv;
        let e = 0, eLJ = 0;
        // force on i, as a coefficient of each component: accumulated as fx, fy, fz
        let fx = 0, fy = 0, fz = 0;

        // Lennard-Jones (no special-bond factor on the dispersion term beyond specialLJ)
        if (rsq < this.cutLJsq[t]) {
          const eps = this.p.get('epsilon', ti_, tj_), sig = this.p.get('sigma', ti_, tj_);
          const cutLJ = this.p.get('cutlj', ti_, tj_);
          const sr2 = sig * sig * rinv2, sr6 = sr2 * sr2 * sr2, sr12 = sr6 * sr6;
          let elj: number, fljr: number;
          if (!sf) {
            elj = 4 * eps * (sr12 - sr6);
            fljr = (48 * eps * sr12 - 24 * eps * sr6) * rinv2;
          } else {
            const src2 = (sig / cutLJ) ** 2, src6 = src2 * src2 * src2, src12 = src6 * src6;
            elj = 4 * eps * ((sr12 - sr6) + (6 * src12 - 3 * src6) * (rsq / (cutLJ * cutLJ)) - 7 * src12 + 4 * src6);
            fljr = (48 * eps * sr12 - 24 * eps * sr6) * rinv2 - (48 * eps * src12 - 24 * eps * src6) / (cutLJ * cutLJ);
          }
          // Measured with native LAMMPS (black box): scale multiplies only the Coulomb (q, p) terms, not E_LJ.
          const sfac = fl;
          eLJ = sfac * elj;
          evdwl += eLJ;
          fx += sfac * fljr * dx; fy += sfac * fljr * dy; fz += sfac * fljr * dz;
        }

        // Coulomb (charge, dipole) terms, C/eps * scale * special_coul prefactor
        if (rsq < this.cutCsq[t]) {
          const qq = qi * qj;
          const A = qqrd2e * scale * fc;
          const pjr = pjx * dx + pjy * dy + pjz * dz;
          const pir = pix * dx + piy * dy + piz * dz;
          const pipj = pix * pjx + piy * pjy + piz * pjz;
          const hasDi = pix !== 0 || piy !== 0 || piz !== 0;
          const hasDj = pjx !== 0 || pjy !== 0 || pjz !== 0;
          // radial functions: charge-dipole f1 = S/r^3 (S = 1 for cut), dipole-dipole T (T = 1 for cut)
          let f1 = rinv2 * rinv, f1p = -3 * rinv2 * rinv2, tg = 1, tgp = 0;
          let qe = 0, qf = 0; // charge-charge: energy and force / r
          if (!sf) {
            qe = qq * rinv;
            qf = qq * rinv2 * rinv;
          } else {
            const rc = Math.sqrt(this.cutCsq[t]);
            const xr = r / rc, x2 = xr * xr, x3 = x2 * xr;
            const S = 1 - 3 * x2 + 2 * x3;
            const Sp = (-6 * xr + 6 * x2) / rc;
            f1 = S * rinv2 * rinv;
            f1p = Sp * rinv2 * rinv - 3 * S * rinv2 * rinv2;
            tg = 1 - 4 * x3 + 3 * x2 * x2;
            tgp = (-12 * x2 + 12 * x3) / rc;
            const omx = 1 - xr;
            qe = qq * omx * omx * rinv;
            qf = qq * (rinv2 - 1 / (rc * rc)) * rinv;
          }
          if (qq !== 0) {
            e += A * qe;
            fx += A * qf * dx; fy += A * qf * dy; fz += A * qf * dz;
          }
          // charge-dipole: U = f1 (qi (pj.r) - qj (pi.r))
          if (qi !== 0 || qj !== 0) {
            const Ud = qi * pjr - qj * pir;
            e += A * f1 * Ud;
            // F_i = -(f1'/r) r_vec Ud - f1 (qi pj - qj pi)
            const c = -A * f1p * rinv * Ud;
            fx += c * dx - A * f1 * (qi * pjx - qj * pix);
            fy += c * dy - A * f1 * (qi * pjy - qj * piy);
            fz += c * dz - A * f1 * (qi * pjz - qj * piz);
            // torque on i: tau_i = qj f1 (p_i x r); on j: tau_j = -qi f1 (p_j x r)
            const ci = A * qj * f1, cj = -A * qi * f1;
            if (hasDi) {
              tq[3 * i] += ci * (piy * dz - piz * dy);
              tq[3 * i + 1] += ci * (piz * dx - pix * dz);
              tq[3 * i + 2] += ci * (pix * dy - piy * dx);
            }
            if (hasDj) {
              tq[3 * j] += cj * (pjy * dz - pjz * dy);
              tq[3 * j + 1] += cj * (pjz * dx - pjx * dz);
              tq[3 * j + 2] += cj * (pjx * dy - pjy * dx);
            }
          }
          // dipole-dipole: U = T(r) a, a = (pi.pj)/r^3 - 3 (pi.r)(pj.r)/r^5
          if (hasDi && hasDj) {
            const r3 = rinv2 * rinv, r5 = r3 * rinv2, r7 = r5 * rinv2;
            const a = pipj * r3 - 3 * pir * pjr * r5;
            e += A * tg * a;
            // grad_r a = -3 (pi.pj) r/r^5 - 3 [pi (pj.r) + pj (pi.r)]/r^5 + 15 (pi.r)(pj.r) r/r^7
            const gx = -3 * pipj * dx * r5 - 3 * (pix * pjr + pjx * pir) * r5 + 15 * pir * pjr * dx * r7;
            const gy = -3 * pipj * dy * r5 - 3 * (piy * pjr + pjy * pir) * r5 + 15 * pir * pjr * dy * r7;
            const gz = -3 * pipj * dz * r5 - 3 * (piz * pjr + pjz * pir) * r5 + 15 * pir * pjr * dz * r7;
            const Tp = tgp * rinv * a;
            fx -= A * (Tp * dx + tg * gx);
            fy -= A * (Tp * dy + tg * gy);
            fz -= A * (Tp * dz + tg * gz);
            // tau_i = -T/r^3 (p_i x p_j) + 3T/r^5 (p_j.r)(p_i x r); tau_j = +T/r^3 (p_i x p_j) + 3T/r^5 (p_i.r)(p_j x r)
            const cI = A * tg * r3, cI2 = 3 * A * tg * r5;
            const cxx = piy * pjz - piz * pjy, cxy = piz * pjx - pix * pjz, cxz = pix * pjy - piy * pjx;
            const ix = piy * dz - piz * dy, iy = piz * dx - pix * dz, iz = pix * dy - piy * dx;
            const jx = pjy * dz - pjz * dy, jy = pjz * dx - pjx * dz, jz = pjx * dy - pjy * dx;
            tq[3 * i] += -cI * cxx + cI2 * pjr * ix;
            tq[3 * i + 1] += -cI * cxy + cI2 * pjr * iy;
            tq[3 * i + 2] += -cI * cxz + cI2 * pjr * iz;
            tq[3 * j] += cI * cxx + cI2 * pir * jx;
            tq[3 * j + 1] += cI * cxy + cI2 * pir * jy;
            tq[3 * j + 2] += cI * cxz + cI2 * pir * jz;
          }
          ecoul += e;
        }
        f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        if (pc.eatom) {
          const h = 0.5 * (eLJ + e);
          pc.eatom[i] += h; pc.eatom[j] += h;
        }
        if (pc.vatom) {
          const va = pc.vatom;
          const w = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
          for (let c = 0; c < 6; c++) { va[6 * i + c] += 0.5 * w[c]; va[6 * j + c] += 0.5 * w[c]; }
        }
      }
    }
    nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }
}

export class PairLJCutDipoleCut extends PairDipoleBase {
  readonly name = 'lj/cut/dipole/cut';
}

export class PairLJSFDipoleSF extends PairDipoleBase {
  readonly name = 'lj/sf/dipole/sf';
  protected sf = true;
}
