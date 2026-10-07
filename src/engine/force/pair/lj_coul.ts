import { PairParams, StyleError, mixDistance, type PairCompute } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { PairLJCut, tallyAtom } from './lj_cut';
import { fmtCoeff, parseNum } from '../util';
import { erfcExact, erfcPoly } from '../erfc';

/*
 * pair_style lj/cut/coul/cut, lj/cut/coul/debye, lj/cut/coul/dsf and
 * lj/cut/coul/wolf — docs.lammps.org/pair_lj_cut_coul.html (the bare coul/
 * styles carry the same formulas on docs.lammps.org/pair_coul.html).
 *
 * The Lennard-Jones part is the base lj/cut potential (PairLJCut). From
 * pair_lj_cut_coul.html:
 *   "The lj/cut/coul styles compute the standard 12/6 Lennard-Jones
 *   potential, given by"
 *     E = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
 *         \left(\frac{\sigma}{r}\right)^6 \right]
 *                         \qquad r < r_c
 * pair_coeff coefficients (pair_lj_cut_coul.html):
 *   * :math:`\epsilon` (energy units)
 *   * :math:`\sigma` (distance units)
 *   * cutoff1 (distance units)
 *   * cutoff2 (distance units)
 *   "The latter 2 coefficients are optional.  If not specified, the global
 *   LJ and Coulombic cutoffs specified in the pair_style command are used.
 *   If only one cutoff is specified, it is used as the cutoff for both LJ
 *   and Coulombic interactions for this type pair.  If both coefficients
 *   are specified, they are used as the LJ and Coulombic cutoffs for this
 *   type pair."
 * Mixing (pair_lj_cut_coul.html): "For atom type pairs I,J and I != J, the
 * epsilon and sigma coefficients and cutoff distance for all of the lj/cut
 * pair styles can be mixed. The default mix value is geometric." Both
 * cutoffs mix like sigma (pair_modify.html: "the cutoff distance is mixed
 * the same way as sigma").
 * pair_modify shift (pair_lj_cut_coul.html): "All of the lj/cut pair
 * styles support the pair_modify shift option for the energy of the
 * Lennard-Jones portion of the pair interaction." — handled by PairLJCut.
 * Coulomb prefactor: pc.qqrd2e = qqr2e/dielectric; pair_lj_cut_coul.html
 * writes the Coulombic energy with "C is an energy-conversion constant ...
 * and epsilon is the dielectric constant which can be set by the dielectric
 * command" (docs.lammps.org/dielectric.html: "The value is used in the
 * denominator of the formulas for Coulombic interactions").
 * erfc() in the dsf/wolf formulas is evaluated at machine precision
 * (erfc.ts erfcExact), matching the erfc() function the formula names.
 * special_bonds Coulomb weights scale the whole pair term for coul/cut and
 * coul/debye. For coul/dsf and coul/wolf, measured with native LAMMPS (a
 * charged 4-atom chain, weights 0.25, 0.5, 1.0 on each neighbor order;
 * ecoul and forces to 15 digits), a special pair keeps the full damped term
 * minus (1 - w) times the bare C q_i q_j / r, and pairs with weight 0.0
 * stay in the neighbor list (Pair.keepExcluded, from special_bonds.html).
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/**
 * Shared base for the lj/cut/coul/* styles: PairLJCut for the LJ part plus a
 * per-pair Coulombic cutoff ('cut_coul') and a style-specific coulPair().
 */
abstract class PairLJCutCoul extends PairLJCut {
  /** Global Coulombic cutoff from pair_style; per-pair values live in p 'cut_coul'. */
  cutCoul = 0;

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cut', 'cut_coul']);
  }

  /**
   * Whether pair_coeff may set a per-pair Coulomb cutoff. Measured with
   * native LAMMPS: lj/cut/coul/cut and lj/cut/coul/debye follow the
   * documented rule above; lj/cut/coul/dsf and lj/cut/coul/wolf accept only
   * the LJ cutoff (a second one is "Incorrect args for pair coefficients")
   * and keep the global Coulomb cutoff for every pair.
   */
  protected coulCutPerPair = true;

  /** dsf/wolf: a special pair keeps the damped term minus (1 - w) of the bare C q_i q_j / r (see top). */
  protected coulSpecialSubtract = false;

  coeff(args: string[]): void {
    if (args.length < 4 || args.length > (this.coulCutPerPair ? 6 : 5)) {
      throw new StyleError(`usage: pair_coeff I J epsilon sigma [cutoff1${this.coulCutPerPair ? ' [cutoff2]' : ''}] (${this.name})`);
    }
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const cut = args[4] !== undefined ? parseNum(args[4], 'cutoff1') : this.cutGlobal;
    // "If only one cutoff is specified, it is used as the cutoff for both LJ
    // and Coulombic interactions for this type pair." (pair_lj_cut_coul.html)
    const cutC = !this.coulCutPerPair ? this.cutCoul
      : args[5] !== undefined ? parseNum(args[5], 'cutoff2') : args[4] !== undefined ? cut : this.cutCoul;
    if (!(cut > 0) || !(cutC > 0)) throw new StyleError('pair_coeff cutoffs must be > 0');
    this.p.setRange(args[0], args[1], [eps, sig, cut, cutC]);
  }

  initOne(i: number, j: number): number {
    const cutLJ = super.initOne(i, j);
    const p = this.p;
    if (!p.isSet(i, j)) {
      p.setMixed(i, j, 'cut_coul', mixDistance(this.mix, p.get('cut_coul', i, i), p.get('cut_coul', j, j)));
    }
    this.initCoul(i, j);
    return Math.max(cutLJ, p.get('cut_coul', i, j));
  }

  /** Hook for per-pair Coulomb shift constants (dsf/wolf). */
  protected initCoul(_i: number, _j: number): void {}

  /** Coulomb self energy of one owned atom (dsf/wolf), 0 otherwise. */
  protected coulSelf(_qi: number, _qqrd2e: number): number { return 0; }

  /** Unscaled Coulomb energy e and force-over-r f of one pair. */
  protected abstract coulPair(t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number };

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { lj1, lj2, lj3, lj4, offset } = this;
    const cutljsq = new Float64Array(nt * nt);
    const cutcoulsq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        cutljsq[i * nt + j] = this.p.get('cut', i, j) ** 2;
        cutcoulsq[i * nt + j] = this.p.get('cut_coul', i, j) ** 2;
      }
    }
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      const eself = this.coulSelf(qi, pc.qqrd2e);
      if (eself !== 0) {
        ecoul += eself;
        if (pc.eatom) pc.eatom[i] += eself;
      }
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        let fpair = 0, e = 0;
        if (rsq < cutcoulsq[t] && qi !== 0 && q[j] !== 0) {
          const fc = sC[sb];
          const c = this.coulPair(t, rsq, qi, q[j], pc.qqrd2e);
          let ef = fc * c.e, ff = fc * c.f;
          if (fc !== 1 && this.coulSpecialSubtract) {
            const bare = pc.qqrd2e * qi * q[j] / Math.sqrt(rsq);
            ef = c.e - (1 - fc) * bare;
            ff = c.f - (1 - fc) * bare / rsq;
          }
          fpair += ff;
          ecoul += ef;
          e += ef;
        }
        if (rsq < cutljsq[t]) {
          const factor = sLJ[sb];
          const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
          fpair += factor * r6inv * (lj1[t] * r6inv - lj2[t]) * r2inv;
          const ev = factor * (r6inv * (lj3[t] * r6inv - lj4[t]) - offset[t]);
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

  /**
   * The tail correction is for "the Lennard-Jones portion of the pair
   * interaction" (pair_lj_cut_coul.html), so it uses the LJ cutoff, not the
   * neighbor-list cutoff (max of LJ and Coulombic).
   */
  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    let e = 0, pr = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j), rc = this.p.get('cut', i, j);
        if (!(rc > 0)) continue;
        const s6 = sig ** 6, s12 = s6 * s6, rc3 = rc ** 3, rc9 = rc3 ** 3;
        const nn = count[i] * count[j];
        e += nn * 4 * eps * (s12 / (9 * rc9) - s6 / (3 * rc3));
        pr += nn * 4 * eps * (-4 * s12 / (3 * rc9) + 2 * s6 / rc3);
      }
    }
    // caller divides by V (energy) and V^2 (pressure); returns the V-free sums
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  // write_data, measured with native LAMMPS: "PairIJ Coeffs" lines are
  // "I J epsilon sigma cutoff1" (no Coulomb cutoff column), which is
  // PairLJCut.dataCoeffsIJ; lj/cut/coul/dsf writes no pair section at all.

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }
}

/*
 * docs.lammps.org/pair_lj_cut_coul.html:
 *   "lj/cut/coul/cut args = cutoff (cutoff2)"
 *   "cutoff = global cutoff for LJ (and Coulombic if only 1 arg) (distance units)"
 *   "cutoff2 = global cutoff for Coulombic (optional) (distance units)"
 *   "Style lj/cut/coul/cut adds a Coulombic pairwise interaction given by"
 *     E = \frac{C q_i q_j}{\epsilon  r} \qquad r < r_c
 * Force: F = C q_i q_j / r^2 is -dE/dr, applied as f_i += dx * F/r.
 */
export class PairLJCutCoulCut extends PairLJCutCoul {
  readonly name: string = 'lj/cut/coul/cut';

  settings(args: string[]): void {
    if (args.length !== 1 && args.length !== 2) throw new StyleError('usage: pair_style lj/cut/coul/cut cutoff (cutoff2)');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.cutCoul = args.length === 2 ? parseNum(args[1], 'cutoff2') : this.cutGlobal;
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
  }

  protected coulPair(_t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    const pre = (qqrd2e * qi * qj) / Math.sqrt(rsq);
    return { e: pre, f: pre / rsq };
  }
}

/*
 * docs.lammps.org/pair_lj_cut_coul.html:
 *   "lj/cut/coul/debye args = kappa cutoff (cutoff2)"
 *   "kappa = inverse of the Debye length (inverse distance units)"
 *   "Style lj/cut/coul/debye adds an additional exp() damping factor
 *   to the Coulombic term, given by"
 *     E = \frac{C q_i q_j}{\epsilon  r} \exp(- \kappa r) \qquad r < r_c
 * Force: -dE/dr = C q_i q_j exp(-kappa r) (1/r^2 + kappa/r), applied as
 * f_i += dx * F/r.
 */
export class PairLJCutCoulDebye extends PairLJCutCoul {
  readonly name: string = 'lj/cut/coul/debye';
  kappa = 0;

  settings(args: string[]): void {
    if (args.length !== 2 && args.length !== 3) throw new StyleError('usage: pair_style lj/cut/coul/debye kappa cutoff (cutoff2)');
    this.kappa = parseNum(args[0], 'kappa');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    this.cutCoul = args.length === 3 ? parseNum(args[2], 'cutoff2') : this.cutGlobal;
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
  }

  protected coulPair(_t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    const r = Math.sqrt(rsq);
    const pre = (qqrd2e * qi * qj * Math.exp(-this.kappa * r)) / r;
    return { e: pre, f: (pre * (1 + this.kappa * r)) / rsq };
  }
}

/*
 * docs.lammps.org/pair_lj_cut_coul.html:
 *   "lj/cut/coul/dsf args = alpha cutoff (cutoff2)"
 *   "alpha = damping parameter (inverse distance units)"
 *   "Style lj/cut/coul/dsf computes the Coulombic term via the damped
 *   shifted force model described in Fennell, given by:"
 *     E =
 *      q_iq_j \left[ \frac{\mbox{erfc} (\alpha r)}{r} -  \frac{\mbox{erfc} (\alpha r_c)}{r_c} +
 *     \left( \frac{\mbox{erfc} (\alpha r_c)}{r_c^2} +  \frac{2\alpha}{\sqrt{\pi}}\frac{\exp (-\alpha^2    r^2_c)}{r_c} \right)(r-r_c) \right] \qquad r < r_c
 * With A = erfc(alpha rc)/rc and B = erfc(alpha rc)/rc^2
 * + 2 alpha/sqrt(pi) exp(-alpha^2 rc^2)/rc (both per type pair, rc the
 * pair's Coulombic cutoff):
 *   E = C q_i q_j [ erfc(alpha r)/r - A + B (r - rc) ]
 *   F = C q_i q_j [ erfc(alpha r)/r^2 + 2 alpha/sqrt(pi) exp(-alpha^2 r^2)/r - B ]
 * (F is -dE/dr; the B term is the shifted force, which makes F(rc) = 0 —
 * pair_coul.html: the model provides "consistent forces and energies ...
 * and smooth decay to zero").
 */
export class PairLJCutCoulDsf extends PairLJCutCoul {
  readonly name: string = 'lj/cut/coul/dsf';
  protected coulCutPerPair = false;
  protected coulSpecialSubtract = true;
  keepExcluded = true;
  alpha = 0;
  private dsfA = new Float64Array(0);
  private dsfB = new Float64Array(0);
  private dsfRC = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 2 && args.length !== 3) throw new StyleError('usage: pair_style lj/cut/coul/dsf alpha cutoff (cutoff2)');
    this.alpha = parseNum(args[0], 'alpha');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    this.cutCoul = args.length === 3 ? parseNum(args[2], 'cutoff2') : this.cutGlobal;
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = (ntypes + 1) * (ntypes + 1);
    this.dsfA = new Float64Array(n);
    this.dsfB = new Float64Array(n);
    this.dsfRC = new Float64Array(n);
  }

  /**
   * Self energy, not given in pair_coul.html; measured with native LAMMPS
   * (a lone charge in a large box, coul/dsf and lj/cut/coul/dsf alike):
   *   E_self = -(erfc(a rc)/rc + a/sqrt(pi) (1 + exp(-a^2 rc^2))) q^2 C
   * (alpha 0.6, rc 3, q 1: -0.355407766583; alpha 0.3, rc 4: -0.231780032557).
   */
  dataCoeffs(): string[] | null { return null; }
  dataCoeffsIJ(): string[] | null { return null; }

  protected coulSelf(qi: number, qqrd2e: number): number {
    const arc = this.alpha * this.cutCoul;
    return -(erfcExact(arc) / this.cutCoul + (this.alpha / Math.sqrt(Math.PI)) * (1 + Math.exp(-arc * arc))) * qi * qi * qqrd2e;
  }

  protected initCoul(i: number, j: number): void {
    const nt = this.ntypes + 1;
    const k1 = i * nt + j, k2 = j * nt + i;
    const rc = this.p.get('cut_coul', i, j);
    const arc = this.alpha * rc;
    const erc = erfcExact(arc);
    this.dsfRC[k1] = this.dsfRC[k2] = rc;
    this.dsfA[k1] = this.dsfA[k2] = erc / rc;
    this.dsfB[k1] = this.dsfB[k2] = erc / (rc * rc) + (TWO_OVER_SQRTPI * this.alpha * Math.exp(-arc * arc)) / rc;
  }

  protected coulPair(t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    const r = Math.sqrt(rsq);
    const grij = this.alpha * r;
    const ex = Math.exp(-grij * grij);
    // measured with native LAMMPS (two charges at r = 1, 1.5, 2; 13 digits): erfc(alpha r) is the
    // Abramowitz-Stegun polynomial (erfcPoly), the shift constants use the exact erfc(alpha rc)
    const erfcc = erfcPoly(grij, ex);
    const pref = qqrd2e * qi * qj;
    const a = this.dsfA[t], b = this.dsfB[t];
    return {
      e: pref * (erfcc / r - a + b * (r - this.dsfRC[t])),
      f: pref * (erfcc / (rsq * r) + (TWO_OVER_SQRTPI * this.alpha * ex) / rsq - b / r),
    };
  }
}

/*
 * docs.lammps.org/pair_lj_cut_coul.html:
 *   "lj/cut/coul/wolf args = alpha cutoff (cutoff2)"
 *   "alpha = damping parameter (inverse distance units)"
 *   "Style lj/cut/coul/wolf adds a Coulombic pairwise interaction via the Wolf
 *   summation method, described in Wolf, given by:"
 *     E_i = \frac{1}{2} \sum_{j \neq i}
 *     \frac{q_i q_j \mathrm{erfc}(\alpha r_{ij})}{r_{ij}} +
 *     \frac{1}{2} \sum_{j \neq i}
 *     \frac{q_i q_j \mathrm{erf}(\alpha r_{ij})}{r_{ij}} \qquad r < r_c
 * The doc describes the potential as "a short-range, spherically-truncated,
 * charge-neutralized, shifted, pairwise *1/r* summation" where "charge
 * neutralization within the cutoff radius is enforced by shifting the
 * potential through placement of image charges on the cutoff sphere", i.e.
 * the per-pair term is
 *   E = C q_i q_j [ erfc(alpha r)/r - erfc(alpha rc)/rc ]   (rc per pair)
 * and the force is its derivative,
 *   F = C q_i q_j [ erfc(alpha r)/r^2 + 2 alpha/sqrt(pi) exp(-alpha^2 r^2)/r ].
 * Unlike the dsf model there is no force shift: pair_coul.html says the dsf
 * potential "corrects issues in the Wolf model ... to provide consistent
 * forces and energies (the Wolf potential is not differentiable at the
 * cutoff)".
 */
export class PairLJCutCoulWolf extends PairLJCutCoul {
  readonly name: string = 'lj/cut/coul/wolf';
  protected coulCutPerPair = false;
  protected coulSpecialSubtract = true;
  keepExcluded = true;
  alpha = 0;
  private wolfA = new Float64Array(0);
  private wolfF = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 2 && args.length !== 3) throw new StyleError('usage: pair_style lj/cut/coul/wolf alpha cutoff (cutoff2)');
    this.alpha = parseNum(args[0], 'alpha');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    this.cutCoul = args.length === 3 ? parseNum(args[2], 'cutoff2') : this.cutGlobal;
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.wolfA = new Float64Array((ntypes + 1) * (ntypes + 1));
    this.wolfF = new Float64Array((ntypes + 1) * (ntypes + 1));
  }

  /**
   * Self energy, not given in pair_coul.html; measured with native LAMMPS
   * (a lone charge, coul/wolf and lj/cut/coul/wolf alike):
   *   E_self = -(erfc(a rc)/(2 rc) + a/sqrt(pi)) q^2 C
   * (alpha 0.6, rc 3, q 1: -0.340331999856; alpha 0.3, rc 4: -0.180467627786).
   */
  protected coulSelf(qi: number, qqrd2e: number): number {
    return -(erfcExact(this.alpha * this.cutCoul) / (2 * this.cutCoul) + this.alpha / Math.sqrt(Math.PI)) * qi * qi * qqrd2e;
  }

  protected initCoul(i: number, j: number): void {
    const nt = this.ntypes + 1;
    const k1 = i * nt + j, k2 = j * nt + i;
    const rc = this.p.get('cut_coul', i, j);
    const a = erfcExact(this.alpha * rc) / rc;
    this.wolfA[k1] = this.wolfA[k2] = a;
    const arc = this.alpha * rc;
    this.wolfF[k1] = this.wolfF[k2] = -(a + TWO_OVER_SQRTPI * this.alpha * Math.exp(-arc * arc)) / rc;
  }

  protected coulPair(t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    const r = Math.sqrt(rsq);
    const grij = this.alpha * r;
    const ex = Math.exp(-grij * grij);
    const erfcc = erfcExact(grij);
    const pref = qqrd2e * qi * qj;
    return {
      e: pref * (erfcc / r - this.wolfA[t]),
      // measured with native LAMMPS (two charges at r = 1, 1.5, 2; 12 digits): the force carries the
      // damped-shifted-force constant f_shift = -(erfc(a rc)/rc + 2a/sqrt(pi) exp(-a^2 rc^2))/rc although
      // the energy is only potential-shifted ("the Wolf potential is not differentiable at the cutoff")
      f: pref * (erfcc / (rsq * r) + (TWO_OVER_SQRTPI * this.alpha * ex) / rsq + this.wolfF[t] / r),
    };
  }
}
