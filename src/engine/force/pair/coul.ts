import { Pair, PairParams, StyleError, mixDistance, type PairCompute } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { fmtCoeff, parseNum } from '../util';
import { erfcExact, erfcPoly } from '../erfc';

/*
 * pair_style coul/cut, coul/debye, coul/dsf and coul/wolf —
 * docs.lammps.org/pair_coul.html:
 *   pair_style coul/cut cutoff
 *   pair_style coul/debye kappa cutoff
 *   pair_style coul/dsf alpha cutoff
 *   pair_style coul/wolf alpha cutoff
 *   * cutoff = global cutoff for Coulombic interactions
 *   * kappa = Debye length (inverse distance units)
 *   * alpha = damping parameter (inverse distance units)
 * "The *coul/cut* style computes the standard Coulombic interaction
 * potential given by"
 *   E = \frac{C q_i q_j}{\epsilon  r} \qquad r < r_c
 * "where C is an energy-conversion constant, Qi and Qj are the charges on
 * the two atoms, and :math:`\epsilon` is the dielectric constant which can
 * be set by the :doc:`dielectric <dielectric>` command." (C / epsilon is
 * pc.qqrd2e.)
 * Coefficients: "* cutoff (distance units)" and "For *coul/cut* and
 * *coul/debye* the cutoff coefficient is optional.  If it is not used (as in
 * some of the examples above), the default global value specified in the
 * pair_style command is used."
 * Mixing: "For atom type pairs I,J and I != J, the cutoff distance for the
 * *coul/cut* style can be mixed.  The default mix value is *geometric*\ ."
 *
 * Measured with native LAMMPS (black box), where the page is silent:
 * - coul/debye mixes its cutoff the same way (pair_coeff 1 1 3.0 and 2 2 4.0:
 *   the 1-2 pair interacts at r = 3.4 and not at 3.6, i.e. sqrt(12));
 * - coul/dsf and coul/wolf take no pair_coeff values ("pair_coeff 1 1 3.0"
 *   is "Incorrect args for pair coefficients"): one global cutoff;
 * - each style gives the same energies and forces (15 digits, a charged
 *   4-atom chain, every special_bonds weight set probed) as its
 *   lj/cut/coul/* counterpart with epsilon = 0, so the dsf/wolf self
 *   energy, erfc evaluation, wolf force shift and special-pair rules below
 *   are those measured for lj_coul.ts.
 * "The pair_modify shift option is not relevant for these pair styles" and
 * "These pair styles do not support the pair_modify tail option".
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** Shared loop: a per-pair Coulomb cutoff 'cut' and a style-specific coulPair(). */
abstract class PairCoul extends Pair {
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  /** coul/cut and coul/debye take an optional per-pair cutoff; dsf/wolf none. */
  protected cutPerPair = true;
  /** dsf/wolf: a special pair keeps the damped term minus (1 - w) of the bare C q_i q_j / r. */
  protected coulSpecialSubtract = false;

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['cut']);
  }

  coeff(args: string[]): void {
    const max = this.cutPerPair ? 3 : 2;
    if (args.length < 2 || args.length > max) {
      throw new StyleError(`usage: pair_coeff I J${this.cutPerPair ? ' [cutoff]' : ''} (${this.name})`);
    }
    const cut = args[2] !== undefined ? parseNum(args[2], 'cutoff') : this.cutGlobal;
    if (!(cut > 0)) throw new StyleError('pair_coeff cutoff must be > 0');
    this.p.setRange(args[0], args[1], [cut]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      p.setMixed(i, j, 'cut', this.cutPerPair ? mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)) : this.cutGlobal);
    }
    this.initCoul(i, j);
    return p.get('cut', i, j);
  }

  /** Hook for per-pair shift constants (dsf/wolf). */
  protected initCoul(_i: number, _j: number): void {}

  /** Self energy of one owned charge (dsf/wolf), 0 otherwise. */
  protected coulSelf(_qi: number, _qqrd2e: number): number { return 0; }

  /** Unscaled energy e and force-over-r f of one pair. */
  protected abstract coulPair(t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number };

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const qi = q[i];
      if (qi === 0) continue;
      const eself = this.coulSelf(qi, pc.qqrd2e);
      if (eself !== 0) {
        ecoul += eself;
        if (pc.eatom) pc.eatom[i] += eself;
      }
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const qj = q[j];
        if (qj === 0) continue;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const fc = sC[jj >>> SBBITS];
        const c = this.coulPair(t, rsq, qi, qj, pc.qqrd2e);
        let e = fc * c.e, fpair = fc * c.f;
        if (fc !== 1 && this.coulSpecialSubtract) {
          const bare = pc.qqrd2e * qi * qj / Math.sqrt(rsq);
          e = c.e - (1 - fc) * bare;
          fpair = c.f - (1 - fc) * bare / rsq;
        }
        fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        ecoul += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.ecoul += ecoul;
  }

  /*
   * write_data, measured with native LAMMPS: coul/cut and coul/debye write
   * "Pair Coeffs" as bare type numbers and "PairIJ Coeffs" as "I J cutoff";
   * coul/dsf and coul/wolf write no pair coefficient section at all.
   */
  dataCoeffs(): string[] | null {
    if (!this.cutPerPair) return null;
    return Array.from({ length: this.ntypes }, (_, k) => `${k + 1}`);
  }

  dataCoeffsIJ(): string[] | null {
    if (!this.cutPerPair) return null;
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) out.push(`${i} ${j} ${fmtCoeff(this.p.get('cut', i, j))}`);
    }
    return out;
  }

  extract(name: string): unknown {
    return name === 'cut_coul' ? this.cutGlobal : undefined;
  }
}

export class PairCoulCut extends PairCoul {
  readonly name: string = 'coul/cut';

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style coul/cut cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  protected coulPair(_t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    const pre = (qqrd2e * qi * qj) / Math.sqrt(rsq);
    return { e: pre, f: pre / rsq };
  }
}

/*
 * "Style *coul/debye* adds an additional exp() damping factor to the
 * Coulombic term, given by"
 *   E = \frac{C q_i q_j}{\epsilon  r} \exp(- \kappa r) \qquad r < r_c
 * Force: -dE/dr = C q_i q_j exp(-kappa r) (1/r^2 + kappa/r).
 */
export class PairCoulDebye extends PairCoul {
  readonly name: string = 'coul/debye';
  kappa = 0;

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('usage: pair_style coul/debye kappa cutoff');
    this.kappa = parseNum(args[0], 'kappa');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  protected coulPair(_t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    const r = Math.sqrt(rsq);
    const pre = (qqrd2e * qi * qj * Math.exp(-this.kappa * r)) / r;
    return { e: pre, f: (pre * (1 + this.kappa * r)) / rsq };
  }
}

/*
 * "Style *coul/dsf* computes Coulombic interactions via the damped shifted
 * force model described in :ref:`Fennell <Fennell1>`, given by:"
 *   E = q_iq_j \left[ \frac{\mbox{erfc} (\alpha r)}{r} -  \frac{\mbox{erfc} (\alpha r_c)}{r_c} +
 *   \left( \frac{\mbox{erfc} (\alpha r_c)}{r_c^2} +  \frac{2\alpha}{\sqrt{\pi}}\frac{\exp (-\alpha^2    r^2_c)}{r_c} \right)(r-r_c) \right] \qquad r < r_c
 * Measured (see lj_coul.ts): self energy per charge
 * -(erfc(a rc)/rc + a/sqrt(pi) (1 + exp(-a^2 rc^2))) q^2 C, and erfc(alpha r)
 * is the polynomial approximation (erfcPoly) while the shift constants use
 * the exact erfc(alpha rc).
 */
export class PairCoulDsf extends PairCoul {
  readonly name: string = 'coul/dsf';
  protected cutPerPair = false;
  protected coulSpecialSubtract = true;
  keepExcluded = true;
  alpha = 0;
  private eShift = 0;
  private fShift = 0;

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('usage: pair_style coul/dsf alpha cutoff');
    this.alpha = parseNum(args[0], 'alpha');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  protected initCoul(): void {
    const rc = this.cutGlobal, arc = this.alpha * rc;
    const erc = erfcExact(arc);
    this.eShift = erc / rc;
    this.fShift = erc / (rc * rc) + (TWO_OVER_SQRTPI * this.alpha * Math.exp(-arc * arc)) / rc;
  }

  protected coulSelf(qi: number, qqrd2e: number): number {
    const arc = this.alpha * this.cutGlobal;
    return -(erfcExact(arc) / this.cutGlobal + (this.alpha / Math.sqrt(Math.PI)) * (1 + Math.exp(-arc * arc))) * qi * qi * qqrd2e;
  }

  protected coulPair(_t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    const r = Math.sqrt(rsq);
    const grij = this.alpha * r;
    const ex = Math.exp(-grij * grij);
    const erfcc = erfcPoly(grij, ex);
    const pref = qqrd2e * qi * qj;
    return {
      e: pref * (erfcc / r - this.eShift + this.fShift * (r - this.cutGlobal)),
      f: pref * (erfcc / (rsq * r) + (TWO_OVER_SQRTPI * this.alpha * ex) / rsq - this.fShift / r),
    };
  }
}

/*
 * "Style *coul/wolf* computes Coulombic interactions via the Wolf summation
 * method, described in :ref:`Wolf <Wolf1>`, given by:"
 *   E_i = \frac{1}{2} \sum_{j \neq i}
 *   \frac{q_i q_j \mathrm{erfc}(\alpha r_{ij})}{r_{ij}} +
 *   \frac{1}{2} \sum_{j \neq i}
 *   \frac{q_i q_j \mathrm{erf}(\alpha r_{ij})}{r_{ij}} \qquad r < r_c
 * "This potential is essentially a short-range, spherically-truncated,
 * charge-neutralized, shifted, pairwise *1/r* summation."
 * Measured (see lj_coul.ts): pair energy C q_i q_j [erfc(a r)/r - erfc(a rc)/rc],
 * self energy -(erfc(a rc)/(2 rc) + a/sqrt(pi)) q^2 C, and the force carries
 * the shift -(erfc(a rc)/rc + 2a/sqrt(pi) exp(-a^2 rc^2))/rc.
 */
export class PairCoulWolf extends PairCoul {
  readonly name: string = 'coul/wolf';
  protected cutPerPair = false;
  protected coulSpecialSubtract = true;
  keepExcluded = true;
  alpha = 0;
  private eShift = 0;
  private fShift = 0;

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('usage: pair_style coul/wolf alpha cutoff');
    this.alpha = parseNum(args[0], 'alpha');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  protected initCoul(): void {
    const rc = this.cutGlobal, arc = this.alpha * rc;
    this.eShift = erfcExact(arc) / rc;
    this.fShift = -(this.eShift + TWO_OVER_SQRTPI * this.alpha * Math.exp(-arc * arc)) / rc;
  }

  protected coulSelf(qi: number, qqrd2e: number): number {
    return -(erfcExact(this.alpha * this.cutGlobal) / (2 * this.cutGlobal) + this.alpha / Math.sqrt(Math.PI)) * qi * qi * qqrd2e;
  }

  protected coulPair(_t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    const r = Math.sqrt(rsq);
    const grij = this.alpha * r;
    const ex = Math.exp(-grij * grij);
    const erfcc = erfcExact(grij);
    const pref = qqrd2e * qi * qj;
    return {
      e: pref * (erfcc / r - this.eShift),
      f: pref * (erfcc / (rsq * r) + (TWO_OVER_SQRTPI * this.alpha * ex) / rsq + this.fShift / r),
    };
  }
}
