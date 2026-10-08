import { StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { PairBorn } from './simple';
import { tallyAtom } from './lj_cut';
import { fmtCoeff, parseNum } from '../util';
import { erfcExact } from '../erfc';

/*
 * pair_style born/coul/wolf and pair_style born/coul/dsf —
 * docs.lammps.org/pair_born.html (docs.lammps.org/pair_born.rst).
 *
 * The Born-Mayer-Huggins part is PairBorn (force/pair/simple.ts):
 *   E = A \exp \left(\frac{\sigma - r}{\rho} \right) -
 *   \frac{C}{r^6} + \frac{D}{r^8} \qquad r < r_c
 * pair_coeff coefficients (pair_born.html): A (energy units),
 * rho (distance units), sigma (distance units), C (energy units *
 * distance units^6), D (energy units * distance units^8), cutoff (distance
 * units). "The second coefficient, rho, must be greater than zero." "The
 * last coefficient is optional.  If not specified, the global A,C,D cutoff
 * specified in the pair_style command is used." "These pair styles do not
 * support mixing.  Thus, coefficients for all I,J pairs must be specified
 * explicitly." (inherited from PairBorn).
 *
 * pair_style args (pair_born.html):
 *   born/coul/wolf args = alpha cutoff (cutoff2)
 *     alpha = damping parameter (inverse distance units)
 *     cutoff = global cutoff for non-Coulombic (and Coulombic if only 1 arg)
 *     cutoff2 = global cutoff for Coulombic (optional)
 *   born/coul/dsf args = alpha cutoff (cutoff2)
 *     (same argument meaning as born/coul/wolf)
 * "For born/coul/long, born/coul/wolf and born/coul/dsf no Coulombic cutoff
 * can be specified for an individual I,J type pair.  All type pairs use the
 * same global Coulombic cutoff specified in the pair_style command."
 *
 * "The born/coul/wolf style adds a Coulombic term as described for the Wolf
 * potential in the coul/wolf pair style." "The born/coul/dsf style computes
 * the Coulomb contribution with the damped shifted force model as in the
 * coul/dsf style." Both are the same per-pair Coulomb kernel as
 * lj/cut/coul/wolf and lj/cut/coul/dsf (force/pair/lj_coul.ts), copied here
 * because that file's kernels are private to the PairLJCut base:
 *   Wolf: E = C q_i q_j [ erfc(alpha r)/r - erfc(alpha rc)/rc ]
 *     (pair_coul.html: "This potential is essentially a short-range,
 *      spherically-truncated, charge-neutralized, shifted, pairwise *1/r*
 *      summation.")
 *   DSF:  E = C q_i q_j [ erfc(alpha r)/r - erfc(alpha rc)/rc
 *          + ( erfc(alpha rc)/rc^2
 *              + 2 alpha/sqrt(pi) exp(-alpha^2 rc^2)/rc ) (r - rc) ]
 *     (pair_coul.html: "The potential corrects issues in the Wolf model ...
 *      to provide consistent forces and energies (the Wolf potential is not
 *      differentiable at the cutoff) and smooth decay to zero.")
 * The Coulomb self energy and the special_bonds treatment were measured with
 * native LAMMPS for the lj/cut/coul variants (see lj_coul.ts) and re-measured
 * for the born variants: a lone unit charge in a large box gives wolf
 * E_self = -(erfc(a rc)/(2 rc) + a/sqrt(pi)) q^2 C (-0.340331999856 for
 * alpha 0.6, rc 3) and dsf
 * E_self = -(erfc(a rc)/rc + a/sqrt(pi)(1 + exp(-a^2 rc^2))) q^2 C
 * (-0.355407766583 for alpha 0.6, rc 3); a weight-0.0 special pair keeps the
 * full damped term minus (1 - w) times the bare C q_i q_j / r (measured with
 * native LAMMPS on a bonded two-atom system, thermo_modify norm no), so
 * keepExcluded is set (special_bonds.html: a 0.0 weight excludes the pair
 * "except for ... pair styles that include "coul/dsf" or "coul/wolf"").
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** Shared born/coul/wolf and born/coul/dsf body: PairBorn + a global Coulomb cutoff and a coulPair() kernel. */
abstract class PairBornCoul18 extends PairBorn {
  /** Global Coulombic cutoff from pair_style; born/coul/wolf/dsf have no per-pair Coulomb cutoff. */
  cutCoul = 0;
  protected alpha = 0;
  keepExcluded = true;
  private bornCutSq = new Float64Array(0);
  private coulCutSq = new Float64Array(0);

  initOne(i: number, j: number): number {
    const bornRC = super.initOne(i, j);
    const nt = this.ntypes + 1;
    if (this.bornCutSq.length !== nt * nt) {
      this.bornCutSq = new Float64Array(nt * nt);
      this.coulCutSq = new Float64Array(nt * nt);
    }
    const k1 = i * nt + j, k2 = j * nt + i;
    this.bornCutSq[k1] = this.bornCutSq[k2] = bornRC * bornRC;
    this.coulCutSq[k1] = this.coulCutSq[k2] = this.cutCoul * this.cutCoul;
    this.initCoul(i, j);
    // "If two cutoffs are specified, the first is used as the cutoff for the
    // A,C,D terms, and the second is the cutoff for the Coulombic term."
    return Math.max(bornRC, this.cutCoul);
  }

  /** Per-pair Coulomb shift constants (dsf/wolf). */
  protected initCoul(_i: number, _j: number): void {}

  /** Coulomb self energy of one owned atom, 0 when the style has none. */
  protected coulSelf(_qi: number, _qqrd2e: number): number { return 0; }

  /** Unscaled Coulomb energy e and force-over-r f of one pair. */
  protected abstract coulPair(t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number };

  compute(pc: PairCompute): void {
    this.qqrd2eSingle = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { bornA, bornIR, bornSig, bornC, bornD, offset, bornCutSq, coulCutSq } = this;
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
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        let fpair = 0, e = 0;
        const qj = q[j];
        // Born-Mayer-Huggins A,C,D term, cut at the per-pair A,C,D cutoff.
        if (rsq < bornCutSq[t]) {
          const factor = sLJ[sb];
          const r = Math.sqrt(rsq);
          const rinv = 1 / r;
          const ex = Math.exp((bornSig[t] - r) * bornIR[t]);
          const r2inv = 1 / rsq;
          const r6inv = r2inv * r2inv * r2inv;
          const r8inv = r6inv * r2inv;
          // F_vec = [(A/rho) e^{(sigma-r)/rho} - 6 C/r^7 + 8 D/r^9] * rhat
          //       = fpair * (ri - rj), fpair = (A/rho) e/r - 6 C/r^8 + 8 D/r^10
          fpair += factor * (bornA[t] * bornIR[t] * ex * rinv - 6 * bornC[t] * r8inv + 8 * bornD[t] * r8inv * r2inv);
          const ev = factor * (bornA[t] * ex - bornC[t] * r6inv + bornD[t] * r8inv - offset[t]);
          evdwl += ev;
          e += ev;
        }
        // Coulomb term at the global Coulombic cutoff.
        if (rsq < coulCutSq[t] && qi !== 0 && qj !== 0) {
          const fc = sC[sb];
          const c = this.coulPair(t, rsq, qi, qj, pc.qqrd2e);
          let ef = fc * c.e, ff = fc * c.f;
          // measured with native LAMMPS: a special pair (weight w) keeps the
          // full damped term minus (1 - w) times the bare C q_i q_j / r.
          if (fc !== 1) {
            const bare = pc.qqrd2e * qi * qj / Math.sqrt(rsq);
            ef = c.e - (1 - fc) * bare;
            ff = c.f - (1 - fc) * bare / rsq;
          }
          fpair += ff;
          ecoul += ef;
          e += ef;
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
   * Tail correction for the A,C,D part only (pair_born.html: "These styles
   * support the pair_modify tail option for adding long-range tail
   * corrections to energy and pressure."; the Coulomb term is not a plain
   * 1/r tail). Uses the per-pair A,C,D cutoff, not the neighbour-list cutoff.
   */
  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    let e = 0, pr = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const a = this.p.get('A', i, j), rho = this.p.get('rho', i, j), sig = this.p.get('sigma', i, j);
        const c = this.p.get('C', i, j), d = this.p.get('D', i, j);
        const cut = this.p.get('cut', i, j);
        const rc = Number.isNaN(cut) ? this.cutGlobal : cut;
        if (!(rc > 0)) continue;
        const nn = count[i] * count[j], ex = Math.exp((sig - rc) / rho);
        e += nn * (a * rho * ex * (rc * rc + 2 * rho * rc + 2 * rho * rho) - c / (3 * rc ** 3) + d / (5 * rc ** 5));
        pr += nn * (-(a / rho) * ex * (rc ** 3 * rho + 3 * rc * rc * rho * rho + 6 * rc * rho ** 3 + 6 * rho ** 4) + 2 * c / rc ** 3 - 8 * d / (5 * rc ** 5));
      }
    }
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    let eng = 0, fforce = 0;
    if (rsq < this.bornCutSq[t]) {
      const r = Math.sqrt(rsq);
      const rinv = 1 / r;
      const ex = Math.exp((this.bornSig[t] - r) * this.bornIR[t]);
      const r2inv = 1 / rsq;
      const r6inv = r2inv * r2inv * r2inv;
      const r8inv = r6inv * r2inv;
      fforce += factorLJ * (this.bornA[t] * this.bornIR[t] * ex * rinv - 6 * this.bornC[t] * r8inv + 8 * this.bornD[t] * r8inv * r2inv);
      eng += factorLJ * (this.bornA[t] * ex - this.bornC[t] * r6inv + this.bornD[t] * r8inv - this.offset[t]);
    }
    if (rsq < this.coulCutSq[t] && qi !== 0 && qj !== 0) {
      const c = this.coulPair(t, rsq, qi, qj, this.qqrd2eSingle);
      let ef = factorCoul * c.e, ff = factorCoul * c.f;
      if (factorCoul !== 1) {
        const bare = this.qqrd2eSingle * qi * qj / Math.sqrt(rsq);
        ef = c.e - (1 - factorCoul) * bare;
        ff = c.f - (1 - factorCoul) * bare / rsq;
      }
      fforce += ff;
      eng += ef;
    }
    return { eng, fforce };
  }

  /** qqrd2e of the current run, set by compute(); single() has no PairCompute. */
  qqrd2eSingle = 1;

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const cut = this.p.get('cut', i, j);
        const rc = Number.isNaN(cut) ? this.cutGlobal : cut;
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('A', i, j))} ${fmtCoeff(this.p.get('rho', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))} ${fmtCoeff(this.p.get('C', i, j))} ${fmtCoeff(this.p.get('D', i, j))} ${fmtCoeff(rc)}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }
}

/*
 * pair_style born/coul/wolf — Wolf summation Coulomb added to the Born term
 * (pair_born.html; pair_coul.html formula quoted in the file header).
 */
export class PairBornCoulWolf extends PairBornCoul18 {
  readonly name: string = 'born/coul/wolf';
  private wolfA = new Float64Array(0);
  private wolfF = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 2 && args.length !== 3) throw new StyleError('usage: pair_style born/coul/wolf alpha cutoff (cutoff2)');
    this.alpha = parseNum(args[0], 'alpha');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    this.cutCoul = args.length === 3 ? parseNum(args[2], 'cutoff2') : this.cutGlobal;
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = (ntypes + 1) * (ntypes + 1);
    this.wolfA = new Float64Array(n);
    this.wolfF = new Float64Array(n);
  }

  protected initCoul(i: number, j: number): void {
    const nt = this.ntypes + 1;
    const k1 = i * nt + j, k2 = j * nt + i;
    const rc = this.cutCoul;
    const arc = this.alpha * rc;
    const a = erfcExact(arc) / rc;
    this.wolfA[k1] = this.wolfA[k2] = a;
    this.wolfF[k1] = this.wolfF[k2] = -(a + TWO_OVER_SQRTPI * this.alpha * Math.exp(-arc * arc)) / rc;
  }

  protected coulSelf(qi: number, qqrd2e: number): number {
    return -(erfcExact(this.alpha * this.cutCoul) / (2 * this.cutCoul) + this.alpha / Math.sqrt(Math.PI)) * qi * qi * qqrd2e;
  }

  protected coulPair(t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    const r = Math.sqrt(rsq);
    const grij = this.alpha * r;
    const ex = Math.exp(-grij * grij);
    const erfcc = erfcExact(grij);
    const pref = qqrd2e * qi * qj;
    return {
      e: pref * (erfcc / r - this.wolfA[t]),
      f: pref * (erfcc / (rsq * r) + (TWO_OVER_SQRTPI * this.alpha * ex) / rsq + this.wolfF[t] / r),
    };
  }
}

/*
 * pair_style born/coul/dsf — damped shifted force Coulomb added to the Born
 * term (pair_born.html; pair_coul.html formula quoted in the file header).
 */
export class PairBornCoulDsf extends PairBornCoul18 {
  readonly name: string = 'born/coul/dsf';
  private dsfA = new Float64Array(0);
  private dsfB = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 2 && args.length !== 3) throw new StyleError('usage: pair_style born/coul/dsf alpha cutoff (cutoff2)');
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
  }

  protected initCoul(i: number, j: number): void {
    const nt = this.ntypes + 1;
    const k1 = i * nt + j, k2 = j * nt + i;
    const rc = this.cutCoul;
    const arc = this.alpha * rc;
    const erc = erfcExact(arc);
    this.dsfA[k1] = this.dsfA[k2] = erc / rc;
    this.dsfB[k1] = this.dsfB[k2] = erc / (rc * rc) + (TWO_OVER_SQRTPI * this.alpha * Math.exp(-arc * arc)) / rc;
  }

  protected coulSelf(qi: number, qqrd2e: number): number {
    const arc = this.alpha * this.cutCoul;
    return -(erfcExact(arc) / this.cutCoul + (this.alpha / Math.sqrt(Math.PI)) * (1 + Math.exp(-arc * arc))) * qi * qi * qqrd2e;
  }

  protected coulPair(t: number, rsq: number, qi: number, qj: number, qqrd2e: number): { e: number; f: number } {
    const r = Math.sqrt(rsq);
    const grij = this.alpha * r;
    const ex = Math.exp(-grij * grij);
    // Measured with native LAMMPS: born/coul/dsf evaluates erfc(alpha r) exactly,
    // not with the Abramowitz-Stegun polynomial erfcPoly that lj/cut/coul/dsf uses
    // (the two natives differ by ~1.9e-8 in ecoul on the same 108-atom test system;
    // the exact form matches born/coul/dsf, erfcPoly matches lj/cut/coul/dsf).
    const erfcc = erfcExact(grij);
    const pref = qqrd2e * qi * qj;
    const a = this.dsfA[t], b = this.dsfB[t];
    return {
      e: pref * (erfcc / r - a + b * (r - this.cutCoul)),
      f: pref * (erfcc / (rsq * r) + (TWO_OVER_SQRTPI * this.alpha * ex) / rsq - b / r),
    };
  }
}
