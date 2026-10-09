import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { fmtCoeff, parseNum } from '../util';

/*
 * Simple two-body pair styles (Buckingham, Born-Mayer-Huggins, Morse),
 * written from the LAMMPS documentation only. All four styles share the
 * loop shape of PairLJCut (half neighbor list, special weights, per-atom
 * tallies, virial via the x.f dot product).
 *
 * None of these styles mixes (docs.lammps.org/pair_buck.html: "These pair
 * styles do not support mixing. Thus, coefficients for all I,J pairs must
 * be specified explicitly."; the same sentences appear on pair_born.html
 * and pair_morse.html), so an unset I,J pair is a StyleError.
 */

/*
 * pair_style buck — docs.lammps.org/pair_buck.html:
 *   E = A e^{-r / \rho} - \frac{C}{r^6} \qquad r < r_c
 * "The following coefficients must be defined for each pair of atoms types
 * via the pair_coeff command":
 *   * A (energy units)
 *   * :math:`\rho` (distance units)
 *   * C (energy-distance^6 units)
 *   * cutoff (distance units)
 *   * cutoff2 (distance units)
 * "The second coefficient, rho, must be greater than zero." "The latter 2
 * coefficients are optional. If not specified, the global A,C and Coulombic
 * cutoffs are used. ... You cannot specify 2 cutoffs for style buck, since
 * it has no Coulombic terms."
 * "These styles support the pair_modify shift option for the energy of the
 * exp() and 1/r^6 portion of the pair interaction." "These styles support
 * the pair_modify tail option for adding long-range tail corrections to
 * energy and pressure for the A,C terms in the pair interaction." The tail
 * integrals follow the Sun-form given on docs.lammps.org/pair_lj.html:
 * etail = 2 pi/V sum_ij N_i N_j int_rc^inf u(r) r^2 dr and
 * ptail = -2 pi/(3 V^2) sum_ij N_i N_j int_rc^inf r^3 u'(r) dr, evaluated
 * in closed form for the exp/6 potential.
 */
export class PairBuck extends Pair {
  readonly name: string = 'buck';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  buckA = new Float64Array(0);
  buckIR = new Float64Array(0);
  buckC = new Float64Array(0);
  offset = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style buck cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['A', 'rho', 'C', 'cut']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 5 || args.length > 6) throw new StyleError('usage: pair_coeff I J A rho C [cutoff]');
    const a = parseNum(args[2], 'A');
    const rho = parseNum(args[3], 'rho');
    if (!(rho > 0)) throw new StyleError(`Buckingham coefficient rho must be > 0 (got '${args[3]}')`);
    const c = parseNum(args[4], 'C');
    const cut = args[5] !== undefined ? parseNum(args[5], 'cutoff') : Number.NaN;
    this.p.setRange(args[0], args[1], [a, rho, c, cut]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.ntypes + 1;
    if (this.buckA.length !== nt * nt) {
      this.buckA = new Float64Array(nt * nt);
      this.buckIR = new Float64Array(nt * nt);
      this.buckC = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
    const a = p.get('A', i, j), rho = p.get('rho', i, j), c = p.get('C', i, j);
    const cut = p.get('cut', i, j);
    const rc = Number.isNaN(cut) ? this.cutGlobal : cut;
    const k1 = i * nt + j, k2 = j * nt + i;
    this.buckA[k1] = this.buckA[k2] = a;
    this.buckIR[k1] = this.buckIR[k2] = 1 / rho;
    this.buckC[k1] = this.buckC[k2] = c;
    this.offset[k1] = this.offset[k2] = this.shift && rc > 0 ? a * Math.exp(-rc / rho) - c / rc ** 6 : 0;
    return rc;
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    let e = 0, pr = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const a = this.p.get('A', i, j), rho = this.p.get('rho', i, j), c = this.p.get('C', i, j);
        const rc = this.cut[i * nt + j];
        if (!(rc > 0)) continue;
        const nn = count[i] * count[j], ex = Math.exp(-rc / rho);
        // int_rc^inf u(r) r^2 dr = A rho e^{-rc/rho} (rc^2 + 2 rho rc + 2 rho^2) - C/(3 rc^3)
        e += nn * (a * rho * ex * (rc * rc + 2 * rho * rc + 2 * rho * rho) - c / (3 * rc ** 3));
        // int_rc^inf r^3 u'(r) dr = -(A/rho) e^{-rc/rho} (rc^3 rho + 3 rc^2 rho^2 + 6 rc rho^3 + 6 rho^4) + 2 C/rc^3
        pr += nn * (-(a / rho) * ex * (rc ** 3 * rho + 3 * rc * rc * rho * rho + 6 * rc * rho ** 3 + 6 * rho ** 4) + 2 * c / rc ** 3);
      }
    }
    // caller divides by V (energy) and V^2 (pressure)
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, buckA, buckIR, buckC, offset } = this;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const factor = sLJ[jj >>> SBBITS];
        const r = Math.sqrt(rsq);
        const ex = Math.exp(-r * buckIR[t]);
        const r2inv = 1 / rsq;
        // F_vec = [(A/rho) e^{-r/rho} - 6 C/r^7] * rhat = fpair * (ri - rj)
        const fpair = factor * (buckA[t] * buckIR[t] * ex / r - 6 * buckC[t] * r2inv * r2inv * r2inv * r2inv);
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * (buckA[t] * ex - buckC[t] * (r2inv * r2inv * r2inv) - offset[t]);
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    const r = Math.sqrt(rsq);
    const ex = Math.exp(-r * this.buckIR[t]);
    const r2inv = 1 / rsq;
    return {
      fforce: factorLJ * (this.buckA[t] * this.buckIR[t] * ex / r - 6 * this.buckC[t] * r2inv * r2inv * r2inv * r2inv),
      eng: factorLJ * (this.buckA[t] * ex - this.buckC[t] * (r2inv * r2inv * r2inv) - this.offset[t]),
    };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('A', i, i))} ${fmtCoeff(this.p.get('rho', i, i))} ${fmtCoeff(this.p.get('C', i, i))}`);
    }
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    const nt = this.ntypes + 1;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('A', i, j))} ${fmtCoeff(this.p.get('rho', i, j))} ${fmtCoeff(this.p.get('C', i, j))} ${fmtCoeff(this.cut[i * nt + j])}`);
      }
    }
    return out;
  }
}

/*
 * pair_style buck/coul/cut — docs.lammps.org/pair_buck.html:
 *   buck/coul/cut args = cutoff (cutoff2)
 *     cutoff = global cutoff for Buckingham (and Coulombic if only 1 arg) (distance units)
 *     cutoff2 = global cutoff for Coulombic (optional) (distance units)
 * "The styles with coul/cut or coul/long or coul/msm add a Coulombic term
 * as described for the lj/cut pair styles." That term is documented on
 * docs.lammps.org/pair_lj_cut_coul.html:
 *   E = \frac{C q_i q_j}{\epsilon  r} \qquad r < r_c
 * "where :math:`C` is an energy-conversion constant, :math:`q_i` and :math:`q_j`
 * are the charges on the two atoms, and :math:`\epsilon` is the dielectric
 * constant" (C = qqr2e/dielectric,
 * passed in as pc.qqrd2e). "If one cutoff is specified for the born/coul/cut
 * and born/coul/long and born/coul/msm styles, it is used for both the A,C
 * and Coulombic terms. If two cutoffs are specified, the first is used as the
 * cutoff for the A,C terms, and the second is the cutoff for the Coulombic
 * term." (the sentence names the born variants; for buck/coul/cut the
 * equivalent rule is: "If only one cutoff is specified, it is used as the
 * cutoff for both A,C and Coulombic interactions for this type pair. If both
 * coefficients are specified, they are used as the A,C and Coulombic cutoffs
 * for this type pair.") "For all these pair styles, the terms with A and C
 * are always cutoff." Shift applies only to "the energy of the exp() and
 * 1/r^6 portion of the pair interaction", never to the Coulomb term.
 */
export class PairBuckCoulCut extends PairBuck {
  readonly name: string = 'buck/coul/cut';
  cutCoul = 0;
  /** Coulomb prefactor of the current run (pc.qqrd2e); single() uses the last-seen value. */
  qqrd2e = 1;
  private buckCutSq = new Float64Array(0);
  private coulCut = new Float64Array(0);
  private coulCutSq = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1 && args.length !== 2) throw new StyleError('usage: pair_style buck/coul/cut cutoff (cutoff2)');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.cutCoul = args.length === 2 ? parseNum(args[1], 'Coulomb cutoff') : this.cutGlobal;
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['A', 'rho', 'C', 'cut', 'cut2']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 5 || args.length > 7) throw new StyleError('usage: pair_coeff I J A rho C [cutoff [cutoff2]]');
    const a = parseNum(args[2], 'A');
    const rho = parseNum(args[3], 'rho');
    if (!(rho > 0)) throw new StyleError(`Buckingham coefficient rho must be > 0 (got '${args[3]}')`);
    const c = parseNum(args[4], 'C');
    const cut = args[5] !== undefined ? parseNum(args[5], 'cutoff') : Number.NaN;
    const cut2 = args[6] !== undefined ? parseNum(args[6], 'Coulomb cutoff') : Number.NaN;
    this.p.setRange(args[0], args[1], [a, rho, c, cut, cut2]);
  }

  initOne(i: number, j: number): number {
    const cutAC = super.initOne(i, j);
    const nt = this.ntypes + 1;
    if (this.buckCutSq.length !== nt * nt) {
      this.buckCutSq = new Float64Array(nt * nt);
      this.coulCut = new Float64Array(nt * nt);
      this.coulCutSq = new Float64Array(nt * nt);
    }
    const c1 = this.p.get('cut', i, j), c2 = this.p.get('cut2', i, j);
    const cutCoul = Number.isNaN(c2) ? (Number.isNaN(c1) ? this.cutCoul : c1) : c2;
    const k1 = i * nt + j, k2 = j * nt + i;
    this.buckCutSq[k1] = this.buckCutSq[k2] = cutAC * cutAC;
    this.coulCut[k1] = this.coulCut[k2] = cutCoul;
    this.coulCutSq[k1] = this.coulCutSq[k2] = cutCoul * cutCoul;
    return Math.max(cutAC, cutCoul);
  }

  compute(pc: PairCompute): void {
    this.qqrd2e = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { buckA, buckIR, buckC, offset, buckCutSq, coulCutSq } = this;
    const qqrd2e = pc.qqrd2e;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
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
        if (rsq < buckCutSq[t]) {
          const factor = sLJ[sb];
          const r = Math.sqrt(rsq);
          const ex = Math.exp(-r * buckIR[t]);
          const r2inv = 1 / rsq;
          fpair += factor * (buckA[t] * buckIR[t] * ex / r - 6 * buckC[t] * r2inv * r2inv * r2inv * r2inv);
          const ev = factor * (buckA[t] * ex - buckC[t] * (r2inv * r2inv * r2inv) - offset[t]);
          evdwl += ev;
          e += ev;
        }
        if (rsq < coulCutSq[t] && qi !== 0) {
          const qj = q[j];
          if (qj !== 0) {
            const factorC = sC[sb];
            const rinv = 1 / Math.sqrt(rsq);
            const pre = qqrd2e * qi * qj * rinv;
            fpair += factorC * pre / rsq;
            const ec = factorC * pre;
            ecoul += ec;
            e += ec;
          }
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

  single(i: number, j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number) {
    const b = super.single(i, j, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
    const t = itype * (this.ntypes + 1) + jtype;
    let eng = b.eng, fforce = b.fforce;
    if (rsq < this.coulCutSq[t] && qi !== 0 && qj !== 0) {
      const pre = this.qqrd2e * qi * qj / Math.sqrt(rsq);
      fforce += factorCoul * pre / rsq;
      eng += factorCoul * pre;
    }
    return { eng, fforce };
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    const nt = this.ntypes + 1;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('A', i, j))} ${fmtCoeff(this.p.get('rho', i, j))} ${fmtCoeff(this.p.get('C', i, j))} ${fmtCoeff(Math.sqrt(this.buckCutSq[i * nt + j]))} ${fmtCoeff(this.coulCut[i * nt + j])}`);
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
 * pair_style born — docs.lammps.org/pair_born.html:
 *   E = A \exp \left(\frac{\sigma - r}{\rho} \right) -
 *   \frac{C}{r^6} + \frac{D}{r^8} \qquad r < r_c
 * "The following coefficients must be defined for each pair of atoms types
 * via the pair_coeff command":
 *   * A (energy units)
 *   * :math:`\rho` (distance units)
 *   * :math:`\sigma` (distance units)
 *   * C (energy units * distance units^6)
 *   * D (energy units * distance units^8)
 *   * cutoff (distance units)
 * "The second coefficient, rho, must be greater than zero." "The last
 * coefficient is optional. If not specified, the global A,C,D cutoff
 * specified in the pair_style command is used."
 * "These pair styles do not support mixing." "These styles support the
 * pair_modify shift option for the energy of the exp(), 1/r^6, and 1/r^8
 * portion of the pair interaction." "These styles support the pair_modify
 * tail option for adding long-range tail corrections to energy and
 * pressure." (same Sun-form tail integrals as buck, plus the 1/r^8 term)
 */
export class PairBorn extends Pair {
  readonly name: string = 'born';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  bornA = new Float64Array(0);
  bornIR = new Float64Array(0);
  bornSig = new Float64Array(0);
  bornC = new Float64Array(0);
  bornD = new Float64Array(0);
  offset = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style born cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['A', 'rho', 'sigma', 'C', 'D', 'cut']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 7 || args.length > 8) throw new StyleError('usage: pair_coeff I J A rho sigma C D [cutoff]');
    const a = parseNum(args[2], 'A');
    const rho = parseNum(args[3], 'rho');
    if (!(rho > 0)) throw new StyleError(`Born-Mayer-Huggins coefficient rho must be > 0 (got '${args[3]}')`);
    const sig = parseNum(args[4], 'sigma');
    const c = parseNum(args[5], 'C');
    const d = parseNum(args[6], 'D');
    const cut = args[7] !== undefined ? parseNum(args[7], 'cutoff') : Number.NaN;
    this.p.setRange(args[0], args[1], [a, rho, sig, c, d, cut]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.ntypes + 1;
    if (this.bornA.length !== nt * nt) {
      this.bornA = new Float64Array(nt * nt);
      this.bornIR = new Float64Array(nt * nt);
      this.bornSig = new Float64Array(nt * nt);
      this.bornC = new Float64Array(nt * nt);
      this.bornD = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
    const a = p.get('A', i, j), rho = p.get('rho', i, j), sig = p.get('sigma', i, j);
    const c = p.get('C', i, j), d = p.get('D', i, j);
    const cut = p.get('cut', i, j);
    const rc = Number.isNaN(cut) ? this.cutGlobal : cut;
    const k1 = i * nt + j, k2 = j * nt + i;
    this.bornA[k1] = this.bornA[k2] = a;
    this.bornIR[k1] = this.bornIR[k2] = 1 / rho;
    this.bornSig[k1] = this.bornSig[k2] = sig;
    this.bornC[k1] = this.bornC[k2] = c;
    this.bornD[k1] = this.bornD[k2] = d;
    this.offset[k1] = this.offset[k2] = this.shift && rc > 0
      ? a * Math.exp((sig - rc) / rho) - c / rc ** 6 + d / rc ** 8
      : 0;
    return rc;
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    let e = 0, pr = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const a = this.p.get('A', i, j), rho = this.p.get('rho', i, j), sig = this.p.get('sigma', i, j);
        const c = this.p.get('C', i, j), d = this.p.get('D', i, j);
        const rc = this.cut[i * nt + j];
        if (!(rc > 0)) continue;
        const nn = count[i] * count[j], ex = Math.exp((sig - rc) / rho);
        // + D/(5 rc^5) and -8 D/(5 rc^5) from the 1/r^8 term
        e += nn * (a * rho * ex * (rc * rc + 2 * rho * rc + 2 * rho * rho) - c / (3 * rc ** 3) + d / (5 * rc ** 5));
        pr += nn * (-(a / rho) * ex * (rc ** 3 * rho + 3 * rc * rc * rho * rho + 6 * rc * rho ** 3 + 6 * rho ** 4) + 2 * c / rc ** 3 - 8 * d / (5 * rc ** 5));
      }
    }
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, bornA, bornIR, bornSig, bornC, bornD, offset } = this;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const factor = sLJ[jj >>> SBBITS];
        const r = Math.sqrt(rsq);
        const rinv = 1 / r;
        const ex = Math.exp((bornSig[t] - r) * bornIR[t]);
        const r2inv = 1 / rsq;
        const r6inv = r2inv * r2inv * r2inv;
        const r8inv = r6inv * r2inv;
        // F_vec = [(A/rho) e^{(sigma-r)/rho} - 6 C/r^7 + 8 D/r^9] * rhat
        //       = fpair * (ri - rj) with fpair = (A/rho) e/r - 6 C/r^8 + 8 D/r^10
        const fpair = factor * (bornA[t] * bornIR[t] * ex * rinv - 6 * bornC[t] * r8inv + 8 * bornD[t] * r8inv * r2inv);
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * (bornA[t] * ex - bornC[t] * r6inv + bornD[t] * r8inv - offset[t]);
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    const r = Math.sqrt(rsq);
    const rinv = 1 / r;
    const ex = Math.exp((this.bornSig[t] - r) * this.bornIR[t]);
    const r2inv = 1 / rsq;
    const r6inv = r2inv * r2inv * r2inv;
    const r8inv = r6inv * r2inv;
    return {
      fforce: factorLJ * (this.bornA[t] * this.bornIR[t] * ex * rinv - 6 * this.bornC[t] * r8inv + 8 * this.bornD[t] * r8inv * r2inv),
      eng: factorLJ * (this.bornA[t] * ex - this.bornC[t] * r6inv + this.bornD[t] * r8inv - this.offset[t]),
    };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('A', i, i))} ${fmtCoeff(this.p.get('rho', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))} ${fmtCoeff(this.p.get('C', i, i))} ${fmtCoeff(this.p.get('D', i, i))}`);
    }
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    const nt = this.ntypes + 1;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('A', i, j))} ${fmtCoeff(this.p.get('rho', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))} ${fmtCoeff(this.p.get('C', i, j))} ${fmtCoeff(this.p.get('D', i, j))} ${fmtCoeff(this.cut[i * nt + j])}`);
      }
    }
    return out;
  }
}

/*
 * pair_style morse — docs.lammps.org/pair_morse.html:
 *   E = D_0 \left[ e^{- 2 \alpha (r - r_0)} - 2 e^{- \alpha (r - r_0)} \right]
 *       \qquad r < r_c
 * "The following coefficients must be defined for each pair of atoms types
 * via the pair_coeff command":
 *   * :math:`D_0` (energy units)
 *   * :math:`\alpha` (1/distance units)
 *   * :math:`r_0` (distance units)
 *   * cutoff (distance units)
 * "The last coefficient is optional. If not specified, the global morse
 * cutoff is used."
 * "None of these pair styles support mixing." "All of these pair styles
 * support the pair_modify shift option for the energy of the pair
 * interaction." "None of these pair styles support the pair_modify tail
 * option for adding long-range tail corrections to energy and pressure."
 */
export class PairMorse extends Pair {
  readonly name: string = 'morse';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  morseD0 = new Float64Array(0);
  morseAlpha = new Float64Array(0);
  morseR0 = new Float64Array(0);
  offset = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style morse cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['D0', 'alpha', 'r0', 'cut']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 5 || args.length > 6) throw new StyleError('usage: pair_coeff I J D0 alpha r0 [cutoff]');
    const d0 = parseNum(args[2], 'D0');
    const alpha = parseNum(args[3], 'alpha');
    const r0 = parseNum(args[4], 'r0');
    const cut = args[5] !== undefined ? parseNum(args[5], 'cutoff') : Number.NaN;
    this.p.setRange(args[0], args[1], [d0, alpha, r0, cut]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.tail) throw new StyleError('pair style morse does not support the pair_modify tail option');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.ntypes + 1;
    if (this.morseD0.length !== nt * nt) {
      this.morseD0 = new Float64Array(nt * nt);
      this.morseAlpha = new Float64Array(nt * nt);
      this.morseR0 = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
    const d0 = p.get('D0', i, j), alpha = p.get('alpha', i, j), r0 = p.get('r0', i, j);
    const cut = p.get('cut', i, j);
    const rc = Number.isNaN(cut) ? this.cutGlobal : cut;
    const k1 = i * nt + j, k2 = j * nt + i;
    this.morseD0[k1] = this.morseD0[k2] = d0;
    this.morseAlpha[k1] = this.morseAlpha[k2] = alpha;
    this.morseR0[k1] = this.morseR0[k2] = r0;
    if (this.shift && rc > 0) {
      const d = rc - r0;
      const e1 = Math.exp(-2 * alpha * d), e2 = Math.exp(-alpha * d);
      this.offset[k1] = this.offset[k2] = d0 * (e1 - 2 * e2);
    } else this.offset[k1] = this.offset[k2] = 0;
    return rc;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, morseD0, morseAlpha, morseR0, offset } = this;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const factor = sLJ[jj >>> SBBITS];
        const dr = Math.sqrt(rsq) - morseR0[t];
        const e1 = Math.exp(-2 * morseAlpha[t] * dr), e2 = Math.exp(-morseAlpha[t] * dr);
        // F_vec = 2 alpha D0 [e^{-2 alpha (r-r0)} - e^{-alpha (r-r0)}] * rhat
        const fpair = factor * 2 * morseAlpha[t] * morseD0[t] * (e1 - e2) / Math.sqrt(rsq);
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * (morseD0[t] * (e1 - 2 * e2) - offset[t]);
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    const r = Math.sqrt(rsq);
    const dr = r - this.morseR0[t];
    const e1 = Math.exp(-2 * this.morseAlpha[t] * dr), e2 = Math.exp(-this.morseAlpha[t] * dr);
    return {
      fforce: factorLJ * 2 * this.morseAlpha[t] * this.morseD0[t] * (e1 - e2) / r,
      eng: factorLJ * (this.morseD0[t] * (e1 - 2 * e2) - this.offset[t]),
    };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('D0', i, i))} ${fmtCoeff(this.p.get('alpha', i, i))} ${fmtCoeff(this.p.get('r0', i, i))}`);
    }
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    const nt = this.ntypes + 1;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('D0', i, j))} ${fmtCoeff(this.p.get('alpha', i, j))} ${fmtCoeff(this.p.get('r0', i, j))} ${fmtCoeff(this.cut[i * nt + j])}`);
      }
    }
    return out;
  }
}
