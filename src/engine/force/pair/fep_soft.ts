import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { parseNum } from '../util';
import { erfcPoly, EWALD_F } from '../erfc';
import { tallyAtom } from './lj_cut';

/*
 * Soft-core free-energy pair styles (docs.lammps.org/pair_fep_soft.html).
 *
 * Doc: "These pair styles have a soft repulsive core, tunable by a parameter lambda,
 * in order to avoid singularities during free energy calculations when sites are
 * created or annihilated". The 12-6 soft-core LJ energy of the page is
 *   E = lambda^n 4 eps { 1/[alpha_LJ (1-lambda)^2 + (r/sigma)^6]^2 - 1/[alpha_LJ (1-lambda)^2 + (r/sigma)^6] }  r < rc
 * and the soft Coulomb energy is
 *   E = lambda^n C q_i q_j / (eps [alpha_C (1-lambda)^2 + r^2]^(1/2))  r < rc
 * Doc: "The *lj/class2/soft* style is a 9-6 potential with the exponent of the
 * denominator of the first term in brackets taking the value 1.5 instead of 2", which gives
 * E = lambda^n eps { 2 D^(-3/2) - 3 D^(-1) } with the same D = alpha_LJ (1-lambda)^2 + (r/sigma)^6.
 * The long-range styles use the erfc real-space term of the Ewald sum on the soft distance
 * r_s = [alpha_C (1-lambda)^2 + r^2]^(1/2): E = lambda^n C q_i q_j erfc(g r) / r_s.
 * Measured with native LAMMPS (black box): all of the above agree with the native energies to
 * 1e-13 (lj/cut/soft, coul/cut/soft, lj/class2/soft and the real-space Ewald term at alpha_C 4 and
 * 10); the forces are their analytic derivatives (tests/enginePairSoft.test.ts).
 *
 * Special bonds (long styles): the excluded fraction (1 - f) of the soft bare Coulomb term is
 * removed, E_corr = -(1 - f) lambda^n C q_i q_j / r_s. Measured with native LAMMPS (black box) on a
 * bonded pair with special_bonds coul 0.5: the difference of ecoul equals -(1 - f) lambda^n C q q / r_s
 * to 1e-16. The coul/cut/soft style multiplies its whole energy by the special factor.
 *
 * pair_modify table: measured with native LAMMPS (black box): the long soft styles give identical
 * ecoul with pair_modify table 0 and table 12, so the direct erfc polynomial (erfc.ts erfcPoly) is
 * always used here.
 *
 * Mixing: doc: "atom type pairs I,J and I != J, the :math:`\epsilon` and :math:`\sigma`
 * coefficients and cutoff distance for these pair styles can be mixed." Measured with native
 * LAMMPS (black box): when the lambda values of types I and J differ, the cross term cannot be
 * mixed (native stops with an error); equal lambda values are kept. The class2 styles use the sixthpower rule for epsilon and
 * sigma: doc: "The mixing rule for epsilon and sigma for *lj/class2/soft* 9-6 potentials is to use
 * the *sixthpower* formulas." The cutoffs follow pair_modify mix.
 *
 * pair_modify shift: doc: "All of the pair styles with soft core support the" pair_modify "shift
 * option for the energy of the Lennard-Jones portion of the pair interaction." The offset is the soft
 * LJ energy at the LJ cutoff times lambda^n. Measured with native LAMMPS (black box): the class2 offset
 * equals E(r) - E(rc) for the lambda^n scaled 9-6 form.
 *
 * pair_modify tail: doc: "The different versions of the *lj/cut/soft* pair styles support the" tail
 * option "for adding a long-range tail correction to the energy and pressure for the Lennard-Jones
 * portion of the pair interaction." The correction is the standard 12-6 tail of each type pair scaled
 * by lambda^n. Measured with native LAMMPS (black box): with lambda 0.6 and n 2 the etail is 0.36 of
 * the lambda 1 value (tests/oracle/w17soft_tail.in).
 *
 * Charmm (lj/charmm/coul/long/soft): the LJ soft energy is multiplied by the energy switch S(r) of
 * pair_charmm.ts. Measured with native LAMMPS (black box): evdwl equals S(r) times the soft energy at
 * r = 8.5, 9.5 and 9.9 (inner 8, outer 10). The Coulomb term is not switched.
 */

export type SoftLJ = 'lj12' | 'class2' | 'none';
export type SoftCoul = 'none' | 'cut' | 'long';

/** Parameter names: epsilon, sigma, lambda, LJ cutoff, Coulomb cutoff. */
const VALUES = ['epsilon', 'sigma', 'lambda', 'cut', 'cutc'];

export class PairFepSoft extends Pair {
  readonly name: string;
  readonly lj: SoftLJ;
  readonly coul: SoftCoul;
  virialFdotr = true;

  n = 0;
  alphaLJ = 0;
  alphaC = 0;
  cutLJGlobal = 0;
  cutCGlobal = 0;
  p!: PairParams;

  // per type pair after init (ntypes+1)^2, both orders
  protected epsT = new Float64Array(0);
  protected sinv6 = new Float64Array(0);
  protected lamN = new Float64Array(0);
  protected aL = new Float64Array(0);
  protected aC = new Float64Array(0);
  protected offset = new Float64Array(0);
  protected cutLJT = new Float64Array(0);
  protected cutLJsq = new Float64Array(0);
  protected cutCsq = new Float64Array(0);

  /** Scratch outputs of ljPair (energy, F/r) to avoid allocation in the pair loop. */
  protected kE = 0;
  protected kF = 0;

  constructor(name: string, lj: SoftLJ, coul: SoftCoul) {
    super();
    this.name = name;
    this.lj = lj;
    this.coul = coul;
    this.coulLong = coul === 'long';
    this.keepExcluded = coul === 'long';
  }

  settings(args: string[]): void {
    const usage = this.usage();
    if (this.lj === 'none') {
      if (args.length !== 3) throw new StyleError(usage);
      this.n = parseNum(args[0], 'n');
      this.alphaC = parseNum(args[1], 'alpha_C');
      this.cutCGlobal = parseNum(args[2], 'cutoff');
    } else if (this.coul === 'none') {
      if (args.length !== 3) throw new StyleError(usage);
      this.n = parseNum(args[0], 'n');
      this.alphaLJ = parseNum(args[1], 'alpha_LJ');
      this.cutLJGlobal = parseNum(args[2], 'cutoff');
    } else {
      if (args.length !== 4 && args.length !== 5) throw new StyleError(usage);
      this.n = parseNum(args[0], 'n');
      this.alphaLJ = parseNum(args[1], 'alpha_LJ');
      this.alphaC = parseNum(args[2], 'alpha_C');
      this.cutLJGlobal = parseNum(args[3], 'cutoff');
      // "cutoff2 = global cutoff for Coulombic (optional)"
      this.cutCGlobal = args.length === 5 ? parseNum(args[4], 'cutoff2') : this.cutLJGlobal;
    }
    this.checkCutoffs();
  }

  protected checkCutoffs(): void {
    if (this.lj !== 'none' && !(this.cutLJGlobal > 0)) throw new StyleError(`${this.name}: the LJ cutoff must be > 0`);
    if (this.coul !== 'none' && !(this.cutCGlobal > 0)) throw new StyleError(`${this.name}: the Coulomb cutoff must be > 0`);
  }

  protected usage(): string {
    if (this.lj === 'none') return `usage: pair_style ${this.name} n alpha_C cutoff`;
    if (this.coul === 'none') return `usage: pair_style ${this.name} n alpha_LJ cutoff`;
    return `usage: pair_style ${this.name} n alpha_LJ alpha_C cutoff (cutoff2)`;
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, VALUES);
  }

  coeff(args: string[]): void {
    if (this.lj === 'none') {
      // Doc: "only lambda and the optional cutoff2 are to be" (for coul/cut/soft and coul/long/soft)
      // Measured with native LAMMPS (black box): coul/long/soft accepts the lambda only (a cutoff is rejected).
      const maxArgs = this.coul === 'long' ? 3 : 4;
      if (args.length !== 3 && args.length !== maxArgs) throw new StyleError(`usage: pair_coeff I J lambda${maxArgs === 4 ? ' [cutoff]' : ''} for pair style ${this.name}`);
      const lam = parseNum(args[2], 'lambda');
      const cut = args[3] !== undefined ? parseNum(args[3], 'cutoff') : this.cutCGlobal;
      this.p.setRange(args[0], args[1], [0, 0, lam, cut, cut]);
      return;
    }
    // Measured with native LAMMPS (black box): the long styles take one per-pair cutoff (the LJ one);
    // the Coulomb cutoff stays global for them, so 7 arguments are rejected there.
    const maxArgs = this.coul === 'long' ? 6 : 7;
    if (args.length < 5 || args.length > maxArgs) {
      throw new StyleError(`usage: pair_coeff I J epsilon sigma lambda [cutoff1 [cutoff2]] for pair style ${this.name}`);
    }
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const lam = parseNum(args[4], 'lambda');
    // "If not specified, the global LJ and Coulombic cutoffs specified in the pair_style command are used.
    // If only one cutoff is specified, it is used as the cutoff for both LJ and Coulombic interactions"
    let cutLJ = this.cutLJGlobal, cutC = this.coul === 'none' ? this.cutLJGlobal : this.cutCGlobal;
    if (args.length === 6 && this.coul === 'long') cutLJ = parseNum(args[5], 'cutoff');
    else if (args.length === 6) cutLJ = cutC = parseNum(args[5], 'cutoff');
    if (args.length === 7) {
      cutLJ = parseNum(args[5], 'cutoff1');
      cutC = parseNum(args[6], 'cutoff2');
    }
    if (this.coul === 'none' && args.length === 7) throw new StyleError(`pair_coeff: ${this.name} has no Coulombic cutoff`);
    this.p.setRange(args[0], args[1], [eps, sig, lam, cutLJ, cutC]);
  }

  initStyle(_ctx: StyleContext): void {
    // pair_modify.html is the source of the shift/tail rules; fep_soft.rst gives the tail for lj/cut/soft only
    if (this.tail && this.lj !== 'lj12') {
      throw new StyleError(`pair_modify tail yes is not supported for pair style ${this.name} (only the lj/cut/soft styles have a tail correction)`);
    }
    if (this.shift && this.lj === 'none') {
      throw new StyleError(`pair_modify shift yes is not applicable to pair style ${this.name} (it has no Lennard-Jones portion)`);
    }
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      const li = p.get('lambda', i, i), lj = p.get('lambda', j, j);
      if (li !== lj) {
        throw new StyleError(`pair ${this.name}: types ${i} and ${j} have different lambda values, so their cross term cannot be mixed (set pair_coeff ${i} ${j} explicitly)`);
      }
      if (this.lj !== 'none') {
        const rule = this.lj === 'class2' ? 'sixthpower' : this.mix;
        const e = mixEpsilon(rule, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j));
        const s = mixDistance(rule, p.get('sigma', i, i), p.get('sigma', j, j));
        p.setMixed(i, j, 'epsilon', e);
        p.setMixed(i, j, 'sigma', s);
      }
      p.setMixed(i, j, 'lambda', li);
      const cl = mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j));
      const cc = mixDistance(this.mix, p.get('cutc', i, i), p.get('cutc', j, j));
      p.setMixed(i, j, 'cut', cl);
      p.setMixed(i, j, 'cutc', cc);
    }
    const nt = this.ntypes + 1;
    if (this.lamN.length !== nt * nt) {
      const z = () => new Float64Array(nt * nt);
      this.epsT = z(); this.sinv6 = z(); this.lamN = z(); this.aL = z(); this.aC = z();
      this.offset = z(); this.cutLJT = z(); this.cutLJsq = z(); this.cutCsq = z();
    }
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), lam = p.get('lambda', i, j);
    const cutLJ = p.get('cut', i, j), cutC = p.get('cutc', i, j);
    if (this.lj !== 'none' && !(cutLJ > 0)) throw new StyleError(`pair ${this.name}: LJ cutoff for types ${i} ${j} must be > 0`);
    if (this.coul !== 'none' && !(cutC > 0)) throw new StyleError(`pair ${this.name}: Coulomb cutoff for types ${i} ${j} must be > 0`);
    // lambda, n and the alpha parameters, per the functional forms in the file header
    const pl = lam ** this.n;
    const oml2 = (1 - lam) * (1 - lam);
    const aL = this.alphaLJ * oml2, aC = this.alphaC * oml2;
    const sinv6 = 1 / sig ** 6;
    const k1 = i * nt + j, k2 = j * nt + i;
    for (const k of [k1, k2]) {
      this.epsT[k] = eps;
      this.sinv6[k] = sinv6;
      this.lamN[k] = pl;
      this.aL[k] = aL;
      this.aC[k] = aC;
      this.cutLJT[k] = cutLJ;
      this.cutLJsq[k] = this.lj !== 'none' ? cutLJ * cutLJ : 0;
      this.cutCsq[k] = this.coul !== 'none' ? cutC * cutC : 0;
      this.offset[k] = 0;
    }
    // pair_modify shift: the soft LJ energy at the cutoff (times lambda^n) is subtracted
    if (this.shift && this.lj !== 'none') {
      const off = pl * this.ljEnergyAt(cutLJ * cutLJ, eps, sinv6, aL);
      this.offset[k1] = this.offset[k2] = off;
    }
    if (this.lj === 'none') return cutC;
    return this.coul !== 'none' ? Math.max(cutLJ, cutC) : cutLJ;
  }

  /** Soft LJ energy without lambda^n, the offset kernel. */
  private ljEnergyAt(rsq: number, eps: number, sinv6: number, aL: number): number {
    const x = rsq * rsq * rsq * sinv6;
    const D = aL + x;
    const iD = 1 / D;
    return this.lj === 'class2' ? eps * (2 * iD * Math.sqrt(iD) - 3 * iD) : 4 * eps * (iD * iD - iD);
  }

  /**
   * Soft LJ energy (without the offset) and F/r into kE/kF for type-pair index t.
   * docs.lammps.org/pair_fep_soft.html: the LJ form above; F/r = -(dE/dr)/r with
   * D = aL + (r/sigma)^6 and dD/dr = 6 x / r.
   */
  protected ljPair(rsq: number, t: number): void {
    const x = rsq * rsq * rsq * this.sinv6[t];
    const D = this.aL[t] + x;
    const iD = 1 / D;
    const pl = this.lamN[t], eps = this.epsT[t];
    if (this.lj === 'class2') {
      const sq = Math.sqrt(iD);
      this.kE = pl * eps * (2 * iD * sq - 3 * iD);
      this.kF = (pl * eps * 18 * x * (iD * iD * sq - iD * iD)) / rsq;
    } else {
      this.kE = pl * 4 * eps * (iD * iD - iD);
      this.kF = (pl * 24 * eps * x * (2 * iD * iD * iD - iD * iD)) / rsq;
    }
  }

  /**
   * Soft Coulomb for one pair: energy into kE, F/r into kF. fc is the special_bonds
   * Coulomb weight. Cut: E = fc lambda^n C qq / r_s. Long: E = lambda^n C qq (erfc(g r) - (1 - fc)) / r_s.
   */
  protected coulPair(rsq: number, qq: number, t: number, fc: number, qqrd2e: number, g: number): void {
    const rs2 = this.aC[t] + rsq;
    const rs = Math.sqrt(rs2);
    if (this.coul === 'cut') {
      const pre = fc * this.lamN[t] * qqrd2e * qq;
      this.kE = pre / rs;
      this.kF = pre / (rs2 * rs);
      return;
    }
    const r = Math.sqrt(rsq);
    const gr = g * r;
    const ex = Math.exp(-gr * gr);
    const ec = erfcPoly(gr, ex);
    const P = this.lamN[t] * qqrd2e * qq;
    const corr = ec - (1 - fc);
    this.kE = (P * corr) / rs;
    this.kF = P * (EWALD_F * g * ex / (r * rs) + corr / (rs2 * rs));
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const hasLJ = this.lj !== 'none', hasC = this.coul !== 'none';
    const tally = pc.eatom !== null || pc.vatom !== null;
    const g = this.gEwald, qqrd2e = pc.qqrd2e;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
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
        let fpair = 0, e = 0;
        if (hasLJ && rsq < this.cutLJsq[t]) {
          const fl = sLJ[sb];
          this.ljPair(rsq, t);
          const ev = fl * (this.kE - this.offset[t]);
          fpair += fl * this.kF;
          e += ev;
          evdwl += ev;
        }
        if (hasC && qi !== 0 && q[j] !== 0 && rsq < this.cutCsq[t]) {
          this.coulPair(rsq, qi * q[j], t, sC[sb], qqrd2e, g);
          fpair += this.kF;
          e += this.kE;
          ecoul += this.kE;
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

  /** Standard 12-6 tail per type pair scaled by lambda^n (fep_soft.rst, pair_modify tail). */
  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail || this.lj !== 'lj12') return { etail: 0, ptail: 0 };
    let e = 0, pr = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
        const rc = this.cutLJT[i * nt + j];
        if (!(rc > 0)) continue;
        const s6 = sig ** 6, s12 = s6 * s6, rc3 = rc ** 3, rc9 = rc3 ** 3;
        const nn = count[i] * count[j] * this.lamN[i * nt + j];
        e += nn * 4 * eps * (s12 / (9 * rc9) - s6 / (3 * rc3));
        pr += nn * 4 * eps * (-4 * s12 / (3 * rc9) + 2 * s6 / rc3);
      }
    }
    return { etail: 2 * Math.PI * e, ptail: (-2 * Math.PI / 3) * pr };
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCGlobal;
    return undefined;
  }
}

/**
 * lj/charmm/coul/long/soft: the soft LJ energy times the CHARMM energy switch S(r) between the
 * inner and outer cutoffs; the Coulomb term is the unswitched soft long-range real-space term.
 * docs.lammps.org/pair_fep_soft.html syntax: "*lj/charmm/coul/long/soft* args = n alpha_LJ alpha_C inner outer (cutoff)" and "inner, outer = global switching cutoffs for LJ (and Coulombic if only 5 args)".
 * S(r) = (b^2 - r^2)^2 (b^2 + 2 r^2 - 3 a^2) / (b^2 - a^2)^3 as in pair_charmm.ts.
 */
export class PairLJCharmmSoftCoulLong extends PairFepSoft {
  private innerCut = 0;
  private outerCut = 0;

  constructor() {
    super('lj/charmm/coul/long/soft', 'lj12', 'long');
  }

  settings(args: string[]): void {
    if (args.length !== 5 && args.length !== 6) {
      throw new StyleError('usage: pair_style lj/charmm/coul/long/soft n alpha_LJ alpha_C inner outer (cutoff)');
    }
    this.n = parseNum(args[0], 'n');
    this.alphaLJ = parseNum(args[1], 'alpha_LJ');
    this.alphaC = parseNum(args[2], 'alpha_C');
    this.innerCut = parseNum(args[3], 'inner');
    this.outerCut = parseNum(args[4], 'outer');
    if (!(this.innerCut > 0) || !(this.outerCut > 0)) throw new StyleError(`${this.name}: cutoffs must be > 0`);
    if (this.innerCut >= this.outerCut) throw new StyleError(`${this.name}: the inner cutoff must be less than the outer cutoff`);
    this.cutLJGlobal = this.outerCut;
    // "outer is Coulombic cutoff if only 5 args"
    this.cutCGlobal = args.length === 6 ? parseNum(args[5], 'cutoff') : this.outerCut;
    this.checkCutoffs();
  }

  initStyle(ctx: StyleContext): void {
    // the page's shift and tail notes cover the lj/cut/soft and class2 families, not the charmm switch
    if (this.shift) throw new StyleError(`pair_modify shift yes is not supported for pair style ${this.name} (the switching function already goes to 0.0 at the cutoff)`);
    if (this.tail) throw new StyleError(`pair_modify tail yes is not supported for pair style ${this.name}`);
    super.initStyle(ctx);
  }

  coeff(args: string[]): void {
    // eps14/sigma14 (the 1-4 CHARMM parameters) are not implemented for the soft styles
    if (args.length !== 5) {
      throw new StyleError(`usage: pair_coeff I J epsilon sigma lambda for pair style ${this.name} (epsilon14/sigma14 are not supported)`);
    }
    super.coeff(args);
  }

  protected ljPair(rsq: number, t: number): void {
    super.ljPair(rsq, t);
    const a2 = this.innerCut * this.innerCut, b2 = this.outerCut * this.outerCut;
    if (rsq <= a2) return;
    const den = (b2 - a2) ** 3;
    const S = (b2 - rsq) ** 2 * (b2 + 2 * rsq - 3 * a2) / den;
    // -S'(r)/r = 12 (b^2 - r^2)(r^2 - a^2) / (b^2 - a^2)^3 (switching term of the force)
    const Sx = 12 * (b2 - rsq) * (rsq - a2) / den;
    this.kF = S * this.kF + this.kE * Sx;
    this.kE *= S;
  }
}

/**
 * Soft-core styles the page indexes but this engine does not implement: a StyleError names
 * the style at pair_style time (rule 3: no silent approximation).
 */
export class PairSoftUnsupported extends Pair {
  readonly name: string;
  constructor(name: string, private readonly why: string) {
    super();
    this.name = name;
  }
  private fail(): never {
    throw new StyleError(`pair_style ${this.name} is not supported by the engine: ${this.why}`);
  }
  settings(): void { this.fail(); }
  coeff(): void { this.fail(); }
  initOne(): number { return this.fail(); }
  compute(): void { this.fail(); }
}
