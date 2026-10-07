import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type MixRule, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { fmtCoeff, parseNum } from '../util';

/*
 * Second wave of LJ-family pair styles, written from the LAMMPS documentation
 * only. All four share the loop shape of PairLJCut (docs.lammps.org/pair_lj.html):
 * half neighbor list, special_bonds weights, per-atom tallies split half and
 * half, virial computed by the force field as the x.f dot product
 * (virialFdotr = true). Subclasses provide the per-pair energy/force kernel
 * and the mixing rules documented on their own page.
 *
 * Tail corrections (where supported) follow docs.lammps.org/pair_modify.html,
 * "The formulas used for the long-range corrections come from equation 5 of
 * (Sun)":
 *   etail = 2 pi / V sum_ij N_i N_j  int_rc^inf u(r) r^2 dr
 *   ptail = -2 pi / (3 V^2) sum_ij N_i N_j  int_rc^inf r^3 u'(r) dr
 * evaluated in closed form per potential; tailSums returns the V-free sums
 * (the caller divides by V, resp. V^2).
 */

/**
 * Shared frame for the wave-2 LJ variants. fpairBase is the radial force
 * divided by r (the "pair force magnitude / r" convention of PairLJCut:
 * fx = dx * fpair); eng is the pair energy including the shift offset, both
 * BEFORE the special_bonds factor.
 */
export abstract class PairLJ2Variant extends Pair {
  virialFdotr = true;
  p!: PairParams;
  /** Coefficient names in the documented pair_coeff order (after I J). */
  protected abstract readonly paramNames: readonly string[];
  protected abstract readonly coeffUsage: string;

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, this.paramNames);
  }

  /** The pair_modify mix rule that applies to one coefficient (class2 overrides). */
  protected mixRule(_name: string): MixRule {
    return this.mix;
  }

  /** Mixed value of one coefficient from the I,I and J,J self pairs. */
  protected mixOne(name: string, i: number, j: number): number {
    const rule = this.mixRule(name);
    if (name === 'epsilon') {
      return mixEpsilon(rule, this.p.get('epsilon', i, i), this.p.get('epsilon', j, j), this.p.get('sigma', i, i), this.p.get('sigma', j, j));
    }
    return mixDistance(rule, this.p.get(name, i, i), this.p.get(name, j, j));
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      for (const name of this.paramNames) p.setMixed(i, j, name, this.mixOne(name, i, j));
    }
    const nt = this.ntypes + 1;
    this.ensureTables(nt);
    return this.initKernel(i * nt + j, j * nt + i, i, j);
  }

  /** (Re)allocates this style's per-pair tables when the type count changed. */
  protected abstract ensureTables(nt: number): void;
  /** Fills the per-pair tables at index k1 (and k2 = mirror); returns the cutoff. */
  protected abstract initKernel(k1: number, k2: number, i: number, j: number): number;
  /** Energy (with offset, without special factor) and radial-force/r of one pair. */
  protected abstract pairKernel(t: number, rsq: number): { eng: number; fpairBase: number };

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
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
        if (rsq >= this.cutsq[t]) continue;
        const factor = sLJ[jj >>> SBBITS];
        const kern = this.pairKernel(t, rsq);
        const fpair = factor * kern.fpairBase;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * kern.eng;
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _factorCoul: number, factorLJ: number): { eng: number; fforce: number } {
    const t = itype * (this.ntypes + 1) + jtype;
    const k = this.pairKernel(t, rsq);
    return { fforce: factorLJ * k.fpairBase, eng: factorLJ * k.eng };
  }

  /** One data-file coefficient line: the leading type word(s) then named values. */
  protected coeffLine(words: readonly (string | number)[], names: readonly string[]): string {
    let out = words.join(' ');
    for (const n of names) out += ` ${fmtCoeff(this.p.get(n, Number(words[0]), Number(words[words.length - 1])))}`;
    return out;
  }

  extract(name: string): unknown {
    if (name === 'epsilon') return this.p.p('epsilon');
    if (name === 'sigma') return this.p.p('sigma');
    return undefined;
  }
}

/*
 * pair_style lj/gromacs inner outer — docs.lammps.org/pair_gromacs.html:
 * "The lj/gromacs styles compute shifted LJ and Coulombic interactions with an
 * additional switching function S(r) that ramps the energy and force smoothly to
 * zero between an inner and outer cutoff."
 *
 *   E_{LJ} = & 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
 *            \left(\frac{\sigma}{r}\right)^6 \right] + S_{LJ}(r)
 *                       \qquad r < r_c \\
 *   S(r) = & C \qquad r < r_1 \\
 *   S(r) = & \frac{A}{3} (r - r_1)^3 + \frac{B}{4} (r - r_1)^4 + C \qquad  r_1 < r < r_c \\
 *   A = & (-3 E'(r_c) + (r_c - r_1) E''(r_c))/(r_c - r_1)^2 \\
 *   B = & (2 E'(r_c) - (r_c - r_1) E''(r_c))/(r_c - r_1)^3 \\
 *   C = & -E(r_c) + \frac{1}{2} (r_c - r_1) E'(r_c) - \frac{1}{12} (r_c - r_1)^2 E''(r_c)
 *
 * "The coefficients A, B, and C are computed by LAMMPS to perform the shifting
 * and smoothing.  The function S(r) is actually applied once to each term of the
 * LJ formula and once to the Coulombic formula". A, B, C are linear in
 * (E, E', E'') evaluated at r_c and the polynomial is linear in (A, B, C), so
 * applying one set to the summed LJ term equals the per-term application.
 * Boundary conditions: ":math:`S'(r_1) = S''(r_1) = 0, S(r_c) = -E(r_c), S'(r_c) =
 * -E'(r_c)`, and :math:`S''(r_c) = -E''(r_c)`".
 * "The inner and outer cutoff for the LJ and Coulombic terms can be the same or
 * different depending on whether 2 or 4 arguments are used in the pair_style
 * command.  The inner LJ cutoff must be > 0".
 * Coefficients ("The following coefficients must be defined for each pair of
 * atoms types via the pair_coeff command"):
 *
 * * :math:`\epsilon` (energy units)
 * * :math:`\sigma` (distance units)
 * * inner (distance units)
 * * outer (distance units)
 *
 * "The last 2 coefficients are optional inner and outer cutoffs for style
 * lj/gromacs"; "If not specified, the global inner and outer values are used."
 * Mixing: "For atom type pairs I,J and I != J, the epsilon and sigma coefficients
 * and cutoff distance for all of the lj/cut pair styles can be mixed.
 * The default mix value is geometric".
 * "None of the GROMACS pair styles support the pair_modify shift option"
 * and "None of the GROMACS pair styles support the pair_modify tail option".
 */
export class PairLJGromacs extends PairLJ2Variant {
  readonly name: string = 'lj/gromacs';
  innerGlobal = 0;
  outerGlobal = 0;
  protected readonly paramNames = ['epsilon', 'sigma', 'inner', 'outer'] as const;
  protected readonly coeffUsage = 'usage: pair_coeff I J epsilon sigma [inner outer]';
  lj1 = new Float64Array(0);
  lj2 = new Float64Array(0);
  lj3 = new Float64Array(0);
  lj4 = new Float64Array(0);
  rinTab = new Float64Array(0);
  aTab = new Float64Array(0);
  bTab = new Float64Array(0);
  cTab = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError(`usage: pair_style ${this.name} inner outer`);
    this.innerGlobal = parseNum(args[0], 'inner');
    this.outerGlobal = parseNum(args[1], 'outer');
    if (!(this.innerGlobal > 0)) throw new StyleError('lj/gromacs: the inner cutoff must be > 0');
    if (!(this.outerGlobal > this.innerGlobal)) throw new StyleError(`lj/gromacs: inner cutoff ${this.innerGlobal} must be less than outer cutoff ${this.outerGlobal}`);
  }

  coeff(args: string[]): void {
    if (args.length !== 4 && args.length !== 6) throw new StyleError(this.coeffUsage);
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const inner = args[4] !== undefined ? parseNum(args[4], 'inner') : this.innerGlobal;
    const outer = args[5] !== undefined ? parseNum(args[5], 'outer') : this.outerGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, inner, outer]);
  }

  init(ctx: StyleContext): void {
    // docs.lammps.org/pair_gromacs.html: "None of the GROMACS pair styles support the
    // pair_modify shift option, since the Lennard-Jones portion of the pair interaction
    // is already smoothed to 0.0 at the cutoff." / "None of the GROMACS pair styles
    // support the pair_modify tail option for adding long-range tail corrections to
    // energy and pressure, since there are no corrections for a potential that goes
    // to 0.0 at the cutoff."
    if (this.shift) throw new StyleError('pair_modify shift yes is not supported for pair style lj/gromacs (the interaction is already smoothed to 0.0 at the cutoff)');
    if (this.tail) throw new StyleError('pair_modify tail yes is not supported for pair style lj/gromacs (there are no corrections for a potential that goes to 0.0 at the cutoff)');
    super.init(ctx);
  }

  protected ensureTables(nt: number): void {
    if (this.lj1.length !== nt * nt) {
      this.lj1 = new Float64Array(nt * nt); this.lj2 = new Float64Array(nt * nt);
      this.lj3 = new Float64Array(nt * nt); this.lj4 = new Float64Array(nt * nt);
      this.rinTab = new Float64Array(nt * nt);
      this.aTab = new Float64Array(nt * nt); this.bTab = new Float64Array(nt * nt);
      this.cTab = new Float64Array(nt * nt);
    }
  }

  protected initKernel(k1: number, k2: number, i: number, j: number): number {
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
    const r1 = this.p.get('inner', i, j), rc = this.p.get('outer', i, j);
    if (!(r1 > 0)) throw new StyleError('lj/gromacs: the inner cutoff must be > 0');
    if (!(rc > r1)) throw new StyleError(`lj/gromacs: inner cutoff ${r1} must be less than outer cutoff ${rc}`);
    const s6 = sig ** 6, s12 = s6 * s6;
    this.lj1[k1] = this.lj1[k2] = 48 * eps * s12;
    this.lj2[k1] = this.lj2[k2] = 24 * eps * s6;
    this.lj3[k1] = this.lj3[k2] = 4 * eps * s12;
    this.lj4[k1] = this.lj4[k2] = 4 * eps * s6;
    this.rinTab[k1] = this.rinTab[k2] = r1;
    // E, E', E'' of the LJ term at r_c, then the documented A, B, C with d = r_c - r_1
    const ec = 4 * eps * (s12 / rc ** 12 - s6 / rc ** 6);
    const dEc = 4 * eps * (-12 * s12 / rc ** 13 + 6 * s6 / rc ** 7);
    const ddEc = 4 * eps * (156 * s12 / rc ** 14 - 42 * s6 / rc ** 8);
    const d = rc - r1;
    this.aTab[k1] = this.aTab[k2] = (-3 * dEc + d * ddEc) / (d * d);
    this.bTab[k1] = this.bTab[k2] = (2 * dEc - d * ddEc) / (d * d * d);
    this.cTab[k1] = this.cTab[k2] = -ec + 0.5 * d * dEc - d * d * ddEc / 12;
    return rc;
  }

  protected pairKernel(t: number, rsq: number): { eng: number; fpairBase: number } {
    const r = Math.sqrt(rsq);
    const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
    const fLJ = r6inv * (this.lj1[t] * r6inv - this.lj2[t]) * r2inv;
    const engLJ = r6inv * (this.lj3[t] * r6inv - this.lj4[t]);
    if (r <= this.rinTab[t]) return { fpairBase: fLJ, eng: engLJ + this.cTab[t] };
    const s = r - this.rinTab[t], s2 = s * s, s3 = s2 * s;
    return {
      // S'(r) = A s^2 + B s^3 ; E = E_LJ + A/3 s^3 + B/4 s^4 + C
      fpairBase: fLJ - (this.aTab[t] * s2 + this.bTab[t] * s3) / r,
      eng: engLJ + this.cTab[t] + (this.aTab[t] / 3) * s3 + (this.bTab[t] / 4) * s3 * s,
    };
  }

  tailSums(): { etail: number; ptail: number } {
    throw new StyleError('pair_modify tail yes is not supported for pair style lj/gromacs (there are no corrections for a potential that goes to 0.0 at the cutoff)');
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(this.coeffLine([i], ['epsilon', 'sigma', 'inner']));
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) out.push(this.coeffLine([i, j], ['epsilon', 'sigma', 'inner', 'outer']));
    }
    return out;
  }
}

/*
 * pair_style lj/class2 cutoff — docs.lammps.org/pair_class2.html:
 * "The lj/class2 styles compute a 6/9 Lennard-Jones potential given by"
 *
 *   E = \epsilon \left[ 2 \left(\frac{\sigma}{r}\right)^9 -
 *     3 \left(\frac{\sigma}{r}\right)^6 \right]
 *   \qquad r < r_c
 *
 * ":math:`r_c` is the cutoff."
 * Coefficients:
 *
 * * :math:`\epsilon` (energy units)
 * * :math:`\sigma` (distance units)
 * * cutoff1 (distance units)
 * * cutoff2 (distance units)
 *
 * "The latter 2 coefficients are optional."; for the plain lj/class2 style
 * "You cannot specify 2 cutoffs for style lj/class2, since it has no
 * Coulombic terms." Only cutoff1 applies (there is no Coulombic part).
 * Mixing: "Epsilon and sigma are always mixed with the value sixthpower.  The
 * cutoff distance is mixed by whatever option is set by the pair_modify command
 * (default = geometric)."
 * "All of the lj/class2 pair styles support the pair_modify shift option" (the
 * offset eps [2 (sigma/rc)^9 - 3 (sigma/rc)^6] makes E(rc) = 0) "and ... the
 * pair_modify tail option for adding a long-range tail correction to the energy
 * and pressure of the Lennard-Jones portion of the pair interaction". Tail
 * integrals (Sun, equation 5) of E = eps (2 s9 r^-9 - 3 s6 r^-6):
 *   int_rc^inf u r^2 dr  = eps (s9/(3 rc^6) - s6/rc^3)
 *   int_rc^inf r^3 u' dr = eps (6 s6/rc^3 - 3 s9/rc^6)
 * with s9 = sigma^9, s6 = sigma^6.
 */
export class PairLJClass2 extends PairLJ2Variant {
  readonly name: string = 'lj/class2';
  cutGlobal = 0;
  protected readonly paramNames = ['epsilon', 'sigma', 'cut'] as const;
  protected readonly coeffUsage = 'usage: pair_coeff I J epsilon sigma [cutoff]';
  e9 = new Float64Array(0);
  e6 = new Float64Array(0);
  f9 = new Float64Array(0);
  f6 = new Float64Array(0);
  offset = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError(`usage: pair_style ${this.name} cutoff`);
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  coeff(args: string[]): void {
    if (args.length < 4 || args.length > 5) {
      throw new StyleError(`${this.coeffUsage} (you cannot specify 2 cutoffs for style ${this.name}, since it has no Coulombic terms)`);
    }
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const cut = args[4] !== undefined ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, cut]);
  }

  protected mixRule(name: string): MixRule {
    // docs.lammps.org/pair_class2.html: "Epsilon and sigma are always mixed with the
    // value sixthpower.  The cutoff distance is mixed by whatever option is set by
    // the pair_modify command (default = geometric)."
    if (name === 'epsilon' || name === 'sigma') return 'sixthpower';
    return this.mix;
  }

  protected ensureTables(nt: number): void {
    if (this.e9.length !== nt * nt) {
      this.e9 = new Float64Array(nt * nt); this.e6 = new Float64Array(nt * nt);
      this.f9 = new Float64Array(nt * nt); this.f6 = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
  }

  protected initKernel(k1: number, k2: number, i: number, j: number): number {
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j), cut = this.p.get('cut', i, j);
    const s6 = sig ** 6, s9 = s6 * sig ** 3;
    // E = eps (2 s9 r^-9 - 3 s6 r^-6) ; F_r = -dE/dr = 18 eps s9 r^-10 - 18 eps s6 r^-7
    this.e9[k1] = this.e9[k2] = 2 * eps * s9;
    this.e6[k1] = this.e6[k2] = 3 * eps * s6;
    this.f9[k1] = this.f9[k2] = 18 * eps * s9;
    this.f6[k1] = this.f6[k2] = 18 * eps * s6;
    this.offset[k1] = this.offset[k2] = this.shift && cut > 0 ? 2 * eps * s9 / cut ** 9 - 3 * eps * s6 / cut ** 6 : 0;
    return cut;
  }

  protected pairKernel(t: number, rsq: number): { eng: number; fpairBase: number } {
    const rinv = 1 / Math.sqrt(rsq);
    const r2inv = rinv * rinv, r3inv = r2inv * rinv, r6inv = r3inv * r3inv, r9inv = r6inv * r3inv;
    return {
      // fpairBase = F_r / r = 18 eps s9 r^-11 - 18 eps s6 r^-8
      fpairBase: r2inv * (r9inv * this.f9[t] - r6inv * this.f6[t]),
      eng: r9inv * this.e9[t] - r6inv * this.e6[t] - this.offset[t],
    };
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    let e = 0, pr = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j), rc = this.cut[i * nt + j];
        if (!(rc > 0)) continue;
        const s6 = sig ** 6, s9 = s6 * sig ** 3, rc3 = rc ** 3, rc6 = rc3 * rc3;
        const nn = count[i] * count[j];
        // int_rc^inf u r^2 dr = eps (s9/(3 rc6) - s6/rc3)
        e += nn * eps * (s9 / (3 * rc6) - s6 / rc3);
        // int_rc^inf r^3 u' dr = eps (6 s6/rc3 - 3 s9/rc6)
        pr += nn * eps * (6 * s6 / rc3 - 3 * s9 / rc6);
      }
    }
    // caller divides by V (energy) and V^2 (pressure)
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(this.coeffLine([i], ['epsilon', 'sigma']));
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) out.push(this.coeffLine([i, j], ['epsilon', 'sigma', 'cut']));
    }
    return out;
  }
}

/*
 * pair_style lj/cubic — docs.lammps.org/pair_lj_cubic.html:
 * "The lj/cubic style computes a truncated LJ interaction potential whose energy
 * and force are continuous everywhere.  Inside the inflection point the
 * interaction is identical to the standard 12/6 Lennard-Jones potential.  The LJ
 * function outside the inflection point is replaced with a cubic function of
 * distance. The energy, force, and second derivative are continuous at the
 * inflection point.  The cubic coefficient A3 is chosen so that both energy and
 * force go to zero at the cutoff distance."
 *
 *   E & = u_{LJ}(r) \qquad r \leq r_s \\
 *     & = u_{LJ}(r_s) + (r-r_s) u'_{LJ}(r_s) - \frac{1}{6} A_3 (r-r_s)^3 \qquad r_s < r \leq r_c \\
 *     & = 0 \qquad r > r_c
 *
 * "The location of the inflection point :math:`r_s` is defined"
 * "by the LJ diameter, :math:`r_s/\sigma = (26/7)^{1/6}`. The cutoff distance"
 * "is defined by :math:`r_c/r_s = 67/48` or :math:`r_c/\sigma = 1.737...`"
 * "The analytic expression for the"
 * "the cubic coefficient"
 * ":math:`A_3 r_{min}^3/\epsilon = 27.93...` is given in the paper by"
 * "Holian and Ravelo". The four doc conditions (cubic matches u, u', u'' at r_s;
 * E and F vanish at r_c) determine A3 = 2 u'(r_s)/(r_c - r_s)^2; with the
 * documented r_s and r_c ratios both cutoff conditions hold exactly (they are
 * consistent only for (sigma/r_s)^6 = 7/26, which the doc fixes), and
 * A3 r_min^3/epsilon = 27.9336 with r_min = 2^(1/6) sigma.
 * Coefficients:
 *
 * * :math:`\epsilon` (energy units)
 * * :math:`\sigma` (distance units)
 *
 * Mixing: "For atom type pairs I,J and I != J, the epsilon and sigma coefficients
 * and cutoff distance for all of the lj/cut pair styles can be mixed.
 * The default mix value is geometric".
 * "The lj/cubic pair style does not support the pair_modify shift option" and
 * "The lj/cubic pair style does not support the pair_modify tail option".
 */
const RHO_S = (26 / 7) ** (1 / 6);
const RHO_C = (67 / 48) * RHO_S;
/** u_LJ(r_s)/eps and u'_LJ(r_s)sigma/eps at the inflection point (sigma-independent). */
const U_S = 4 * (RHO_S ** -12 - RHO_S ** -6);
const U_D = 4 * (-12 * RHO_S ** -13 + 6 * RHO_S ** -7);

export class PairLJCubic extends PairLJ2Variant {
  readonly name: string = 'lj/cubic';
  protected readonly paramNames = ['epsilon', 'sigma'] as const;
  protected readonly coeffUsage = 'usage: pair_coeff I J epsilon sigma';
  lj1 = new Float64Array(0);
  lj2 = new Float64Array(0);
  lj3 = new Float64Array(0);
  lj4 = new Float64Array(0);
  rsTab = new Float64Array(0);
  rs2 = new Float64Array(0);
  us = new Float64Array(0);
  ud = new Float64Array(0);
  wc = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError(`usage: pair_style ${this.name}`);
  }

  coeff(args: string[]): void {
    if (args.length !== 4) throw new StyleError(this.coeffUsage);
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    this.p.setRange(args[0], args[1], [eps, sig]);
  }

  init(ctx: StyleContext): void {
    // docs.lammps.org/pair_lj_cubic.html: "The lj/cubic pair style does not support the
    // pair_modify shift option, since pair interaction is already smoothed to 0.0 at
    // the cutoff." / "The lj/cubic pair style does not support the pair_modify tail
    // option for adding long-range tail corrections to energy and pressure, since
    // there are no corrections for a potential that goes to 0.0 at the cutoff."
    if (this.shift) throw new StyleError('pair_modify shift yes is not supported for pair style lj/cubic (the interaction is already smoothed to 0.0 at the cutoff)');
    if (this.tail) throw new StyleError('pair_modify tail yes is not supported for pair style lj/cubic (there are no corrections for a potential that goes to 0.0 at the cutoff)');
    super.init(ctx);
  }

  protected ensureTables(nt: number): void {
    if (this.lj1.length !== nt * nt) {
      this.lj1 = new Float64Array(nt * nt); this.lj2 = new Float64Array(nt * nt);
      this.lj3 = new Float64Array(nt * nt); this.lj4 = new Float64Array(nt * nt);
      this.rsTab = new Float64Array(nt * nt); this.rs2 = new Float64Array(nt * nt);
      this.us = new Float64Array(nt * nt); this.ud = new Float64Array(nt * nt);
      this.wc = new Float64Array(nt * nt);
    }
  }

  protected initKernel(k1: number, k2: number, i: number, j: number): number {
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
    const s6 = sig ** 6, s12 = s6 * s6;
    // pure LJ branch (r <= r_s), same tables as lj/cut
    this.lj1[k1] = this.lj1[k2] = 48 * eps * s12;
    this.lj2[k1] = this.lj2[k2] = 24 * eps * s6;
    this.lj3[k1] = this.lj3[k2] = 4 * eps * s12;
    this.lj4[k1] = this.lj4[k2] = 4 * eps * s6;
    const rs = RHO_S * sig, rc = RHO_C * sig;
    this.rsTab[k1] = this.rsTab[k2] = rs;
    this.rs2[k1] = this.rs2[k2] = rs * rs;
    // cubic branch: E = u(r_s) + (r-r_s) u'(r_s) - (A3/6)(r-r_s)^3 with
    // A3/6 = u'(r_s) / (3 (r_c - r_s)^2) so that E(r_c) = F(r_c) = 0
    const ud = (eps / sig) * U_D;
    this.us[k1] = this.us[k2] = eps * U_S;
    this.ud[k1] = this.ud[k2] = ud;
    this.wc[k1] = this.wc[k2] = ud / (3 * (rc - rs) ** 2);
    return rc;
  }

  protected pairKernel(t: number, rsq: number): { eng: number; fpairBase: number } {
    if (rsq <= this.rs2[t]) {
      const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
      return {
        fpairBase: r6inv * (this.lj1[t] * r6inv - this.lj2[t]) * r2inv,
        eng: r6inv * (this.lj3[t] * r6inv - this.lj4[t]),
      };
    }
    const r = Math.sqrt(rsq), d = r - this.rsTab[t], d2 = d * d;
    return {
      // E = u(r_s) + d u'(r_s) - (A3/6) d^3 ; F_r = -u'(r_s) + (A3/2) d^2
      fpairBase: (-this.ud[t] + 3 * this.wc[t] * d2) / r,
      eng: this.us[t] + d * (this.ud[t] - this.wc[t] * d2),
    };
  }

  tailSums(): { etail: number; ptail: number } {
    throw new StyleError('pair_modify tail yes is not supported for pair style lj/cubic (there are no corrections for a potential that goes to 0.0 at the cutoff)');
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(this.coeffLine([i], ['epsilon', 'sigma']));
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) out.push(this.coeffLine([i, j], ['epsilon', 'sigma']));
    }
    return out;
  }
}

/*
 * pair_style mie/cut cutoff — docs.lammps.org/pair_mie.html:
 * "The mie/cut style computes the Mie potential, given by"
 *
 *   E =  C \epsilon \left[ \left(\frac{\sigma}{r}\right)^{\gamma_{rep}} - \left(\frac{\sigma}{r}\right)^{\gamma_{att}} \right]
 *                         \qquad r < r_c
 *
 * ":math:`r_c` is the cutoff and C is a function that depends on the repulsive and
 * attractive exponents, given by:"
 *
 *   C = \left(\frac{\gamma_{rep}}{\gamma_{rep}-\gamma_{att}}\right) \left(\frac{\gamma_{rep}}{\gamma_{att}}\right)^{\left(\frac{\gamma_{att}}{\gamma_{rep}-\gamma_{att}}\right)}
 *
 * "Note that for 12/6 exponents, C is equal to 4 and the formula is the same as
 * the standard Lennard-Jones potential."
 * Coefficients:
 *
 * * epsilon (energy units)
 * * sigma (distance units)
 * * gammaR
 * * gammaA
 * * cutoff (distance units)
 *
 * "The last coefficient is optional.  If not specified, the global cutoff specified
 * in the pair_style command is used."
 * Mixing: "For atom type pairs I,J and I != J, the epsilon and sigma coefficients
 * and cutoff distance for all of the mie/cut pair styles can be mixed.
 * If not explicitly defined, both the repulsive and attractive gamma
 * exponents for different atoms will be calculated following the same
 * mixing rule defined for distances.  The default mix value is
 * geometric".
 * "This pair style supports the pair_modify shift option for the energy of the pair
 * interaction" (the offset C eps [(sigma/rc)^gammaR - (sigma/rc)^gammaA] makes
 * E(rc) = 0) "and ... the pair_modify tail option". Tail integrals (Sun,
 * equation 5) of E = C eps (s^gr r^-gr - s^ga r^-ga):
 *   int_rc^inf u r^2 dr  = C eps (s^gr rc^(3-gr)/(gr-3) - s^ga rc^(3-ga)/(ga-3))
 *   int_rc^inf r^3 u' dr = C eps (ga s^ga rc^(3-ga)/(ga-3) - gr s^gr rc^(3-gr)/(gr-3))
 * with s^g = sigma^g.
 */
export class PairMieCut extends PairLJ2Variant {
  readonly name: string = 'mie/cut';
  cutGlobal = 0;
  protected readonly paramNames = ['epsilon', 'sigma', 'gammaR', 'gammaA', 'cut'] as const;
  protected readonly coeffUsage = 'usage: pair_coeff I J epsilon sigma gammaR gammaA [cutoff]';
  cRep = new Float64Array(0);
  cAtt = new Float64Array(0);
  gRep = new Float64Array(0);
  gAtt = new Float64Array(0);
  offset = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError(`usage: pair_style ${this.name} cutoff`);
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  coeff(args: string[]): void {
    if (args.length !== 6 && args.length !== 7) throw new StyleError(this.coeffUsage);
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const gr = parseNum(args[4], 'gammaR');
    const ga = parseNum(args[5], 'gammaA');
    const cut = args[6] !== undefined ? parseNum(args[6], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, gr, ga, cut]);
  }

  protected ensureTables(nt: number): void {
    if (this.cRep.length !== nt * nt) {
      this.cRep = new Float64Array(nt * nt); this.cAtt = new Float64Array(nt * nt);
      this.gRep = new Float64Array(nt * nt); this.gAtt = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
  }

  protected initKernel(k1: number, k2: number, i: number, j: number): number {
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
    const gr = this.p.get('gammaR', i, j), ga = this.p.get('gammaA', i, j), cut = this.p.get('cut', i, j);
    // C = (gr/(gr-ga)) (gr/ga)^(ga/(gr-ga)) needs gr > ga (positive well depth)
    if (!(gr > ga)) throw new StyleError(`mie/cut requires gammaR > gammaA (got ${gr} and ${ga})`);
    const c = (gr / (gr - ga)) * (gr / ga) ** (ga / (gr - ga));
    // E = c eps (sig^gr r^-gr - sig^ga r^-ga) ; F_r = gr cRep r^-(gr+1) - ga cAtt r^-(ga+1)
    this.cRep[k1] = this.cRep[k2] = c * eps * sig ** gr;
    this.cAtt[k1] = this.cAtt[k2] = c * eps * sig ** ga;
    this.gRep[k1] = this.gRep[k2] = gr;
    this.gAtt[k1] = this.gAtt[k2] = ga;
    this.offset[k1] = this.offset[k2] = this.shift && cut > 0 ? c * eps * ((sig / cut) ** gr - (sig / cut) ** ga) : 0;
    return cut;
  }

  protected pairKernel(t: number, rsq: number): { eng: number; fpairBase: number } {
    const rinv = 1 / Math.sqrt(rsq);
    const rep = rinv ** this.gRep[t], att = rinv ** this.gAtt[t];
    return {
      // fpairBase = F_r / r = gr cRep r^-(gr+2) - ga cAtt r^-(ga+2)
      fpairBase: rinv * rinv * (this.gRep[t] * this.cRep[t] * rep - this.gAtt[t] * this.cAtt[t] * att),
      eng: this.cRep[t] * rep - this.cAtt[t] * att - this.offset[t],
    };
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    let e = 0, pr = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
        const gr = this.p.get('gammaR', i, j), ga = this.p.get('gammaA', i, j), rc = this.cut[i * nt + j];
        if (!(rc > 0)) continue;
        const c = (gr / (gr - ga)) * (gr / ga) ** (ga / (gr - ga));
        const sgr = c * eps * sig ** gr, sga = c * eps * sig ** ga;
        const nn = count[i] * count[j];
        // int_rc^inf u r^2 dr = C eps (s^gr rc^(3-gr)/(gr-3) - s^ga rc^(3-ga)/(ga-3))
        e += nn * (sgr * rc ** (3 - gr) / (gr - 3) - sga * rc ** (3 - ga) / (ga - 3));
        // int_rc^inf r^3 u' dr = C eps (ga s^ga rc^(3-ga)/(ga-3) - gr s^gr rc^(3-gr)/(gr-3))
        pr += nn * (ga * sga * rc ** (3 - ga) / (ga - 3) - gr * sgr * rc ** (3 - gr) / (gr - 3));
      }
    }
    // caller divides by V (energy) and V^2 (pressure)
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(this.coeffLine([i], ['epsilon', 'sigma', 'gammaR', 'gammaA']));
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) out.push(this.coeffLine([i, j], ['epsilon', 'sigma', 'gammaR', 'gammaA', 'cut']));
    }
    return out;
  }
}
