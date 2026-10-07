import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { fmtCoeff, parseNum } from '../util';

/*
 * LJ-family pair style variants, written from the LAMMPS documentation only.
 * All four share the loop shape of PairLJCut (docs.lammps.org/pair_lj.html):
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
 * Shared frame for the LJ variants. fpairBase is the radial force divided by
 * r (the "pair force magnitude / r" convention of PairLJCut: fx = dx * fpair);
 * eng is the pair energy including the shift offset, both BEFORE the
 * special_bonds factor.
 */
export abstract class PairLJVariant extends Pair {
  virialFdotr = true;
  p!: PairParams;
  /** Coefficient names in the documented pair_coeff order (after I J). */
  protected abstract readonly paramNames: readonly string[];
  /** Params mixed with the pair_modify rule ("the cutoff distance is mixed the same way as sigma"). */
  protected abstract readonly mixByRule: readonly string[];
  /** Params always mixed arithmetically whatever the pair_modify rule. */
  protected abstract readonly mixArithmetic: readonly string[];
  protected abstract readonly coeffUsage: string;

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, this.paramNames);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      for (const name of this.mixByRule) {
        const v = name === 'epsilon'
          ? mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j))
          : mixDistance(this.mix, p.get(name, i, i), p.get(name, j, j));
        p.setMixed(i, j, name, v);
      }
      for (const name of this.mixArithmetic) {
        p.setMixed(i, j, name, mixDistance('arithmetic', p.get(name, i, i), p.get(name, j, j)));
      }
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

  extract(name: string): unknown {
    if (name === 'epsilon') return this.p.p('epsilon');
    if (name === 'sigma') return this.p.p('sigma');
    return undefined;
  }
}

/*
 * pair_style lj96/cut cutoff — docs.lammps.org/pair_lj96.html:
 * "The lj96/cut style compute a 9/6 Lennard-Jones potential, instead
 * of the standard 12/6 potential, given by"
 *
 *    E = 4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{9} -
 *    \left(\frac{\sigma}{r}\right)^6 \right]
 *                        \qquad r < r_c
 *
 * ":math:`r_c` is the cutoff."
 * Coefficients "defined for each pair of atoms types via the pair_coeff
 * command":
 *
 *    * :math:`\epsilon` (energy units)
 *    * :math:`\sigma` (distance units)
 *    * cutoff (distance units)
 *
 * "The last coefficient is optional.  If not specified, the global LJ
 * cutoff specified in the pair_style command is used."
 * Mixing: "For atom type pairs I,J and I != J, the epsilon and sigma coefficients
 * and cutoff distance for all of the lj/cut pair styles can be mixed."
 * (cutoff mixed like sigma, docs.lammps.org/pair_modify.html).
 * "This pair style supports the pair_modify shift option for the energy of the
 * pair interaction." and "supports the pair_modify tail option". With shift,
 * the offset 4 eps [(sigma/rc)^9 - (sigma/rc)^6] makes E(rc) = 0.
 */
export class PairLJ96Cut extends PairLJVariant {
  readonly name: string = 'lj96/cut';
  cutGlobal = 0;
  protected readonly paramNames = ['epsilon', 'sigma', 'cut'] as const;
  protected readonly mixByRule = ['epsilon', 'sigma', 'cut'] as const;
  protected readonly mixArithmetic = [] as const;
  protected readonly coeffUsage = 'usage: pair_coeff I J epsilon sigma [cutoff]';
  lj1 = new Float64Array(0);
  lj2 = new Float64Array(0);
  lj3 = new Float64Array(0);
  lj4 = new Float64Array(0);
  offset = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError(`usage: pair_style ${this.name} cutoff`);
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 4 || args.length > 5) throw new StyleError(this.coeffUsage);
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const cut = args[4] !== undefined ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, cut]);
  }

  protected ensureTables(nt: number): void {
    if (this.lj1.length !== nt * nt) {
      this.lj1 = new Float64Array(nt * nt); this.lj2 = new Float64Array(nt * nt);
      this.lj3 = new Float64Array(nt * nt); this.lj4 = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
  }

  protected initKernel(k1: number, k2: number, i: number, j: number): number {
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j), cut = this.p.get('cut', i, j);
    const s6 = sig ** 6, s9 = s6 * sig ** 3;
    // E = 4 eps (s9 r^-9 - s6 r^-6) ; F_r = -dE/dr = 36 eps s9 r^-10 - 24 eps s6 r^-7
    this.lj1[k1] = this.lj1[k2] = 36 * eps * s9;
    this.lj2[k1] = this.lj2[k2] = 24 * eps * s6;
    this.lj3[k1] = this.lj3[k2] = 4 * eps * s9;
    this.lj4[k1] = this.lj4[k2] = 4 * eps * s6;
    this.offset[k1] = this.offset[k2] = this.shift && cut > 0 ? 4 * eps * ((sig / cut) ** 9 - (sig / cut) ** 6) : 0;
    return cut;
  }

  protected pairKernel(t: number, rsq: number): { eng: number; fpairBase: number } {
    const rinv = 1 / Math.sqrt(rsq);
    const r2inv = rinv * rinv, r3inv = r2inv * rinv, r6inv = r3inv * r3inv, r9inv = r6inv * r3inv;
    return {
      // fpairBase = F_r / r = 36 eps s9 r^-11 - 24 eps s6 r^-8
      fpairBase: r2inv * (this.lj1[t] * r9inv - this.lj2[t] * r6inv),
      eng: r9inv * this.lj3[t] - r6inv * this.lj4[t] - this.offset[t],
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
        // int_rc^inf u(r) r^2 dr = 4 eps (s9/(6 rc^6) - s6/(3 rc^3))
        e += nn * 4 * eps * (s9 / (6 * rc6) - s6 / (3 * rc3));
        // int_rc^inf r^3 u'(r) dr = 4 eps (-9 s9/(6 rc^6) + 2 s6/rc^3)
        pr += nn * 4 * eps * (-9 * s9 / (6 * rc6) + 2 * s6 / rc3);
      }
    }
    // caller divides by V (energy) and V^2 (pressure)
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('epsilon', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }
}

/*
 * pair_style lj/expand cutoff — docs.lammps.org/pair_lj_expand.html:
 * "Style lj/expand computes a LJ interaction with a distance shifted by
 * delta which can be useful when particles are of different sizes, since
 * it is different that using different sigma values in a standard LJ
 * formula:"
 *
 *    E = 4 \epsilon \left[ \left(\frac{\sigma}{r - \Delta}\right)^{12} -
 *      \left(\frac{\sigma}{r - \Delta}\right)^6 \right]
 *      \qquad r < r_c + \Delta
 *
 * ":math:`r_c` is the cutoff which does not include the :math:`\Delta`
 * distance.  I.e. the actual force cutoff is the sum of :math:`r_c +
 * \Delta`."
 * Coefficients:
 *
 *    * :math:`\epsilon` (energy units)
 *    * :math:`\sigma` (distance units)
 *    * :math:`\Delta` (distance units)
 *    * cutoff (distance units)
 *
 * "The :math:`\Delta` values can be positive or negative.  The last
 * coefficient is optional.  If not specified, the global LJ cutoff is
 * used."
 * Mixing: "For atom type pairs I,J and I != J, the epsilon, sigma, and shift
 * coefficients and cutoff distance for this pair style can be mixed.
 * Shift is always mixed via an arithmetic rule.  The other
 * coefficients are mixed according to the pair_modify mix value." (the
 * cutoff mixes like sigma, docs.lammps.org/pair_modify.html).
 * "This pair style supports the pair_modify shift option" (offset
 * 4 eps [(sigma/rc)^12 - (sigma/rc)^6] at the pair cutoff rc) and "This
 * pair style supports the pair_modify tail option". Tail integrals (Sun formulas) with
 * t = r - Delta running from t = rc (at r = rc + Delta, where u = 0) to
 * infinity, r = t + Delta, in closed form:
 *   int u r^2 dr  = 4 eps [ s12 (t^-9/9 + dt t^-10/5 + d2 t^-11/11)
 *                         - s6  (t^-3/3 + dt t^-4/2 + d2 t^-5/5) ]
 *   int r^3 u' dr = 4 eps [ -12 s12 (t^-9/9 + 3dt t^-10/10 + 3d2 t^-11/11 + d3 t^-12/12)
 *                         +   6 s6  (t^-3/3 + 3dt t^-4/4 + 3d2 t^-5/5 + d3 t^-6/6) ]
 * with dt = Delta, d2 = Delta^2, d3 = Delta^3, all at t = rc.
 */
export class PairLJExpand extends PairLJVariant {
  readonly name: string = 'lj/expand';
  cutGlobal = 0;
  protected readonly paramNames = ['epsilon', 'sigma', 'delta', 'cut'] as const;
  protected readonly mixByRule = ['epsilon', 'sigma', 'cut'] as const;
  protected readonly mixArithmetic = ['delta'] as const;
  protected readonly coeffUsage = 'usage: pair_coeff I J epsilon sigma delta [cutoff]';
  lj1 = new Float64Array(0);
  lj2 = new Float64Array(0);
  lj3 = new Float64Array(0);
  lj4 = new Float64Array(0);
  delta = new Float64Array(0);
  offset = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError(`usage: pair_style ${this.name} cutoff`);
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 5 || args.length > 6) throw new StyleError(this.coeffUsage);
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const delta = parseNum(args[4], 'delta');
    const cut = args[5] !== undefined ? parseNum(args[5], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, delta, cut]);
  }

  protected ensureTables(nt: number): void {
    if (this.lj1.length !== nt * nt) {
      this.lj1 = new Float64Array(nt * nt); this.lj2 = new Float64Array(nt * nt);
      this.lj3 = new Float64Array(nt * nt); this.lj4 = new Float64Array(nt * nt);
      this.delta = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
  }

  protected initKernel(k1: number, k2: number, i: number, j: number): number {
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
    const delta = this.p.get('delta', i, j), cut = this.p.get('cut', i, j);
    if (!(cut + delta > 0)) throw new StyleError(`lj/expand: cutoff + delta must be > 0 (got ${cut} + ${delta})`);
    const s6 = sig ** 6, s12 = s6 * s6;
    // E = 4 eps (s12 u^-12 - s6 u^-6) with u = r - delta ; F_r = 48 eps s12 u^-13 - 24 eps s6 u^-7
    this.lj1[k1] = this.lj1[k2] = 48 * eps * s12;
    this.lj2[k1] = this.lj2[k2] = 24 * eps * s6;
    this.lj3[k1] = this.lj3[k2] = 4 * eps * s12;
    this.lj4[k1] = this.lj4[k2] = 4 * eps * s6;
    this.delta[k1] = this.delta[k2] = delta;
    this.offset[k1] = this.offset[k2] = this.shift && cut > 0 ? 4 * eps * ((sig / cut) ** 12 - (sig / cut) ** 6) : 0;
    // "the actual force cutoff is the sum of r_c + Delta"
    return cut + delta;
  }

  protected pairKernel(t: number, rsq: number): { eng: number; fpairBase: number } {
    const r = Math.sqrt(rsq);
    const u = r - this.delta[t];
    if (u <= 0) throw new StyleError(`lj/expand: pair distance ${r} <= delta ${this.delta[t]} (potential undefined for r <= delta)`);
    const rinv = 1 / r, u2inv = 1 / (u * u), u6inv = u2inv * u2inv * u2inv;
    return {
      // fpairBase = F_r / r = (1/r) (1/u) (48 eps s12 u^-12 - 24 eps s6 u^-6)
      fpairBase: (rinv / u) * u6inv * (this.lj1[t] * u6inv - this.lj2[t]),
      eng: u6inv * (this.lj3[t] * u6inv - this.lj4[t]) - this.offset[t],
    };
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    let e = 0, pr = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
        const dl = this.p.get('delta', i, j), rc = this.p.get('cut', i, j);
        if (!(rc + dl > 0)) continue;
        const s6 = sig ** 6, s12 = s6 * s6, rc2 = rc * rc, rc3 = rc2 * rc;
        const rc4 = rc2 * rc2, rc5 = rc3 * rc2, rc6 = rc3 * rc3;
        const rc9 = rc6 * rc3, rc10 = rc9 * rc, rc11 = rc10 * rc, rc12 = rc11 * rc;
        const nn = count[i] * count[j];
        // t = r - delta from t = rc to infinity (u(r) vanishes at r = rc + delta)
        const a12 = 1 / (9 * rc9) + dl / (5 * rc10) + dl * dl / (11 * rc11);
        const a6 = 1 / (3 * rc3) + dl / (2 * rc4) + dl * dl / (5 * rc5);
        // int u r^2 dr = 4 eps (s12 a12 - s6 a6)
        e += nn * 4 * eps * (s12 * a12 - s6 * a6);
        const p12 = 1 / (9 * rc9) + 3 * dl / (10 * rc10) + 3 * dl * dl / (11 * rc11) + dl ** 3 / (12 * rc12);
        const p6 = 1 / (3 * rc3) + 3 * dl / (4 * rc4) + 3 * dl * dl / (5 * rc5) + dl ** 3 / (6 * rc6);
        // int r^3 u' dr = 4 eps (-12 s12 p12 + 6 s6 p6)
        pr += nn * 4 * eps * (-12 * s12 * p12 + 6 * s6 * p6);
      }
    }
    // caller divides by V (energy) and V^2 (pressure)
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))} ${fmtCoeff(this.p.get('delta', i, i))}`);
    }
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('epsilon', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))} ${fmtCoeff(this.p.get('delta', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }
}

/*
 * pair_style lj/smooth Rin Rc — docs.lammps.org/pair_lj_smooth.html:
 * "Style lj/smooth computes a LJ interaction with a force smoothing
 * applied between the inner and outer cutoff."
 *
 *    E & =  4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
 *                          \left(\frac{\sigma}{r}\right)^6 \right]
 *                          \qquad r < r_{in} \\
 *    F & =  C_1 + C_2 (r - r_{in}) + C_3 (r - r_{in})^2 + C_4 (r - r_{in})^3
 *                      \qquad r_{in} < r < r_c
 *
 * "The polynomial coefficients C1, C2, C3, C4 are computed by LAMMPS to
 * cause the force to vary smoothly from the inner cutoff :math:`r_{in}` to the
 * outer cutoff :math:`r_c`." "At the inner cutoff the force and its first derivative
 * will match the non-smoothed LJ formula.  At the outer cutoff the force
 * and its first derivative will be 0.0.  The inner cutoff cannot be 0.0."
 * Coefficients:
 *
 *    * :math:`\epsilon` (energy units)
 *    * :math:`\sigma` (distance units)
 *    * :math:`r_{in}` (distance units)
 *    * :math:`r_c` (distance units)
 *
 * "The last 2 coefficients are optional inner and outer cutoffs.  If not
 * specified, the global values for :math:`r_{in}` and :math:`r_c` are used."
 * Mixing: "For atom type pairs I,J and I != J, the epsilon, sigma, Rin
 * coefficients and the cutoff distance for this pair style can be mixed.
 * Rin is a cutoff value and is mixed like the cutoff.  The other
 * coefficients are mixed according to the pair_modify mix option."
 * "This pair style supports the pair_modify shift option for the energy of
 * the pair interaction." — the shift removes the documented energy value
 * discontinuity at the cutoff. "This pair style does not support the
 * pair_modify tail option ..., since the energy of the pair interaction is
 * smoothed to 0.0 at the cutoff."
 *
 * The closed forms of C1..C4 are not in the doc page (they live in the
 * cited paper, Leoni et al., PRL 134, 128201 (2025)); they are derived
 * here from the doc's four boundary conditions on the cubic
 * F(s) = C1 + C2 s + C3 s^2 + C4 s^3 (s = r - r_in, h = r_c - r_in), with
 * the non-smoothed LJ radial force F_LJ(r) = -dE/dr
 * = 48 eps s12 r^-13 - 24 eps s6 r^-7 and derivative
 * F'_LJ(r) = -624 eps s12 r^-14 + 168 eps s6 r^-8:
 *   C1 = F_LJ(r_in)         ("the force ... will match the non-smoothed LJ formula")
 *   C2 = F'_LJ(r_in)        ("and its first derivative will match")
 *   F(r_c) = 0, F'(r_c) = 0 ("the force and its first derivative will be 0.0")
 *   => C3 = -(3 C1 + 2 C2 h) / h^2 ; C4 = (2 C1 + C2 h) / h^3
 * The smooth-branch energy integrates -F with E continuous at r_in:
 *   E(r) = E_LJ(r_in) - [C1 s + C2 s^2/2 + C3 s^3/3 + C4 s^4/4]
 * which at r_c gives the documented value discontinuity
 * E(r_c) = E_LJ(r_in) - I with I = C1 h + C2 h^2/2 + C3 h^3/3 + C4 h^4/4;
 * pair_modify shift subtracts that constant (offset = E(r_c), E(r_c) -> 0).
 */
export class PairLJSmooth extends PairLJVariant {
  readonly name: string = 'lj/smooth';
  rinGlobal = 0;
  rcGlobal = 0;
  protected readonly paramNames = ['epsilon', 'sigma', 'rin', 'cut'] as const;
  protected readonly mixByRule = ['epsilon', 'sigma', 'rin', 'cut'] as const;
  protected readonly mixArithmetic = [] as const;
  protected readonly coeffUsage = 'usage: pair_coeff I J epsilon sigma [rin rc]';
  lj1 = new Float64Array(0);
  lj2 = new Float64Array(0);
  lj3 = new Float64Array(0);
  lj4 = new Float64Array(0);
  c1 = new Float64Array(0);
  c2 = new Float64Array(0);
  c3 = new Float64Array(0);
  c4 = new Float64Array(0);
  e0 = new Float64Array(0);
  rinTab = new Float64Array(0);
  offset = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 2) throw new StyleError(`usage: pair_style ${this.name} Rin Rc`);
    this.rinGlobal = parseNum(args[0], 'Rin');
    this.rcGlobal = parseNum(args[1], 'Rc');
    if (!(this.rinGlobal > 0)) throw new StyleError('The inner cutoff cannot be 0.0');
    if (!(this.rcGlobal > this.rinGlobal)) throw new StyleError(`inner cutoff ${this.rinGlobal} must be less than outer cutoff ${this.rcGlobal}`);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length !== 4 && args.length !== 6) throw new StyleError(this.coeffUsage);
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const rin = args[4] !== undefined ? parseNum(args[4], 'rin') : this.rinGlobal;
    const cut = args[5] !== undefined ? parseNum(args[5], 'cutoff') : this.rcGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, rin, cut]);
  }

  protected ensureTables(nt: number): void {
    if (this.lj1.length !== nt * nt) {
      this.lj1 = new Float64Array(nt * nt); this.lj2 = new Float64Array(nt * nt);
      this.lj3 = new Float64Array(nt * nt); this.lj4 = new Float64Array(nt * nt);
      this.c1 = new Float64Array(nt * nt); this.c2 = new Float64Array(nt * nt);
      this.c3 = new Float64Array(nt * nt); this.c4 = new Float64Array(nt * nt);
      this.e0 = new Float64Array(nt * nt);
      this.rinTab = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
  }

  protected initKernel(k1: number, k2: number, i: number, j: number): number {
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
    const rin = this.p.get('rin', i, j), cut = this.p.get('cut', i, j);
    if (!(rin > 0)) throw new StyleError('The inner cutoff cannot be 0.0');
    if (!(cut > rin)) throw new StyleError(`lj/smooth: inner cutoff ${rin} must be less than outer cutoff ${cut}`);
    const s6 = sig ** 6, s12 = s6 * s6;
    // non-smoothed LJ branch (r < rin), same tables as lj/cut
    this.lj1[k1] = this.lj1[k2] = 48 * eps * s12;
    this.lj2[k1] = this.lj2[k2] = 24 * eps * s6;
    this.lj3[k1] = this.lj3[k2] = 4 * eps * s12;
    this.lj4[k1] = this.lj4[k2] = 4 * eps * s6;
    // cubic force smoothing between rin and cut (boundary conditions from the doc, see header)
    const f0 = 48 * eps * s12 / rin ** 13 - 24 * eps * s6 / rin ** 7;      // F_LJ(rin)
    const d0 = -624 * eps * s12 / rin ** 14 + 168 * eps * s6 / rin ** 8;   // F'_LJ(rin)
    const h = cut - rin;
    this.c1[k1] = this.c1[k2] = f0;
    this.c2[k1] = this.c2[k2] = d0;
    this.c3[k1] = this.c3[k2] = -(3 * f0 + 2 * d0 * h) / (h * h);
    this.c4[k1] = this.c4[k2] = (2 * f0 + d0 * h) / (h * h * h);
    this.e0[k1] = this.e0[k2] = 4 * eps * (s12 / rin ** 12 - s6 / rin ** 6);
    this.rinTab[k1] = this.rinTab[k2] = rin;
    // shift: E(cut) = e0 - I must be 0, with I = C1 h + C2 h^2/2 + C3 h^3/3 + C4 h^4/4
    const c1 = this.c1[k1], c2 = this.c2[k1], c3 = this.c3[k1], c4 = this.c4[k1];
    const intF = c1 * h + c2 * h * h / 2 + c3 * h ** 3 / 3 + c4 * h ** 4 / 4;
    this.offset[k1] = this.offset[k2] = this.shift ? this.e0[k1] - intF : 0;
    return cut;
  }

  protected pairKernel(t: number, rsq: number): { eng: number; fpairBase: number } {
    const r = Math.sqrt(rsq);
    if (r < this.rinTab[t]) {
      const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
      return {
        fpairBase: r6inv * (this.lj1[t] * r6inv - this.lj2[t]) * r2inv,
        eng: r6inv * (this.lj3[t] * r6inv - this.lj4[t]) - this.offset[t],
      };
    }
    const s = r - this.rinTab[t];
    const f = this.c1[t] + s * (this.c2[t] + s * (this.c3[t] + s * this.c4[t]));
    const e = this.e0[t] - s * (this.c1[t] + s * (this.c2[t] / 2 + s * (this.c3[t] / 3 + s * this.c4[t] / 4)));
    return { fpairBase: f / r, eng: e - this.offset[t] };
  }

  init(ctx: StyleContext): void {
    // docs.lammps.org/pair_lj_smooth.html: "This pair style does not support the
    // pair_modify tail option for adding long-range tail corrections to energy and
    // pressure, since the energy of the pair interaction is smoothed to 0.0
    // at the cutoff."
    if (this.tail) throw new StyleError('pair_modify tail yes is not supported for pair style lj/smooth (the energy is smoothed to 0.0 at the cutoff)');
    super.init(ctx);
  }

  tailSums(): { etail: number; ptail: number } {
    throw new StyleError('pair_modify tail yes is not supported for pair style lj/smooth (the energy is smoothed to 0.0 at the cutoff)');
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))} ${fmtCoeff(this.p.get('rin', i, i))}`);
    }
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('epsilon', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))} ${fmtCoeff(this.p.get('rin', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }
}

/*
 * pair_style lj/smooth/linear cutoff — docs.lammps.org/pair_lj_smooth_linear.html:
 * "Style lj/smooth/linear computes a truncated and force-shifted LJ
 * interaction (aka Shifted Force Lennard-Jones) that combines the
 * standard 12/6 Lennard-Jones function and subtracts a linear term based
 * on the cutoff distance, so that both, the potential and the force, go
 * continuously to zero at the cutoff" r_c (Toxvaerd):
 *
 *    \phi\left(r\right) & =  4 \epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} -
 *                        \left(\frac{\sigma}{r}\right)^6 \right] \\
 *    E\left(r\right) & =  \phi\left(r\right)  - \phi\left(r_c\right) - \left(r - r_c\right) \left.\frac{d\phi}{d r} \right|_{r=r_c}       \qquad r < r_c
 *
 * Coefficients:
 *
 *    * :math:`\epsilon` (energy units)
 *    * :math:`\sigma` (distance units)
 *    * cutoff (distance units)
 *
 * "The last coefficient is optional. If not specified, the global
 * LJ cutoff specified in the pair_style command is used."
 * Mixing: "For atom type pairs I,J and I != J, the epsilon and sigma coefficients
 * and cutoff distance can be mixed. The default mix value is geometric."
 * "This pair style does not support the pair_modify shift option for the
 * energy of the pair interaction, since it goes to 0.0 at the cutoff by
 * construction." and "does not support the pair_modify tail option for
 * adding long-range tail corrections to energy and pressure, since the
 * energy of the pair interaction is smoothed to 0.0 at the cutoff."
 */
export class PairLJSmoothLinear extends PairLJVariant {
  readonly name: string = 'lj/smooth/linear';
  cutGlobal = 0;
  protected readonly paramNames = ['epsilon', 'sigma', 'cut'] as const;
  protected readonly mixByRule = ['epsilon', 'sigma', 'cut'] as const;
  protected readonly mixArithmetic = [] as const;
  protected readonly coeffUsage = 'usage: pair_coeff I J epsilon sigma [cutoff]';
  lj1 = new Float64Array(0);
  lj2 = new Float64Array(0);
  lj3 = new Float64Array(0);
  lj4 = new Float64Array(0);
  phiRc = new Float64Array(0);
  dphiRc = new Float64Array(0);
  cutTab = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError(`usage: pair_style ${this.name} cutoff`);
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 4 || args.length > 5) throw new StyleError(this.coeffUsage);
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const cut = args[4] !== undefined ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, cut]);
  }

  init(ctx: StyleContext): void {
    // docs.lammps.org/pair_lj_smooth_linear.html: "This pair style does not support the
    // pair_modify shift option for the energy of the pair interaction, since it goes
    // to 0.0 at the cutoff by construction."
    if (this.shift) throw new StyleError('pair_modify shift yes is not supported for pair style lj/smooth/linear (it goes to 0.0 at the cutoff by construction)');
    super.init(ctx);
  }

  protected ensureTables(nt: number): void {
    if (this.lj1.length !== nt * nt) {
      this.lj1 = new Float64Array(nt * nt); this.lj2 = new Float64Array(nt * nt);
      this.lj3 = new Float64Array(nt * nt); this.lj4 = new Float64Array(nt * nt);
      this.phiRc = new Float64Array(nt * nt);
      this.dphiRc = new Float64Array(nt * nt);
      this.cutTab = new Float64Array(nt * nt);
    }
  }

  protected initKernel(k1: number, k2: number, i: number, j: number): number {
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j), cut = this.p.get('cut', i, j);
    const s6 = sig ** 6, s12 = s6 * s6;
    this.lj1[k1] = this.lj1[k2] = 48 * eps * s12;
    this.lj2[k1] = this.lj2[k2] = 24 * eps * s6;
    this.lj3[k1] = this.lj3[k2] = 4 * eps * s12;
    this.lj4[k1] = this.lj4[k2] = 4 * eps * s6;
    // phi(rc) and dphi/dr(rc): E(r) = phi(r) - phi(rc) - (r - rc) * dphi(rc)
    this.phiRc[k1] = this.phiRc[k2] = 4 * eps * (s12 / cut ** 12 - s6 / cut ** 6);
    this.dphiRc[k1] = this.dphiRc[k2] = -48 * eps * s12 / cut ** 13 + 24 * eps * s6 / cut ** 7;
    this.cutTab[k1] = this.cutTab[k2] = cut;
    return cut;
  }

  protected pairKernel(t: number, rsq: number): { eng: number; fpairBase: number } {
    const r = Math.sqrt(rsq);
    const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
    const phi = r6inv * (this.lj3[t] * r6inv - this.lj4[t]);
    return {
      // F_r = -E'(r) = -phi'(r) + phi'(rc) = F_LJ(r) + dphiRc ; fpairBase = F_r / r
      fpairBase: r6inv * (this.lj1[t] * r6inv - this.lj2[t]) * r2inv + this.dphiRc[t] / r,
      eng: phi - this.phiRc[t] - (r - this.cutTab[t]) * this.dphiRc[t],
    };
  }

  tailSums(): { etail: number; ptail: number } {
    throw new StyleError('pair_modify tail yes is not supported for pair style lj/smooth/linear (the energy is smoothed to 0.0 at the cutoff)');
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('epsilon', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }
}
