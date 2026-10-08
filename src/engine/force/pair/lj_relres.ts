import { PairParams, StyleError, mixDistance, mixEpsilon, type StyleContext } from '../types';
import { fmtCoeff, parseNum } from '../util';
import { PairLJVariant } from './lj_variants';

/*
 * pair_style lj/relres Rsi Rso Rci Rco — docs.lammps.org/pair_lj_relres.html.
 * The page writes the piecewise energy with an FG LJ term below r_si, a
 * quartic switch on [r_si, r_so), a CG LJ term on [r_so, r_ci), a quartic
 * cut on [r_ci, r_co) and E = 0 beyond r_co, with offsets Gamma_si, Gamma_so
 * and Gamma_c (the page's piecewise formula is the E(r) quoted in the rst).
 * Quotes from the page (each a verbatim line):
 *   "Pair style *lj/relres* computes a LJ interaction using the Relative"
 *   "The FG parameters of the LJ potential (:math:`\epsilon^{FG}` and" apply up to
 *   "the inner switching cutoff"; the CG parameters apply beyond the outer one.
 *   "force and its derivative are continuous between the FG and CG"
 *   "The corresponding polynomial coefficients :math:`\gamma_{sm}`" and the offsets
 *   are "automatically" "computed by LAMMPS." (the page gives no formula for them)
 *   "ordinary sites, :math:`\epsilon^{CG}` must be set to 0 (zero) while the"
 *   "If this override option is employed, all four"
 *   "arguments must be specified."
 *   "All parameters are mixed according to the"
 *   "The default mix value is *geometric*,"
 *   "This pair style supports the :doc:`pair_modify <pair_modify>` shift"
 *   "Otherwise, the offset :math:`\Gamma_{c}`"
 *   "is set to zero. Constants :math:`\Gamma_{si}` and :math:`\Gamma_{so}` are"
 *   "This pair style does not support the :doc:`pair_modify <pair_modify>`"
 *   "pressure, since the energy of the pair interaction is smoothed to 0.0"
 *   "smoothing can be eliminated by setting"
 *
 * The coefficients gamma_sm, gamma_cm and the offsets are therefore fixed
 * against native LAMMPS (black box, pair_write and write_data output):
 *   - Measured with native LAMMPS (black box): the switching polynomial matches
 *     E' and E'' to the FG potential at r_si, and E' to the CG potential at
 *     r_so; its second derivative at r_so equals minus the CG second derivative,
 *     so the force derivative is not continuous at r_so.
 *   - Measured with native LAMMPS (black box): the cut polynomial matches E,
 *     E' and E'' to the CG potential at r_ci and has zero E' and E'' at r_co.
 *   - Measured with native LAMMPS (black box): with pair_modify shift yes the
 *     energy vanishes at r_co (Gamma_c is the cut polynomial's value at r_co);
 *     with shift no Gamma_c = 0. Gamma_so and Gamma_si follow from energy
 *     continuity, whatever the shift setting.
 *   - Measured with native LAMMPS (black box): with r_si = r_so the FG branch
 *     joins the CG branch directly; with r_ci = r_co the cut is a plain CG step.
 *   - Measured with native LAMMPS (black box): a pair with eps^CG = 0 (ordinary
 *     site) is written with r_ci = r_co = r_so. Its energy is zero beyond r_so
 *     either way, so only the neighbour cutoff changes.
 *   - Measured with native LAMMPS (black box): the cutoffs must satisfy
 *     0 < Rsi <= Rso <= Rci <= Rco; equal neighbours are accepted.
 */

/** Per type-pair table: FG/CG LJ constants, switching polynomials and offsets. */
interface RelResTab {
  eFG: number; s6FG: number; s12FG: number;
  eCG: number; s6CG: number; s12CG: number;
  rsi: number; rso: number; rci: number; rco: number;
  /** Switching polynomial in s = r - rsi (gs[0] = 0 by construction). */
  gs: Float64Array;
  /** Cut polynomial in s = r - rci. */
  gc: Float64Array;
  offSi: number; offSo: number; offC: number;
}

/** Radii check shared by pair_style and pair_coeff (docs.lammps.org/pair_lj_relres.html). */
const checkRadii = (rsi: number, rso: number, rci: number, rco: number): void => {
  if (!(rsi > 0)) throw new StyleError('lj/relres: the inner switching cutoff Rsi must be greater than 0');
  if (!(rsi <= rso && rso <= rci && rci <= rco)) {
    throw new StyleError(`lj/relres: cutoffs must satisfy Rsi <= Rso <= Rci <= Rco (got ${rsi} ${rso} ${rci} ${rco})`);
  }
};

/** Ordinary sites (eps^CG = 0) use r_ci = r_co = r_so (measured with native LAMMPS, black box). */
const effectiveRadii = (eCG: number, rsi: number, rso: number, rci: number, rco: number): [number, number, number, number] =>
  eCG === 0 ? [rsi, rso, rso, rso] : [rsi, rso, rci, rco];

// Lennard-Jones energy and its first and second r-derivatives (textbook form of 4 eps [(s/r)^12 - (s/r)^6]).
const ljE = (e: number, s6: number, s12: number, r: number): number => {
  const r6 = r ** 6;
  return 4 * e * (s12 / (r6 * r6) - s6 / r6);
};
const ljD1 = (e: number, s6: number, s12: number, r: number): number => {
  const r6 = r ** 6;
  return -48 * e * s12 / (r6 * r6 * r) + 24 * e * s6 / (r6 * r);
};
const ljD2 = (e: number, s6: number, s12: number, r: number): number => {
  const r6 = r ** 6;
  return 624 * e * s12 / (r6 * r6 * r * r) - 168 * e * s6 / (r6 * r * r);
};

/**
 * Solves for g3, g4 of a quartic whose first two derivatives at the end s = h are
 * given by the residuals A1 = P'(h) - (g1 + 2 g2 h) and A2 = P''(h) - 2 g2:
 *   3 g3 h^2 + 4 g4 h^3 = A1,   6 g3 h + 12 g4 h^2 = A2.
 */
const quarticTail = (h: number, A1: number, A2: number): [number, number] => {
  const g4 = (h * A2 / 2 - A1) / (2 * h * h * h);
  const g3 = (A2 - 12 * g4 * h * h) / (6 * h);
  return [g3, g4];
};

const polyVal = (g: Float64Array, s: number): number => g[0] + s * (g[1] + s * (g[2] + s * (g[3] + s * g[4])));
const polyD1 = (g: Float64Array, s: number): number => g[1] + s * (2 * g[2] + s * (3 * g[3] + s * 4 * g[4]));

/** Builds the switching and cut polynomials and the three offsets for one type pair. */
const buildTab = (
  eFG: number, sFG: number, eCG: number, sCG: number,
  rsi: number, rso: number, rci: number, rco: number, shift: boolean,
): RelResTab => {
  const s6FG = sFG ** 6, s12FG = s6FG * s6FG;
  const s6CG = sCG ** 6, s12CG = s6CG * s6CG;
  const hs = rso - rsi, hc = rco - rci;

  // Switching polynomial: P(0) = 0 (gamma_s0 cancels against Gamma_so), P' and P'' match FG at
  // r_si, P' matches CG at r_so, and P'' at r_so is minus the CG second derivative (measured).
  const gs = new Float64Array(5);
  if (hs > 0) {
    const g1 = ljD1(eFG, s6FG, s12FG, rsi);
    const g2 = ljD2(eFG, s6FG, s12FG, rsi) / 2;
    const A1 = ljD1(eCG, s6CG, s12CG, rso) - g1 - 2 * g2 * hs;
    const A2 = -ljD2(eCG, s6CG, s12CG, rso) - 2 * g2;
    const [g3, g4] = quarticTail(hs, A1, A2);
    gs[1] = g1; gs[2] = g2; gs[3] = g3; gs[4] = g4;
  }

  // Cut polynomial: Q(0) = CG(r_ci), Q' and Q'' match CG at r_ci, Q' and Q'' vanish at r_co.
  const gc = new Float64Array(5);
  const c0 = ljE(eCG, s6CG, s12CG, rci);
  if (hc > 0) {
    const q1 = ljD1(eCG, s6CG, s12CG, rci);
    const q2 = ljD2(eCG, s6CG, s12CG, rci) / 2;
    const [q3, q4] = quarticTail(hc, -q1 - 2 * q2 * hc, -2 * q2);
    gc[0] = c0; gc[1] = q1; gc[2] = q2; gc[3] = q3; gc[4] = q4;
  }

  // Offsets: Gamma_c makes E(r_co) = 0 with shift yes (and 0 otherwise); Gamma_so and Gamma_si
  // follow from continuity at r_so and r_si. With r_si = r_so the switching polynomial is empty and
  // the FG branch is joined to the CG branch directly.
  const endC = hc > 0 ? polyVal(gc, hc) : c0;
  const offC = shift ? endC : 0;
  const offSo = polyVal(gs, hs) - (ljE(eCG, s6CG, s12CG, rso) - offC);
  // continuity at r_si: E_FG(r_si) - Gamma_si = P(0) - Gamma_so
  const offSi = ljE(eFG, s6FG, s12FG, rsi) - polyVal(gs, 0) + offSo;

  return { eFG, s6FG, s12FG, eCG, s6CG, s12CG, rsi, rso, rci, rco, gs, gc, offSi, offSo, offC };
};

/**
 * pair_style lj/relres: FG LJ below r_si, polynomial switch to CG LJ on [r_si, r_so),
 * CG LJ on [r_so, r_ci), polynomial cut on [r_ci, r_co). Tail corrections are not supported.
 */
export class PairLJRelRes extends PairLJVariant {
  readonly name: string = 'lj/relres';
  protected readonly paramNames = ['epsilon', 'sigma', 'epsilon_cg', 'sigma_cg', 'rsi', 'rso', 'rci', 'rco'] as const;
  protected readonly mixByRule = [] as const;
  protected readonly mixArithmetic = [] as const;
  protected readonly coeffUsage = 'usage: pair_coeff I J epsilon^FG sigma^FG epsilon^CG sigma^CG [Rsi Rso Rci Rco]';
  rsiGlobal = 0;
  rsoGlobal = 0;
  rciGlobal = 0;
  rcoGlobal = 0;
  tab: RelResTab[] = [];

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 4) throw new StyleError('usage: pair_style lj/relres Rsi Rso Rci Rco');
    const [rsi, rso, rci, rco] = args.map((a, k) => parseNum(a, ['Rsi', 'Rso', 'Rci', 'Rco'][k]));
    checkRadii(rsi, rso, rci, rco);
    this.rsiGlobal = rsi; this.rsoGlobal = rso; this.rciGlobal = rci; this.rcoGlobal = rco;
  }

  coeff(args: string[], _ctx: StyleContext): void {
    // docs.lammps.org/pair_lj_relres.html: "If this override option is employed, all four
    // arguments must be specified."
    if (args.length !== 6 && args.length !== 10) throw new StyleError(this.coeffUsage);
    const eFG = parseNum(args[2], 'epsilon^FG');
    const sFG = parseNum(args[3], 'sigma^FG');
    const eCG = parseNum(args[4], 'epsilon^CG');
    const sCG = parseNum(args[5], 'sigma^CG');
    const radii = args.length === 10
      ? args.slice(6).map((a, k) => parseNum(a, ['Rsi', 'Rso', 'Rci', 'Rco'][k]))
      : [this.rsiGlobal, this.rsoGlobal, this.rciGlobal, this.rcoGlobal];
    checkRadii(radii[0], radii[1], radii[2], radii[3]);
    this.p.setRange(args[0], args[1], [eFG, sFG, eCG, sCG, radii[0], radii[1], radii[2], radii[3]]);
  }

  /**
   * Cross terms not set by pair_coeff are mixed for all eight parameters with the pair_modify mix
   * rule (docs.lammps.org/pair_lj_relres.html: "All parameters are mixed according to the
   * pair_modify mix option."). The CG epsilon uses the CG sigmas; a zero CG epsilon stays zero.
   */
  initOne(i: number, j: number): number {
    const p: PairParams = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      const g = (name: string, k: number) => p.get(name, k, k);
      const eFG = mixEpsilon(this.mix, g('epsilon', i), g('epsilon', j), g('sigma', i), g('sigma', j));
      const eCG = g('epsilon_cg', i) * g('epsilon_cg', j) === 0
        ? 0
        : mixEpsilon(this.mix, g('epsilon_cg', i), g('epsilon_cg', j), g('sigma_cg', i), g('sigma_cg', j));
      p.setMixed(i, j, 'epsilon', eFG);
      p.setMixed(i, j, 'sigma', mixDistance(this.mix, g('sigma', i), g('sigma', j)));
      p.setMixed(i, j, 'epsilon_cg', eCG);
      p.setMixed(i, j, 'sigma_cg', mixDistance(this.mix, g('sigma_cg', i), g('sigma_cg', j)));
      for (const name of ['rsi', 'rso', 'rci', 'rco']) {
        p.setMixed(i, j, name, mixDistance(this.mix, g(name, i), g(name, j)));
      }
    }
    const nt = this.ntypes + 1;
    this.ensureTables(nt);
    return this.initKernel(i * nt + j, j * nt + i, i, j);
  }

  protected ensureTables(nt: number): void {
    if (this.tab.length !== nt * nt) this.tab = new Array<RelResTab>(nt * nt);
  }

  protected initKernel(k1: number, k2: number, i: number, j: number): number {
    const p = this.p;
    const eCG = p.get('epsilon_cg', i, j);
    const [rsi, rso, rci, rco] = effectiveRadii(eCG, p.get('rsi', i, j), p.get('rso', i, j), p.get('rci', i, j), p.get('rco', i, j));
    checkRadii(rsi, rso, rci, rco);
    const tab = buildTab(
      p.get('epsilon', i, j), p.get('sigma', i, j), eCG, p.get('sigma_cg', i, j),
      rsi, rso, rci, rco, this.shift,
    );
    this.tab[k1] = this.tab[k2] = tab;
    return rco;
  }

  protected pairKernel(t: number, rsq: number): { eng: number; fpairBase: number } {
    const T = this.tab[t];
    const r = Math.sqrt(rsq);
    if (r >= T.rco) return { eng: 0, fpairBase: 0 };
    if (r < T.rsi) {
      const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv, r12inv = r6inv * r6inv;
      return {
        eng: 4 * T.eFG * (T.s12FG * r12inv - T.s6FG * r6inv) - T.offSi,
        fpairBase: r2inv * (48 * T.eFG * T.s12FG * r12inv - 24 * T.eFG * T.s6FG * r6inv),
      };
    }
    if (r < T.rso) {
      const s = r - T.rsi;
      return { eng: polyVal(T.gs, s) - T.offSo, fpairBase: -polyD1(T.gs, s) / r };
    }
    if (r < T.rci) {
      const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv, r12inv = r6inv * r6inv;
      return {
        eng: 4 * T.eCG * (T.s12CG * r12inv - T.s6CG * r6inv) - T.offC,
        fpairBase: r2inv * (48 * T.eCG * T.s12CG * r12inv - 24 * T.eCG * T.s6CG * r6inv),
      };
    }
    const s = r - T.rci;
    return { eng: polyVal(T.gc, s) - T.offC, fpairBase: -polyD1(T.gc, s) / r };
  }

  init(ctx: StyleContext): void {
    // docs.lammps.org/pair_lj_relres.html: "This pair style does not support the
    // pair_modify tail option for adding long-range tail corrections to energy and
    // pressure, since the energy of the pair interaction is smoothed to 0.0 at the cutoff."
    if (this.tail) throw new StyleError('pair_modify tail yes is not supported for pair style lj/relres (the energy of the pair interaction is smoothed to 0.0 at the cutoff)');
    super.init(ctx);
  }

  tailSums(): { etail: number; ptail: number } {
    throw new StyleError('pair_modify tail yes is not supported for pair style lj/relres (the energy of the pair interaction is smoothed to 0.0 at the cutoff)');
  }

  /** write_data "Pair Coeffs": epsilon^FG sigma^FG epsilon^CG sigma^CG (measured with native LAMMPS). */
  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))} ${fmtCoeff(this.p.get('epsilon_cg', i, i))} ${fmtCoeff(this.p.get('sigma_cg', i, i))}`);
    }
    return out;
  }

  /** write_data "PairIJ Coeffs": the eight values with the effective radii (measured with native LAMMPS). */
  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        const eCG = this.p.get('epsilon_cg', i, j);
        const [rsi, rso, rci, rco] = effectiveRadii(eCG, this.p.get('rsi', i, j), this.p.get('rso', i, j), this.p.get('rci', i, j), this.p.get('rco', i, j));
        out.push([
          i, j,
          fmtCoeff(this.p.get('epsilon', i, j)), fmtCoeff(this.p.get('sigma', i, j)),
          fmtCoeff(eCG), fmtCoeff(this.p.get('sigma_cg', i, j)),
          fmtCoeff(rsi), fmtCoeff(rso), fmtCoeff(rci), fmtCoeff(rco),
        ].join(' '));
      }
    }
    return out;
  }
}

