import { StyleError, type StyleContext } from '../types';
import { parseNum, joinPotentialEntries } from '../util';
import { PairTersoff, key3, type ParsedFile, type TersoffEntry } from './tersoff';

/*
 * pair_style tersoff/mod, tersoff/mod/c and tersoff/zbl: docs.lammps.org/pair_tersoff_mod.html
 * and docs.lammps.org/pair_tersoff_zbl.html (sources: plans/lammps-docs/pair_tersoff_mod.rst,
 * plans/lammps-docs/pair_tersoff_zbl.rst). All three reuse the bond-order machinery of
 * PairTersoff (pair_tersoff.ts) and replace only the formula pieces through its hooks
 * (cutFn, cutDeriv, angular, angularDeriv, zetaExp, zetaLogDeriv, bondOrder, bondOrderDeriv,
 * pairRadial, onPair, onTriplet, parseFile).
 *
 * tersoff/mod: the docs say "The *tersoff/mod* and *tersoff/mod/c* styles computes a bond-order
 * type interatomic potential" based on a Tersoff potential with a modified cutoff function
 * f_C (Murty form: 1/2 - 9/16 sin(pi/2 x) - 1/16 sin(3 pi/2 x), x = (r-R)/D inside R-D..R+D),
 * b_ij = (1 + zeta_ij^eta)^(-1/(2n)), zeta_ij = sum_k f_C(r_ik) g(theta_ijk) exp[alpha (r_ij - r_ik)^beta],
 * g(theta) = c1 + g_o(theta) g_a(theta) with g_o = c2 (h - cos theta)^2 / (c3 + (h - cos theta)^2)
 * and g_a = 1 + c4 exp[-c5 (h - cos theta)^2].
 * tersoff/mod/c: "The *tersoff/mod/c* style differs from *tersoff/mod* only in the formulation of the
 * V_ij term, where it contains an additional c0 term": V_ij = f_C [f_R + b_ij f_A + c0].
 * File columns, in the doc's list order: element 1, element 2, element 3, beta, alpha, h, eta,
 * beta_ters (dummy, 1), lambda2, B, R, D, lambda1, A, n, c1, c2, c3, c4, c5, and c0 for tersoff/mod/c.
 *
 * tersoff/zbl: V_ij = (1 - f_F) V^ZBL + f_F V^Tersoff with f_F = 1/(1 + exp(-A_F (r - r_C))); the ZBL
 * term is the Coulomb repulsion times the universal screening function phi(x) (four exponentials,
 * the constants of pair_tersoff_zbl.html), with screening length a = 0.8854 a_0 / (Z_i^0.23 + Z_j^0.23)
 * and a_0 = 0.529 Angstrom. The Tersoff part is the standard form (PairTersoff). File columns: the 14
 * Tersoff columns of pair_tersoff.html followed by Z_i, Z_j, ZBLcut (r_C) and ZBLexpscale (A_F).
 *
 * Shared behaviour: energies use the documented E = 1/2 sum_i sum_{j!=i} V_ij with the two-body
 * entry (i,j,j) and three-body entries (i,j,k) (see pair_tersoff.ts). The shift keyword replaces r by
 * r + delta in every term (measured with native LAMMPS for tersoff/zbl and tersoff/mod, see tests).
 */

const MOD_COLUMNS = ['beta', 'alpha', 'h', 'eta', 'betaTers', 'lambda2', 'B', 'R', 'D', 'lambda1', 'A', 'n', 'c1', 'c2', 'c3', 'c4', 'c5'];
const TERSOFF_ZBL_COLUMNS = ['m', 'gamma', 'lambda3', 'c', 'd', 'cos0', 'n', 'beta', 'lambda2', 'B', 'R', 'D', 'lambda1', 'A', 'Zi', 'Zj', 'ZBLcut', 'ZBLexpscale'];

/**
 * Parses a variant potential file: three element names followed by the named columns.
 * Entries may span several lines (joinPotentialEntries); a leading UNITS: tag line is read
 * as in the base style. Returns the named values of every entry in ext.
 */
const parseVariantFile = (
  text: string,
  fileName: string,
  columns: string[],
  build: (named: Record<string, number>, k: string, ln: number) => TersoffEntry,
): ParsedFile => {
  const ntok = columns.length + 3;
  const lines = joinPotentialEntries(text.split(/\r?\n/), ntok);
  let unitTag: string | null = null;
  let start = 0;
  if (lines.length > 0) {
    const m = /UNITS:\s*(\S+)/.exec(lines[0]);
    if (m) {
      unitTag = m[1];
      start = 1;
    }
  }
  const entries = new Map<string, TersoffEntry>();
  const elems = new Set<string>();
  for (let ln = start; ln < lines.length; ln++) {
    const raw = lines[ln];
    const hash = raw.indexOf('#');
    const line = (hash >= 0 ? raw.slice(0, hash) : raw).trim();
    if (line === '') continue;
    const t = line.split(/\s+/);
    if (t.length !== ntok) {
      throw new StyleError(
        `potential file ${fileName} line ${ln + 1}: expected 'element1 element2 element3 ${columns.join(' ')}' (${ntok} values), got ${t.length}`,
      );
    }
    const k = key3(t[0], t[1], t[2]);
    if (entries.has(k)) throw new StyleError(`potential file ${fileName} line ${ln + 1}: duplicate entry for elements ${k}`);
    const named: Record<string, number> = {};
    for (let c = 0; c < columns.length; c++) named[columns[c]] = parseNum(t[c + 3], `${columns[c]} of the potential file entry ${k}`);
    entries.set(k, build(named, k, ln));
    elems.add(t[0]);
    elems.add(t[1]);
    elems.add(t[2]);
  }
  if (entries.size === 0) throw new StyleError(`potential file ${fileName} contains no parameter entries`);
  return { entries, elems, unitTag };
};

/** Two-body and shared fields of a Tersoff-layout entry built from named columns. */
const baseEntry = (
  k: string, named: Record<string, number>,
  pick: { m: number; gamma: number; lambda3: number; c: number; d: number; cos0: number; n: number; beta: number },
): TersoffEntry => {
  const [e1, e2, e3] = k.split(' ');
  if (!(named.D > 0) || !(pick.n > 0)) throw new StyleError(`potential file entry ${k}: D and n must be > 0`);
  return {
    e1, e2, e3,
    m: pick.m, gamma: pick.gamma, lambda3: pick.lambda3, c: pick.c, d: pick.d, costheta0: pick.cos0,
    n: pick.n, beta: pick.beta,
    lambda2: named.lambda2, B: named.B, R: named.R, D: named.D, lambda1: named.lambda1, A: named.A,
    ext: { ...named },
  };
};

/** tersoff/mod: entries of the MOD layout (no m, gamma, lambda3, c, d, cos theta0 columns). */
const modEntry = (named: Record<string, number>, k: string): TersoffEntry =>
  baseEntry(k, named, { m: 1, gamma: 1, lambda3: 0, c: 0, d: 1, cos0: 0, n: named.n, beta: named.eta });

/** tersoff/zbl: entries of the Tersoff/ZBL layout. */
const zblEntry = (named: Record<string, number>, k: string): TersoffEntry => {
  if (named.m !== 3 && named.m !== 1) throw new StyleError(`potential file entry ${k}: m must be 3 or 1 (got ${named.m})`);
  return baseEntry(k, named, {
    m: named.m, gamma: named.gamma, lambda3: named.lambda3, c: named.c, d: named.d, cos0: named.cos0,
    n: named.n, beta: named.beta,
  });
};

/**
 * Refuses a "UNITS:" header in a tersoff/mod or tersoff/zbl potential file. Native LAMMPS converts
 * such files (warning printed) but, measured with a tersoff/mod file whose first entry is A A A,
 * then reports a missing entry, so no conversion is verified here.
 */
const refuseUnitsTag = (text: string, fileName: string, style: string): void => {
  const first = text.split(/\r?\n/)[0] ?? '';
  if (/UNITS:/.test(first)) {
    throw new StyleError(`potential file ${fileName} has a UNITS: header; ${style} does not convert potential units in this engine`);
  }
};

/**
 * Two-body entries (i,j,j) and (j,i,i) that differ: the documented rule uses one per ordered
 * pair. Measured with native LAMMPS (black box): with such entries native gives other energies
 * (a pair's repulsive term follows one orientation only), which this engine does not reproduce;
 * the warning makes that visible instead of silent.
 */
const warnAsymmetricTwoBody = (entries: Map<string, TersoffEntry>, elemNames: string[], fileName: string, ctx: StyleContext, style: string): void => {
  const key = (e: TersoffEntry): string => {
    const x = e.ext ?? {};
    return [e.lambda1, e.lambda2, e.A, e.B, e.R, e.D, e.n, e.beta, x.c0, x.ZBLcut, x.ZBLexpscale, x.Zi !== undefined ? x.Zi * x.Zj : undefined]
      .map((v) => String(v)).join(' ');
  };
  const names = elemNames;
  for (let a = 0; a < names.length; a++) {
    for (let b = a + 1; b < names.length; b++) {
      const e1 = entries.get(key3(names[a], names[b], names[b]));
      const e2 = entries.get(key3(names[b], names[a], names[a]));
      if (e1 && e2 && key(e1) !== key(e2)) {
        ctx.log(`WARNING: ${style} potential file ${fileName}: two-body entries ${names[a]} ${names[b]} ${names[b]} and ${names[b]} ${names[a]} ${names[a]} differ; LAMMPS gives other energies for such files (measured), this engine follows the documented rule`);
      }
    }
  }
};

/** tersoff/mod (docs.lammps.org/pair_tersoff_mod.html). */
export class PairTersoffMod extends PairTersoff {
  override readonly name: string = 'tersoff/mod';

  /** eta (the exponent of zeta in b_ij) per ordered two-body type index. */
  protected eta2 = new Float64Array(0);
  /** Per three-body type index t: [alpha, beta, h, c1, c2, c3, c4, c5] at 8*t. */
  protected p3 = new Float64Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.eta2 = new Float64Array(nt * nt);
    this.p3 = new Float64Array(nt * nt * nt * 8);
  }

  protected override parseFile(text: string, fileName: string): ParsedFile {
    return parseVariantFile(text, fileName, MOD_COLUMNS, (named, k) => modEntry(named, k));
  }

  protected override readFile(filename: string, ctx: StyleContext): void {
    refuseUnitsTag(ctx.readFile(filename), filename, this.name);
    super.readFile(filename, ctx);
  }

  override initStyle(ctx: StyleContext): void {
    super.initStyle(ctx);
    warnAsymmetricTwoBody(this.entries, this.elemNames, this.fileName, ctx, this.name);
  }

  protected override onPair(t: number, e: TersoffEntry): void {
    this.eta2[t] = e.ext!.eta;
  }

  protected override onTriplet(t: number, e: TersoffEntry): void {
    const x = e.ext!;
    const o = 8 * t;
    this.p3[o] = x.alpha;
    this.p3[o + 1] = x.beta;
    this.p3[o + 2] = x.h;
    this.p3[o + 3] = x.c1;
    this.p3[o + 4] = x.c2;
    this.p3[o + 5] = x.c3;
    this.p3[o + 6] = x.c4;
    this.p3[o + 7] = x.c5;
  }

  /** Murty cutoff: 1/2 - 9/16 sin(pi/2 x) - 1/16 sin(3 pi/2 x), x = (r-R)/D (pair_tersoff_mod.html). */
  protected override cutFn(r: number, R: number, D: number): number {
    if (r < R - D) return 1;
    if (r > R + D) return 0;
    const x = (r - R) / D;
    return 0.5 - (9 / 16) * Math.sin((Math.PI / 2) * x) - (1 / 16) * Math.sin((3 * Math.PI / 2) * x);
  }

  protected override cutDeriv(r: number, R: number, D: number): number {
    if (r <= R - D || r >= R + D) return 0;
    const x = (r - R) / D;
    return -(9 / 16) * (Math.PI / (2 * D)) * Math.cos((Math.PI / 2) * x)
      - (1 / 16) * (3 * Math.PI / (2 * D)) * Math.cos((3 * Math.PI / 2) * x);
  }

  /** g(theta) = c1 + g_o g_a with u = h - cos theta (pair_tersoff_mod.html). */
  protected override angular(t3: number, cth: number): number {
    const p = this.p3, o = 8 * t3;
    const u = p[o + 2] - cth;
    const u2 = u * u;
    const go = p[o + 4] * u2 / (p[o + 5] + u2);
    const ga = 1 + p[o + 6] * Math.exp(-p[o + 7] * u2);
    return p[o + 3] + go * ga;
  }

  /** dg/d(cos theta) = -dg/du. */
  protected override angularDeriv(t3: number, cth: number): number {
    const p = this.p3, o = 8 * t3;
    const u = p[o + 2] - cth;
    const u2 = u * u;
    const c2 = p[o + 4], c3 = p[o + 5], c4 = p[o + 6], c5 = p[o + 7];
    const den = c3 + u2;
    const go = c2 * u2 / den;
    const dgo = c2 * 2 * u * c3 / (den * den);
    const e = Math.exp(-c5 * u2);
    const ga = 1 + c4 * e;
    const dga = c4 * e * (-2 * c5 * u);
    return -(dgo * ga + go * dga);
  }

  /** exp[alpha (r_ij - r_ik)^beta]. */
  protected override zetaExp(t3: number, dr: number): number {
    const alpha = this.p3[8 * t3];
    if (alpha === 0) return 1;
    return Math.exp(alpha * Math.pow(dr, this.p3[8 * t3 + 1]));
  }

  /** d ln(zetaExp) / d r_ij. */
  protected override zetaLogDeriv(t3: number, dr: number): number {
    const alpha = this.p3[8 * t3];
    if (alpha === 0) return 0;
    const beta = this.p3[8 * t3 + 1];
    return alpha * beta * Math.pow(dr, beta - 1);
  }

  /** b_ij = (1 + zeta^eta)^(-1/(2n)). */
  protected override bondOrder(t2: number, zeta: number): number {
    const eta = this.eta2[t2];
    const zn = Math.pow(zeta, eta);
    return Math.pow(1 + zn, -1 / (2 * this.n2[t2]));
  }

  protected override bondOrderDeriv(t2: number, zeta: number, b: number): number {
    const eta = this.eta2[t2];
    const zn = Math.pow(zeta, eta);
    return -(eta / (2 * this.n2[t2])) * Math.pow(zeta, eta - 1) * b / (1 + zn);
  }
}

/** tersoff/mod/c: tersoff/mod with the additional energy term c0 inside the bracket. */
export class PairTersoffModC extends PairTersoffMod {
  override readonly name: string = 'tersoff/mod/c';

  /** c0 (scaled to the simulation energy units) per ordered two-body type index. */
  private c02 = new Float64Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.c02 = new Float64Array(nt * nt);
  }

  protected override parseFile(text: string, fileName: string): ParsedFile {
    return parseVariantFile(text, fileName, [...MOD_COLUMNS, 'c0'], (named, k) => modEntry(named, k));
  }

  protected override onPair(t: number, e: TersoffEntry): void {
    super.onPair(t, e);
    this.c02[t] = e.ext!.c0 * this.unitScale;
  }

  /** f_R -> f_R + c0: the derivative of f_R is unchanged because c0 is a constant. */
  protected override pairRadial(t2: number, r: number, out: Float64Array): void {
    super.pairRadial(t2, r, out);
    out[1] += this.c02[t2];
  }
}

/** Universal ZBL screening constants (pair_tersoff_zbl.html, phi(x)). */
const ZBL_AMP = [0.1818, 0.5099, 0.2802, 0.02817];
const ZBL_DECAY = [3.2, 0.9423, 0.4029, 0.2016];
/** Bohr radius in Angstrom, pair_tersoff_zbl.html ("typically 0.529 Angstroms"). */
const BOHR_A = 0.529;

/**
 * Coulomb prefactor of the ZBL term per unit style. Measured with native LAMMPS (black box):
 * dimers at four separations (0.9 to 2.4 Angstrom, metal; 0.9 to 2.0, real) fit one prefactor
 * to about 1e-9 relative, with the screening length 0.8854 * 0.529 / (Z_i^0.23 + Z_j^0.23)
 * and the file's Tersoff parameters unscaled (no UNITS tag). The value differs from units.ts
 * qqr2e (14.399645, 332.06371) by about 8e-8 relative, which is above the oracle tolerance.
 * Other unit styles are refused (not measured).
 */
const ZBL_QQR2E: Record<string, number> = { metal: 14.3996438057, real: 332.05588901 };

/** tersoff/zbl (docs.lammps.org/pair_tersoff_zbl.html). */
export class PairTersoffZBL extends PairTersoff {
  override readonly name: string = 'tersoff/zbl';

  /** Coulomb prefactor (1/4 pi eps0) e^2 of the unit style, set in initStyle. */
  private qqr2e = 1;
  /** Z_i Z_j qqr2e per ordered two-body type index. */
  private zq2 = new Float64Array(0);
  /** Screening length a per ordered two-body type index (Angstrom). */
  private ai2 = new Float64Array(0);
  /** r_C (ZBLcut) and A_F (ZBLexpscale) of the Fermi-like switch per two-body type index. */
  private rc2 = new Float64Array(0);
  private af2 = new Float64Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.zq2 = new Float64Array(nt * nt);
    this.ai2 = new Float64Array(nt * nt);
    this.rc2 = new Float64Array(nt * nt);
    this.af2 = new Float64Array(nt * nt);
  }

  protected override parseFile(text: string, fileName: string): ParsedFile {
    return parseVariantFile(text, fileName, TERSOFF_ZBL_COLUMNS, (named, k) => zblEntry(named, k));
  }

  protected override readFile(filename: string, ctx: StyleContext): void {
    refuseUnitsTag(ctx.readFile(filename), filename, this.name);
    super.readFile(filename, ctx);
  }

  override initStyle(ctx: StyleContext): void {
    const style = ctx.s?.units.style ?? 'metal';
    const q = ZBL_QQR2E[style];
    if (q === undefined) {
      throw new StyleError(`pair_style tersoff/zbl is only verified for metal and real units (the simulation uses ${style})`);
    }
    this.qqr2e = q;
    super.initStyle(ctx);
    warnAsymmetricTwoBody(this.entries, this.elemNames, this.fileName, ctx, 'tersoff/zbl');
  }

  protected override onPair(t: number, e: TersoffEntry): void {
    const x = e.ext!;
    this.zq2[t] = x.Zi * x.Zj * this.qqr2e;
    this.ai2[t] = 0.8854 * BOHR_A / (Math.pow(x.Zi, 0.23) + Math.pow(x.Zj, 0.23));
    this.rc2[t] = x.ZBLcut;
    this.af2[t] = x.ZBLexpscale;
  }

  /**
   * [0] f_F f_C, [1] f_R, [2] f_A, [3] d(f_F f_C)/dr, [4..5] their f_R, f_A derivatives,
   * [6] (1 - f_F) V^ZBL, [7] its derivative, with f_F = 1/(1 + exp(-A_F (r - r_C))).
   */
  protected override pairRadial(t2: number, r: number, out: Float64Array): void {
    const rs = r + this.shiftDelta;
    const R = this.R2[t2], D = this.D2[t2];
    const fc = this.cutFn(rs, R, D);
    const dfc = this.cutDeriv(rs, R, D);
    const AF = this.af2[t2];
    const fF = 1 / (1 + Math.exp(-AF * (rs - this.rc2[t2])));
    const dfF = AF * fF * (1 - fF);
    const fr = this.A2[t2] * Math.exp(-this.lam1[t2] * rs);
    const fa = -this.B2[t2] * Math.exp(-this.lam2[t2] * rs);
    out[0] = fF * fc;
    out[1] = fr;
    out[2] = fa;
    out[3] = dfF * fc + fF * dfc;
    out[4] = -this.lam1[t2] * fr;
    out[5] = -this.lam2[t2] * fa;
    // ZBL: V = zq phi(x)/r with x = r/a
    const a = this.ai2[t2];
    const x = rs / a;
    let phi = 0, dphi = 0;
    for (let q = 0; q < 4; q++) {
      const e = ZBL_AMP[q] * Math.exp(-ZBL_DECAY[q] * x);
      phi += e;
      dphi += -ZBL_DECAY[q] * e;
    }
    const zq = this.zq2[t2];
    const vz = zq * phi / rs;
    const dvz = zq * (dphi / (a * rs) - phi / (rs * rs));
    out[6] = (1 - fF) * vz;
    out[7] = -dfF * vz + (1 - fF) * dvz;
  }
}
