import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum, joinPotentialEntries } from '../util';

/*
 * pair_style tersoff — docs.lammps.org/pair_tersoff.html
 * (source: plans/lammps-docs/pair_tersoff.rst)
 *
 * Syntax (verbatim):
 *
 *    pair_style style keywords values
 *
 * * style = *tersoff* or *tersoff/table*
 * * keyword = *shift*
 *
 *   .. parsed-literal::
 *
 *        *shift* value = delta
 *          delta = negative shift in equilibrium bond length
 *
 * Energy (verbatim LaTeX from the doc page):
 *
 *   E & = \frac{1}{2} \sum_i \sum_{j \neq i} V_{ij} \\
 *   V_{ij} & = f_C(r_{ij} + \delta) \left[ f_R(r_{ij} + \delta) + b_{ij} f_A(r_{ij} + \delta) \right] \\
 *   f_C(r) & = \left\{ \begin{array} {r@{\quad:\quad}l}
 *     1 & r < R - D \\
 *     \frac{1}{2} - \frac{1}{2} \sin \left( \frac{\pi}{2} \frac{r-R}{D} \right) &
 *       R-D < r < R + D \\
 *     0 & r > R + D
 *     \end{array} \right. \\
 *   f_R(r) & =  A \exp (-\lambda_1 r) \\
 *   f_A(r) & =  -B \exp (-\lambda_2 r) \\
 *   b_{ij} & =  \left( 1 + \beta^n {\zeta_{ij}}^n \right)^{-\frac{1}{2n}} \\
 *   \zeta_{ij} & =  \sum_{k \neq i,j} f_C(r_{ik} + \delta) g \left[ \theta_{ijk}(r_{ij}, r_{ik}) \right]
 *                    \exp \left[ {\lambda_3}^m (r_{ij} - r_{ik})^m \right] \\
 *   g(\theta) & =  \gamma_{ijk} \left( 1 + \frac{c^2}{d^2} -
 *                   \frac{c^2}{\left[ d^2 + (\cos \theta - \cos \theta_0)^2\right]} \right)
 *
 * "where :math:`f_R` is a two-body term and :math:`f_A` includes three-body
 * interactions.  The summations in the formula are over all neighbors J and
 * K of atom I within a cutoff distance = R + D."
 *
 * Potential file entries (verbatim, doc parameters of a single entry):
 *
 * * element 1 (the center atom in a 3-body interaction)
 * * element 2 (the atom bonded to the center atom)
 * * element 3 (the atom influencing the 1-2 bond in a bond-order sense)
 * * m
 * * :math:`\gamma`
 * * :math:`\lambda_3` (1/distance units)
 * * c
 * * d
 * * :math:`\cos\theta_0` (can be a value < -1 or > 1)
 * * n
 * * :math:`\beta`
 * * :math:`\lambda_2` (1/distance units)
 * * B (energy units)
 * * R (distance units)
 * * D (distance units)
 * * :math:`\lambda_1` (1/distance units)
 * * A (energy units)
 *
 * "The n, :math:`\beta`, :math:`\lambda_2`, B, :math:`\lambda_1`, and A
 * parameters are only used for two-body interactions.  The m, :math:`\gamma`,
 * :math:`\lambda_3`, c, d, and :math:`\cos\theta_0` parameters are only used
 * for three-body interactions. The R and D parameters are used for both
 * two-body and three-body interactions."  "The value of m must be 3 or 1."
 *
 * Bond ordering (verbatim): "The parameters used for the two-body
 * interaction come from the entry where the second element is repeated.
 * Thus the two-body parameters for Si interacting with C, comes from the
 * SiCC entry."  "The parameters used for a particular three-body interaction
 * come from the entry with the corresponding three elements."
 *
 * pair_coeff (verbatim): "Only a single pair_coeff command is used with the
 * *tersoff* style which specifies a Tersoff potential file with parameters
 * for all needed elements."  "The first 2 arguments must be \* \* so as to
 * span all LAMMPS atom types."  "If a mapping value is specified as NULL,
 * the mapping is not performed."
 *
 * shift (verbatim): "The *shift* keyword computes the energy E of a system
 * of atoms, whose formula is the same as the Tersoff potential. The only
 * modification is that the original equilibrium bond length ( :math:`r_0`)
 * of the system is shifted to :math:`r_0-\delta`."  "each radial distance
 * :math:`r` is replaced by :math:`r+\delta`."
 *
 * Mixing / modify (verbatim): "This pair style does not support the
 * :doc:`pair_modify <pair_modify>` shift, table, and tail options."
 *
 * Unit conversion (paraphrased, not quoted: the doc says the bundled Tersoff
 * files are parameterized for metal units, and that the pair style supports
 * converting potential parameters on-the-fly between metal and real units.)
 * Only A and B carry energy units; 1 eV =
 * 23.060549 kcal/mol (NIST thermochemical calorie, measured in
 * src/engine/units.ts).
 *
 * Default (verbatim):
 *   shift delta = 0.0
 *
 * Force derivation: the energy is E = 1/2 sum_i sum_{j!=i} V_ij; with the
 * full neighbor list every ordered (owned center i, neighbor j) term is
 * evaluated once and half of V_ij and half of its forces are accumulated
 * (the mirror ordered term is supplied by the other endpoint, or by the
 * periodic image owner of a ghost).  For a fixed bond i-j the two-body
 * radial force follows from d/dr of f_C (f_R + b f_A); the bond-order
 * forces follow from dV/dx = f_C f_A b'(zeta) d(zeta)/d(x), where
 * d(zeta)/d(x) sums, over influencing atoms k, the partials of
 * H_ijk = f_C(r_ik) g(theta) exp[lambda3^m (r_ij - r_ik)^m] with respect to
 * r_ij, r_ik and cos(theta); the force on the center i is -(F_j+F_k) because
 * the cluster interaction is translation invariant.  No global virial is
 * tallied here (virialFdotr).
 */

/** One entry (element triplet) of a Tersoff potential file. */
export interface TersoffEntry {
  e1: string;
  e2: string;
  e3: string;
  m: number;
  gamma: number;
  lambda3: number;
  c: number;
  d: number;
  costheta0: number;
  n: number;
  beta: number;
  lambda2: number;
  B: number;
  R: number;
  D: number;
  lambda1: number;
  A: number;
  /** Columns beyond the base 17 of a variant file (tersoff_variants.ts), by name. */
  ext?: Record<string, number>;
}

/** 1 eV in kcal/mol (NIST thermochemical calorie; src/engine/units.ts real qe2f). */
const EV_TO_KCAL = 23.060549;

export const key3 = (e1: string, e2: string, e3: string): string => `${e1} ${e2} ${e3}`;

/** Result of parsing a potential file. */
export interface ParsedFile { entries: Map<string, TersoffEntry>; elems: Set<string>; unitTag: string | null }

/** Parses a .tersoff potential file: entry lines of 3 element names + 15 numbers. */
export const parseTersoffFile = (text: string, fileName: string): ParsedFile => {
  const lines = joinPotentialEntries(text.split(/\r?\n/), 17);
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
    if (t.length !== 17) {
      throw new StyleError(
        `Tersoff potential file ${fileName} line ${ln + 1}: expected 'element1 element2 element3 m gamma lambda3 c d costheta0 n beta lambda2 B R D lambda1 A' (17 values), got ${t.length}`,
      );
    }
    const k = key3(t[0], t[1], t[2]);
    if (entries.has(k)) throw new StyleError(`Tersoff potential file ${fileName} line ${ln + 1}: duplicate entry for elements ${k}`);
    const nums: number[] = [];
    for (let c = 3; c < 17; c++) nums.push(parseNum(t[c], `parameter ${c - 2} of the Tersoff file entry ${k}`));
    const e: TersoffEntry = {
      e1: t[0], e2: t[1], e3: t[2],
      m: nums[0], gamma: nums[1], lambda3: nums[2], c: nums[3], d: nums[4], costheta0: nums[5],
      n: nums[6], beta: nums[7], lambda2: nums[8], B: nums[9], R: nums[10], D: nums[11],
      lambda1: nums[12], A: nums[13],
    };
    // Measured with native LAMMPS (black box): n = 0 and D = 0 are accepted (SiC.tersoff of the LAMMPS distribution
    // has n = 0 in its mixed Si Si C and C Si C entries, which only carry three-body terms), negative n or D stop the
    // run with an illegal Tersoff parameter error
    if (!(e.D >= 0) || !(e.n >= 0)) {
      throw new StyleError(`Tersoff potential file ${fileName} entry ${k}: illegal Tersoff parameter (D and n must be >= 0)`);
    }
    if (e.m !== 3 && e.m !== 1) {
      throw new StyleError(`Tersoff potential file ${fileName} entry ${k}: m must be 3 or 1 (got ${e.m})`);
    }
    entries.set(k, e);
    elems.add(t[0]);
    elems.add(t[1]);
    elems.add(t[2]);
  }
  if (entries.size === 0) throw new StyleError(`Tersoff potential file ${fileName} contains no parameter entries`);
  return { entries, elems, unitTag };
};

/** Energy scale for the UNITS: metadata tag (only A and B carry energy units). */
export const unitScaleOf = (tag: string | null, ctx: StyleContext, fileName: string): number => {
  if (!tag) return 1;
  const style = ctx.s?.units.style;
  if (tag === style) return 1;
  if (tag === 'metal' && style === 'real') return EV_TO_KCAL;
  if (tag === 'real' && style === 'metal') return 1 / EV_TO_KCAL;
  throw new StyleError(
    `Tersoff potential file ${fileName} requires ${tag} units but the simulation uses ${style ?? 'an unknown'} unit style; automatic conversion is only supported between metal and real`,
  );
};

export class PairTersoff extends Pair {
  readonly name: string = 'tersoff';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  /** pair_style keyword *shift* delta (default 0.0). */
  protected shiftDelta = 0;

  protected fileName = '';
  protected fileRead = false;
  protected entries = new Map<string, TersoffEntry>();
  protected fileElems = new Set<string>();
  protected unitScale = 1;
  protected elemOf = new Int32Array(0);
  protected elemNames: string[] = [];

  // two-body parameters of the (E_i, E_j, E_j) entry, indexed ti*nt+tj (ordered)
  protected A2 = new Float64Array(0);
  protected B2 = new Float64Array(0);
  protected lam1 = new Float64Array(0);
  protected lam2 = new Float64Array(0);
  protected beta2 = new Float64Array(0);
  protected n2 = new Float64Array(0);
  protected R2 = new Float64Array(0);
  protected D2 = new Float64Array(0);
  protected cut2 = new Float64Array(0);
  // three-body parameters of the (E_i, E_j, E_k) entry, indexed (ti*nt+tj)*nt+tk
  protected m3 = new Float64Array(0);
  protected gam3 = new Float64Array(0);
  protected lam3p = new Float64Array(0);
  protected c3 = new Float64Array(0);
  protected d3 = new Float64Array(0);
  protected ct03 = new Float64Array(0);
  protected R3 = new Float64Array(0);
  protected D3 = new Float64Array(0);
  protected cut3 = new Float64Array(0);
  protected valid3 = new Uint8Array(0);
  /** The neighbor-list cutoff for every non-NULL type pair (max R+D of relevant entries). */
  protected pairCut = 0;

  // per-center gathered-neighbor scratch
  protected cap = 0;
  protected gJ = new Int32Array(0);
  protected gDx = new Float64Array(0);
  protected gDy = new Float64Array(0);
  protected gDz = new Float64Array(0);
  protected gR = new Float64Array(0);
  // per-k scratch for the second pass of one bond
  protected kIdx = new Int32Array(0);
  protected kDx = new Float64Array(0);
  protected kDy = new Float64Array(0);
  protected kDz = new Float64Array(0);
  protected kR = new Float64Array(0);
  protected dHdRik = new Float64Array(0);
  protected dHdRij = new Float64Array(0);
  protected dHdc = new Float64Array(0);
  protected kGx = new Float64Array(0);
  protected kGy = new Float64Array(0);
  protected kGz = new Float64Array(0);

  override settings(args: string[], _ctx: StyleContext): void {
    for (let k = 0; k < args.length; k++) {
      const key = args[k];
      if (key === 'shift') {
        this.shiftDelta = parseNum(args[++k], 'shift delta');
      } else {
        throw new StyleError(`pair_style ${this.name} keyword '${key}' is not supported (keywords: shift)`);
      }
    }
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.elemOf = new Int32Array(nt).fill(-1);
    this.elemNames = [];
    const n2 = nt * nt;
    this.A2 = new Float64Array(n2);
    this.B2 = new Float64Array(n2);
    this.lam1 = new Float64Array(n2);
    this.lam2 = new Float64Array(n2);
    this.beta2 = new Float64Array(n2);
    this.n2 = new Float64Array(n2);
    this.R2 = new Float64Array(n2);
    this.D2 = new Float64Array(n2);
    this.cut2 = new Float64Array(n2);
    const n3 = nt * nt * nt;
    this.m3 = new Float64Array(n3);
    this.gam3 = new Float64Array(n3);
    this.lam3p = new Float64Array(n3);
    this.c3 = new Float64Array(n3);
    this.d3 = new Float64Array(n3);
    this.ct03 = new Float64Array(n3);
    this.R3 = new Float64Array(n3);
    this.D3 = new Float64Array(n3);
    this.cut3 = new Float64Array(n3);
    this.valid3 = new Uint8Array(n3);
    this.fileRead = false;
    this.entries.clear();
    this.fileElems.clear();
    this.unitScale = 1;
    this.fileName = '';
    this.pairCut = 0;
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError(`usage: pair_coeff * * filename elem1 ... elemN (style ${this.name})`);
    const filename = args[2];
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError(`pair_coeff for style ${this.name} needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    }
    if (args[0] !== '*' || args[1] !== '*') {
      throw new StyleError(`the first 2 arguments of pair_coeff for style ${this.name} must be * *`);
    }
    if (!this.fileRead) {
      this.readFile(filename, ctx);
    } else if (filename !== this.fileName) {
      throw new StyleError(`only one pair_coeff command is allowed for style ${this.name} (potential file already set to ${this.fileName})`);
    }
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      let idx: number;
      if (name === 'NULL') {
        idx = -1;
      } else {
        if (!this.fileElems.has(name)) {
          throw new StyleError(`element '${name}' is not in Tersoff potential file ${this.fileName} (elements: ${[...this.fileElems].sort().join(' ')})`);
        }
        idx = this.elemNames.indexOf(name);
        if (idx < 0) {
          idx = this.elemNames.length;
          this.elemNames.push(name);
        }
      }
      this.elemOf[t] = idx;
    }
  }

  /** Parses the potential file text; the variant styles (tersoff_variants.ts) override it for their file layouts. */
  protected parseFile(text: string, fileName: string): ParsedFile {
    return parseTersoffFile(text, fileName);
  }

  protected readFile(filename: string, ctx: StyleContext): void {
    const parsed = this.parseFile(ctx.readFile(filename), filename);
    const scale = unitScaleOf(parsed.unitTag, ctx, filename);
    if (scale !== 1) ctx.log(`WARNING: converting Tersoff potential file ${filename} from ${parsed.unitTag} to ${ctx.s?.units.style} units`);
    this.entries = parsed.entries;
    this.fileElems = parsed.elems;
    this.unitScale = scale;
    this.fileName = filename;
    this.fileRead = true;
  }

  protected entry(k: string, what: string): TersoffEntry {
    const e = this.entries.get(k);
    if (!e) throw new StyleError(`Tersoff potential file ${this.fileName} has no entry for elements ${k} (${what})`);
    return e;
  }

  override initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify shift and tail options`);
    }
    if (this.table !== 12) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify table option`);
    }
    if (!this.fileRead) {
      throw new StyleError(`pair_style ${this.name} needs a pair_coeff command with a Tersoff potential file`);
    }
    const nt = this.ntypes + 1;
    const scale = this.unitScale;
    // two-body tables from the (E_i, E_j, E_j) entries
    for (let ti = 1; ti <= this.ntypes; ti++) {
      const ei = this.elemOf[ti];
      if (ei < 0) continue;
      for (let tj = 1; tj <= this.ntypes; tj++) {
        const ej = this.elemOf[tj];
        if (ej < 0) continue;
        const k = key3(this.elemNames[ei], this.elemNames[ej], this.elemNames[ej]);
        const e = this.entry(k, `needed by the two-body interaction of types ${ti} ${tj}`);
        const t = ti * nt + tj;
        this.A2[t] = e.A * scale;
        this.B2[t] = e.B * scale;
        this.lam1[t] = e.lambda1;
        this.lam2[t] = e.lambda2;
        this.beta2[t] = e.beta;
        this.n2[t] = e.n;
        this.R2[t] = e.R;
        this.D2[t] = e.D;
        this.cut2[t] = e.R + e.D;
        this.onPair(t, e);
      }
    }
    // three-body tables from the (E_i, E_j, E_k) entries
    for (let ti = 1; ti <= this.ntypes; ti++) {
      const ei = this.elemOf[ti];
      if (ei < 0) continue;
      for (let tj = 1; tj <= this.ntypes; tj++) {
        const ej = this.elemOf[tj];
        if (ej < 0) continue;
        for (let tk = 1; tk <= this.ntypes; tk++) {
          const ek = this.elemOf[tk];
          if (ek < 0) continue;
          const k = key3(this.elemNames[ei], this.elemNames[ej], this.elemNames[ek]);
          const e = this.entry(k, `needed by the three-body combination of types ${ti} ${tj} ${tk}`);
          const t = (ti * nt + tj) * nt + tk;
          this.m3[t] = e.m;
          this.gam3[t] = e.gamma;
          this.lam3p[t] = Math.pow(e.lambda3, e.m);
          this.c3[t] = e.c;
          this.d3[t] = e.d;
          this.ct03[t] = e.costheta0;
          this.R3[t] = e.R;
          this.D3[t] = e.D;
          this.cut3[t] = e.R + e.D;
          this.valid3[t] = 1;
          this.onTriplet(t, e);
          if (e.R + e.D > this.pairCut) this.pairCut = e.R + e.D;
        }
      }
    }
    for (let ti = 1; ti <= this.ntypes; ti++) {
      if (this.elemOf[ti] < 0) continue;
      for (let tj = 1; tj <= this.ntypes; tj++) {
        const t = ti * nt + tj;
        if (this.elemOf[tj] < 0) continue;
        if (this.cut2[t] > this.pairCut) this.pairCut = this.cut2[t];
      }
    }
  }

  /** Hook: the two-body table of type index t (ordered, ti*nt+tj) was filled from entry e. */
  protected onPair(_t: number, _e: TersoffEntry): void {}

  /** Hook: the three-body table of type index t ((ti*nt+tj)*nt+tk) was filled from entry e. */
  protected onTriplet(_t: number, _e: TersoffEntry): void {}

  override initOne(i: number, j: number): number {
    const ei = this.elemOf[i];
    const ej = this.elemOf[j];
    if (ei < 0 || ej < 0) return 0;
    return this.pairCut;
  }

  /** f_C(r) with cutoff R,D (docs formula above). */
  protected cutFn(r: number, R: number, D: number): number {
    if (r < R - D) return 1;
    if (r > R + D) return 0;
    return 0.5 - 0.5 * Math.sin((Math.PI / 2) * (r - R) / D);
  }

  /** d f_C/dr. */
  protected cutDeriv(r: number, R: number, D: number): number {
    if (r <= R - D || r >= R + D) return 0;
    return -(Math.PI / (4 * D)) * Math.cos((Math.PI / 2) * (r - R) / D);
  }

  /** Angular factor g(cos theta) of the (ijk) entry t3 (docs: g(theta) formula above). */
  protected angular(t3: number, cth: number): number {
    const q = cth - this.ct03[t3];
    const dd = this.d3[t3] * this.d3[t3];
    const den = dd + q * q;
    const c2 = this.c3[t3] * this.c3[t3];
    return this.gam3[t3] * (1 + c2 / dd - c2 / den);
  }

  /** d g / d cos(theta) of the (ijk) entry t3. */
  protected angularDeriv(t3: number, cth: number): number {
    const q = cth - this.ct03[t3];
    const dd = this.d3[t3] * this.d3[t3];
    const den = dd + q * q;
    const c2 = this.c3[t3] * this.c3[t3];
    return 2 * this.gam3[t3] * c2 * q / (den * den);
  }

  /** exp[lambda3^m (r_ij - r_ik)^m] of the (ijk) entry t3, dr = r_ij - r_ik. */
  protected zetaExp(t3: number, dr: number): number {
    const lp = this.lam3p[t3];
    if (lp === 0) return 1;
    return Math.exp(lp * Math.pow(dr, this.m3[t3]));
  }

  /** d ln(zetaExp) / d r_ij of the (ijk) entry t3. */
  protected zetaLogDeriv(t3: number, dr: number): number {
    const lp = this.lam3p[t3];
    if (lp === 0) return 0;
    const mm = this.m3[t3];
    return lp * mm * Math.pow(dr, mm - 1);
  }

  /** b_ij of the (ij) entry t2 for a bond-order sum zeta > 0 (docs: b_ij formula above). */
  protected bondOrder(t2: number, zeta: number): number {
    const bn = Math.pow(this.beta2[t2], this.n2[t2]);
    const zn = Math.pow(zeta, this.n2[t2]);
    const denom = 1 + bn * zn;
    return Math.pow(denom, -1 / (2 * this.n2[t2]));
  }

  /** d b_ij / d zeta_ij (b is the value of bondOrder at this zeta). */
  protected bondOrderDeriv(t2: number, zeta: number, b: number): number {
    const bn = Math.pow(this.beta2[t2], this.n2[t2]);
    const zn = Math.pow(zeta, this.n2[t2]);
    const denom = 1 + bn * zn;
    return -0.5 * b * bn * Math.pow(zeta, this.n2[t2] - 1) / denom;
  }

  /**
   * Radial pieces of the (ij) pair of entry t2 at distance r, written to out:
   * [0] f_C (cutoff factor multiplying the bond-order terms), [1] f_R (plus any
   * zeta-independent constant), [2] f_A, [3] d f_C/dr, [4] d f_R/dr, [5] d f_A/dr,
   * [6] additive zeta-independent pair energy, [7] its derivative d/dr.
   * V_ij = out[0] (out[1] + b out[2]) + out[6].
   */
  protected pairRadial(t2: number, r: number, out: Float64Array): void {
    const rs = r + this.shiftDelta;
    const R = this.R2[t2], D = this.D2[t2];
    const facc = this.cutFn(rs, R, D);
    const fr = this.A2[t2] * Math.exp(-this.lam1[t2] * rs);
    const fa = -this.B2[t2] * Math.exp(-this.lam2[t2] * rs);
    out[0] = facc;
    out[1] = fr;
    out[2] = fa;
    out[3] = this.cutDeriv(rs, R, D);
    out[4] = -this.lam1[t2] * fr;
    out[5] = -this.lam2[t2] * fa;
    out[6] = 0;
    out[7] = 0;
  }

  /** Scratch for pairRadial. */
  protected pr = new Float64Array(8);

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const elemOf = this.elemOf;
    const sft = this.shiftDelta;
    const A2 = this.A2, B2 = this.B2, lam1 = this.lam1, lam2 = this.lam2;
    const beta2 = this.beta2, n2 = this.n2, R2 = this.R2, D2 = this.D2, cut2 = this.cut2;
    const m3 = this.m3, gam3 = this.gam3, lam3p = this.lam3p, c3 = this.c3, d3 = this.d3;
    const ct03 = this.ct03, R3 = this.R3, D3 = this.D3, cut3 = this.cut3, valid3 = this.valid3;
    const eatom = pc.eatom;
    const vatom = pc.vatom;
    let evdwl = 0;

    let maxn = 0;
    for (let i = 0; i < list.inum; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    if (maxn > this.cap) {
      this.cap = maxn;
      this.gJ = new Int32Array(maxn);
      this.gDx = new Float64Array(maxn); this.gDy = new Float64Array(maxn); this.gDz = new Float64Array(maxn);
      this.gR = new Float64Array(maxn);
      this.kIdx = new Int32Array(maxn);
      this.kDx = new Float64Array(maxn); this.kDy = new Float64Array(maxn); this.kDz = new Float64Array(maxn);
      this.kR = new Float64Array(maxn);
      this.dHdRik = new Float64Array(maxn); this.dHdRij = new Float64Array(maxn); this.dHdc = new Float64Array(maxn);
      this.kGx = new Float64Array(maxn); this.kGy = new Float64Array(maxn); this.kGz = new Float64Array(maxn);
    }
    const gJ = this.gJ, gDx = this.gDx, gDy = this.gDy, gDz = this.gDz, gR = this.gR;
    const kIdx = this.kIdx, kDx = this.kDx, kDy = this.kDy, kDz = this.kDz, kR = this.kR;
    const dHdRik = this.dHdRik, dHdRij = this.dHdRij, dHdc = this.dHdc;
    const kGx = this.kGx, kGy = this.kGy, kGz = this.kGz;

    for (let i = 0; i < nlocal; i++) {
      const tiv = type[i];
      if (elemOf[tiv] < 0) continue;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      // gather neighbors of i once (vector from i to neighbor)
      let m = 0;
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (r === 0) continue;
        gJ[m] = j;
        gDx[m] = dx; gDy[m] = dy; gDz[m] = dz;
        gR[m] = r;
        m++;
      }
      const base2 = tiv * nt;
      for (let a = 0; a < m; a++) {
        const j = gJ[a];
        const tjv = type[j];
        if (elemOf[tjv] < 0) continue;
        const r1 = gR[a];
        const t2 = base2 + tjv;
        const c2r = cut2[t2];
        if (r1 >= c2r) continue;
        const e1x = gDx[a] / r1, e1y = gDy[a] / r1, e1z = gDz[a] / r1;
        const pr = this.pr;
        this.pairRadial(t2, r1, pr);
        const facc = pr[0];
        const fr = pr[1];
        const fa = pr[2];

        // bond-order environment: zeta_ij = sum_k f_C(r_ik) g exp[...]
        const base3 = (base2 + tjv) * nt; // (tiv*nt + tjv)*nt
        let zeta = 0;
        let nk = 0;
        for (let b = 0; b < m; b++) {
          if (b === a) continue;
          const kk = gJ[b];
          const tkv = type[kk];
          if (elemOf[tkv] < 0) continue;
          const r2 = gR[b];
          const t3 = base3 + tkv;
          if (valid3[t3] === 0) continue;
          if (r2 >= cut3[t3]) continue;
          const e2x = gDx[b] / r2, e2y = gDy[b] / r2, e2z = gDz[b] / r2;
          const fc2 = this.cutFn(r2 + sft, R3[t3], D3[t3]);
          const cth = (e1x * e2x + e1y * e2y + e1z * e2z);
          const g = this.angular(t3, cth);
          // exp[lambda3^m (r_ij - r_ik)^m]
          const dr = r1 - r2;
          const efil = this.zetaExp(t3, dr);
          const dphi = this.zetaLogDeriv(t3, dr);
          const H = fc2 * g * efil;
          zeta += H;
          // partials of H
          const fc2p = this.cutDeriv(r2 + sft, R3[t3], D3[t3]);
          const drik = g * efil * fc2p - H * dphi;
          const drij = H * dphi;
          const dgdc = this.angularDeriv(t3, cth);
          kIdx[nk] = kk; kR[nk] = r2;
          kGx[nk] = e2x; kGy[nk] = e2y; kGz[nk] = e2z;
          dHdRik[nk] = drik; dHdRij[nk] = drij; dHdc[nk] = fc2 * efil * dgdc;
          nk++;
        }

        // b_ij and d b/d zeta
        let b = 1, dbdz = 0;
        if (zeta > 0) {
          b = this.bondOrder(t2, zeta);
          dbdz = this.bondOrderDeriv(t2, zeta, b);
        }

        const v = facc * (fr + b * fa) + pr[6];
        evdwl += 0.5 * v;

        const dRad = pr[3] * (fr + b * fa) + facc * (pr[4] + b * pr[5]) + pr[7];

        // radial half-force: on j = -0.5*dRad*e1, on i = +0.5*dRad*e1
        const hx = 0.5 * dRad * e1x, hy = 0.5 * dRad * e1y, hz = 0.5 * dRad * e1z;
        f[3 * i] += hx; f[3 * i + 1] += hy; f[3 * i + 2] += hz;
        f[3 * j] -= hx; f[3 * j + 1] -= hy; f[3 * j + 2] -= hz;

        if (eatom) {
          // split the ordered pair energy half to i and j (mirror visit adds the rest)
          eatom[i] += 0.25 * v;
          eatom[j] += 0.25 * v;
        }

        if (dbdz !== 0 && nk > 0) {
          const C = 0.5 * facc * fa * dbdz; // includes the 1/2 energy factor
          for (let b2 = 0; b2 < nk; b2++) {
            const kk = kIdx[b2];
            const r2 = kR[b2];
            const e2x = kGx[b2], e2y = kGy[b2], e2z = kGz[b2];
            const drij = dHdRij[b2], drik = dHdRik[b2], ddc = dHdc[b2];
            // F_j = -C [ drij e1 + ddc (e2/r1 - cth e1/r1) ]
            // F_k = -C [ drik e2 + ddc (e1/r2 - cth e2/r2) ]
            const r1inv = 1 / r1, r2inv = 1 / r2;
            const cth = e1x * e2x + e1y * e2y + e1z * e2z;
            const fjx = -C * (drij * e1x + ddc * (e2x * r1inv - cth * e1x * r1inv));
            const fjy = -C * (drij * e1y + ddc * (e2y * r1inv - cth * e1y * r1inv));
            const fjz = -C * (drij * e1z + ddc * (e2z * r1inv - cth * e1z * r1inv));
            const fkx = -C * (drik * e2x + ddc * (e1x * r2inv - cth * e2x * r2inv));
            const fky = -C * (drik * e2y + ddc * (e1y * r2inv - cth * e2y * r2inv));
            const fkz = -C * (drik * e2z + ddc * (e1z * r2inv - cth * e2z * r2inv));
            f[3 * i] -= fjx + fkx; f[3 * i + 1] -= fjy + fky; f[3 * i + 2] -= fjz + fkz;
            f[3 * j] += fjx; f[3 * j + 1] += fjy; f[3 * j + 2] += fjz;
            f[3 * kk] += fkx; f[3 * kk + 1] += fky; f[3 * kk + 2] += fkz;
            if (vatom) {
              const w0 = (e1x * fjx + e2x * fkx) / 3;
              const w1 = (e1y * fjy + e2y * fky) / 3;
              const w2 = (e1z * fjz + e2z * fkz) / 3;
              const w3 = (e1x * fjy + e2x * fky) / 3;
              const w4 = (e1x * fjz + e2x * fkz) / 3;
              const w5 = (e1y * fjz + e2y * fkz) / 3;
              for (const q of [i, j, kk]) {
                vatom[6 * q] += w0; vatom[6 * q + 1] += w1; vatom[6 * q + 2] += w2;
                vatom[6 * q + 3] += w3; vatom[6 * q + 4] += w4; vatom[6 * q + 5] += w5;
              }
            }
          }
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
