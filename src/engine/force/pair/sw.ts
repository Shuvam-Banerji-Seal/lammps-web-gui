import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum } from '../util';

/*
 * pair_style sw, sw/mod — docs.lammps.org/pair_sw.html
 * (source: plans/lammps-docs/pair_sw.rst)
 *
 * Syntax (verbatim):
 *
 *    pair_style style keyword values
 *
 * * style = *sw* or *sw/mod*
 * * keyword = *maxdelcs* or *threebody*
 *
 *   *maxdelcs* value = delta1 delta2 (optional, sw/mod only)
 *     delta1 = The minimum thershold for the variation of cosine of three-body angle
 *     delta2 = The maximum threshold for the variation of cosine of three-body angle
 *   *threebody* value = *on* or *off* (optional, sw only)
 *     on (default) = Compute both the three-body and two-body terms of the potential
 *     off = Compute only the two-body term of the potential
 *
 * Energy (verbatim LaTeX from the doc page):
 *
 *    E & =  \sum_i \sum_{j > i} \phi_2 (r_{ij}) +
 *          \sum_i \sum_{j \neq i} \sum_{k > j}
 *          \phi_3 (r_{ij}, r_{ik}, \theta_{ijk}) \\
 *   \phi_2(r_{ij}) & =  A_{ij} \epsilon_{ij} \left[ B_{ij} (\frac{\sigma_{ij}}{r_{ij}})^{p_{ij}} -
 *                     (\frac{\sigma_{ij}}{r_{ij}})^{q_{ij}} \right]
 *                     \exp \left( \frac{\sigma_{ij}}{r_{ij} - a_{ij} \sigma_{ij}} \right) \\
 *   \phi_3(r_{ij},r_{ik},\theta_{ijk}) & = \lambda_{ijk} \epsilon_{ijk} \left[ \cos \theta_{ijk} -
 *                     \cos \theta_{0ijk} \right]^2
 *                     \exp \left( \frac{\gamma_{ij} \sigma_{ij}}{r_{ij} - a_{ij} \sigma_{ij}} \right)
 *                     \exp \left( \frac{\gamma_{ik} \sigma_{ik}}{r_{ik} - a_{ik} \sigma_{ik}} \right)
 *
 * "where :math:`\phi_2` is a two-body term and :math:`\phi_3` is a
 * three-body term.  The summations in the formula are over all neighbors J
 * and K of atom I within a cutoff distance :math:`a `\sigma`." (sic)
 *
 * Potential file and parameter mapping (verbatim):
 *   "Lines that are not blank or comments (starting with #) define parameters
 *   for a triplet of elements."  The parameters of one entry are
 *   "element 1 (the center atom in a 3-body interaction)", element 2,
 *   element 3, "epsilon (energy units)", "sigma (distance units)", a, lambda,
 *   gamma, "costheta0", A, B, p, q, tol.  "The A, B, p, and q parameters are
 *   used only for two-body interactions. The lambda and costheta0 parameters
 *   are used only for three-body interactions. The epsilon, sigma and a
 *   parameters are used for both two-body and three-body interactions. gamma
 *   is used only in the three-body interactions, but is defined for pairs of
 *   atoms."  "The parameter values used for the two-body interaction come from the
 *   entry where the second and third elements are the same.  Thus the
 *   two-body parameters for Si interacting with C, comes from the SiCC
 *   entry."  The three-body lambda/epsilon/costheta0 come from the
 *   (center, j, k) entry, and "the function :math:`\phi_3` contains two
 *   exponential screening factors with parameter values from the ij pair and
 *   ik pairs"
 *   (for "a C atom bonded to a Si atom and a second C atom", phi_3 "will
 *   depend on the three-body parameters for the CSiC entry, and also on the
 *   two-body parameters for the CCC and CSiSi entries").
 *
 * pair_coeff (verbatim): "Only a single pair_coeff command is used with the
 * sw and sw/mod styles which specifies a Stillinger-Weber potential file with
 * parameters for all needed elements, except for when the threebody off
 * setting is used (see note below)."  "The first 2 arguments must be \* \*
 * so as to span all LAMMPS atom types."  "If an argument value is specified as
 * NULL, the mapping is not performed."  With threebody off: "multiple
 * pair_coeff commands may be used to specific the pairs of atoms which don't
 * require three-body term.  In these cases, the first 2 arguments are not
 * required to be \* \*, the potential parameter file is only read by the first
 * pair_coeff command and the element to atom type mappings must be consistent
 * across all pair_coeff statements.  If not LAMMPS will abort with an error."
 *
 * tol (verbatim): "LAMMPS provides a tol value for each of the three-body
 * entries so that they can be separately controlled. ... If tol = 0.0, then
 * the standard Stillinger-Weber cutoff is used."  The virtual-cutoff formula
 * for tol > 0 is not given in the documentation, so any needed entry with
 * tol != 0 raises a StyleError here instead of being silently approximated.
 *
 * threebody off (verbatim): "To turn off the threebody contributions all
 * :math:`\lambda_{ijk}` parameters from the potential file are forcibly set
 * to 0."
 *
 * sw/mod (verbatim): "the value of :math:`\delta = \cos \theta_{ijk} - \cos
 * \theta_{0ijk}` used in the original energy and force expression is scaled
 * by a switching factor :math:`f_C(\delta)`" with
 *
 *   f_C(\delta) & = \left\{ \begin{array} {r@{\quad:\quad}l}
 *     1 & \left| \delta \right| < \delta_1 \\
 *     \frac{1}{2} + \frac{1}{2} \cos \left( \pi \frac{\left| \delta \right| - \delta_1}{\delta_2 - \delta_1} \right) &
 *       \delta_1 < \left| \delta \right| < \delta_2 \\
 *     0 & \left| \delta \right| > \delta_2
 *     \end{array} \right. \\
 *
 * and the force caveat "the angle dependence for the cut-off function is not
 * implemented in the force (first derivation of potential)": the force treats
 * f_C as a constant with respect to the angle, i.e. d(zeta^2)/d(angle) is
 * taken as 2*zeta*f_C with zeta = f_C*delta.
 *
 * "This pair style does not support the pair_modify shift, table, and tail
 * options."  "The single() function of the sw pair style is only enabled and
 * supported for the case of the threebody off setting."  Units metadata:
 * "If the potential file contains a 'UNITS:' metadata tag in the first line
 * of the potential file, then LAMMPS can convert it transparently between
 * "metal" and "real" units"; pair_coeff.html: "In those cases, a warning
 * message signaling that an automatic conversion has happened is printed to
 * the screen."  Only epsilon carries energy units; 1 eV = 23.060549 kcal/mol
 * (NIST thermochemical calorie, the factor measured in src/engine/units.ts).
 *
 * Default (verbatim):
 *   The default value for the *threebody* setting of the "sw" pair style is
 *   "on", the default values for the "*maxdelcs* setting of the *sw/mod*
 *   pair style are *delta1* = 0.25 and *delta2* = 0.35`.
 *
 * Cutoffs: "This pair style requires the newton setting to be "on" for pair
 * interactions" (the engine always evaluates each physical pair once).  The
 * interaction cutoff of a type pair is a*sigma of its two-body entry; the
 * three-body screening of each leg is zero beyond the a*sigma of that leg's
 * own (center, neighbor, neighbor) entry, so the neighbor-list cutoff is the
 * largest of the two-body and both leg cutoffs of the type pair.
 *
 * Full-list scheme: the two-body term is summed over the full neighbor list
 * with half the pair energy and half the pair force per visit (every physical
 * pair is visited from both ends); the three-body term is summed per owned
 * center atom over unordered neighbor pairs and counted exactly once, with
 * its forces on the two neighbors added at their (possibly ghost) indices so
 * the force field folds them back to the owners.  Per-atom energy and virial
 * follow compute_stress_atom.html: "The total contribution for the cluster
 * interaction is divided evenly among those atoms."
 */

/** One entry (element triplet) of a Stillinger-Weber potential file. */
interface SWEntry {
  e1: string;
  e2: string;
  e3: string;
  eps: number;
  sigma: number;
  a: number;
  lambda: number;
  gamma: number;
  costheta0: number;
  A: number;
  B: number;
  p: number;
  q: number;
  tol: number;
}

/** 1 eV in kcal/mol (NIST thermochemical calorie; src/engine/units.ts real qe2f). */
const EV_TO_KCAL = 23.060549;

const key3 = (e1: string, e2: string, e3: string): string => `${e1} ${e2} ${e3}`;

/** Parses a .sw potential file: entry lines of 3 element names + 11 numbers. */
const parseSWFile = (text: string, fileName: string): { entries: Map<string, SWEntry>; elems: Set<string>; unitTag: string | null } => {
  const lines = text.split(/\r?\n/);
  let unitTag: string | null = null;
  let start = 0;
  if (lines.length > 0) {
    const m = /UNITS:\s*(\S+)/.exec(lines[0]);
    if (m) {
      unitTag = m[1];
      start = 1;
    }
  }
  const entries = new Map<string, SWEntry>();
  const elems = new Set<string>();
  for (let ln = start; ln < lines.length; ln++) {
    const raw = lines[ln];
    const hash = raw.indexOf('#');
    const line = (hash >= 0 ? raw.slice(0, hash) : raw).trim();
    if (line === '') continue;
    const t = line.split(/\s+/);
    if (t.length !== 14) {
      throw new StyleError(
        `SW potential file ${fileName} line ${ln + 1}: expected 'element1 element2 element3 epsilon sigma a lambda gamma costheta0 A B p q tol' (14 values), got ${t.length}`,
      );
    }
    const k = key3(t[0], t[1], t[2]);
    if (entries.has(k)) throw new StyleError(`SW potential file ${fileName} line ${ln + 1}: duplicate entry for elements ${k}`);
    const nums: number[] = [];
    for (let c = 3; c < 14; c++) nums.push(parseNum(t[c], `parameter ${c - 2} of the SW file entry ${k}`));
    const e: SWEntry = {
      e1: t[0], e2: t[1], e3: t[2],
      eps: nums[0], sigma: nums[1], a: nums[2], lambda: nums[3], gamma: nums[4],
      costheta0: nums[5], A: nums[6], B: nums[7], p: nums[8], q: nums[9], tol: nums[10],
    };
    if (!(e.sigma > 0) || !(e.a > 0)) {
      throw new StyleError(`SW potential file ${fileName} entry ${k}: sigma and a must be > 0`);
    }
    entries.set(k, e);
    elems.add(t[0]);
    elems.add(t[1]);
    elems.add(t[2]);
  }
  if (entries.size === 0) throw new StyleError(`SW potential file ${fileName} contains no parameter entries`);
  return { entries, elems, unitTag };
};

/** Energy scale for the UNITS: metadata tag (only epsilon carries energy units). */
const unitScaleOf = (tag: string | null, ctx: StyleContext, fileName: string): number => {
  if (!tag) return 1;
  const style = ctx.s?.units.style;
  if (tag === style) return 1;
  if (tag === 'metal' && style === 'real') return EV_TO_KCAL;
  if (tag === 'real' && style === 'metal') return 1 / EV_TO_KCAL;
  throw new StyleError(
    `SW potential file ${fileName} requires ${tag} units but the simulation uses ${style ?? 'an unknown'} unit style; automatic conversion is only supported between metal and real`,
  );
};

/**
 * Shared implementation of pair_style sw and sw/mod. The two-body term runs
 * over the full neighbor list (half energy/force per visit, both ends
 * visited); the three-body term runs per owned center atom over unordered
 * neighbor pairs of the gathered list, once per triplet.
 */
abstract class PairSWBase extends Pair {
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  /** sw: the threebody keyword; sw/mod always computes the three-body term. */
  protected threebody = true;
  /** sw/mod switches the angle term by f_C(delta). */
  protected readonly isMod: boolean;

  protected constructor(isMod: boolean) {
    super();
    this.isMod = isMod;
  }

  protected fileName = '';
  protected fileRead = false;
  protected entries = new Map<string, SWEntry>();
  protected fileElems = new Set<string>();
  /** Multiply file epsilons by this (UNITS: tag conversion). */
  protected unitScale = 1;
  protected elemOf = new Int32Array(0);
  protected elemNames: string[] = [];
  protected mappingSet = false;
  /** threebody off: type pairs with a two-body interaction (symmetric). */
  protected active = new Uint8Array(0);

  // two-body pair parameters of the (i,j,j) entry, symmetric tables (nt*nt)
  protected eps2 = new Float64Array(0);
  protected sigma2 = new Float64Array(0);
  protected a2 = new Float64Array(0);
  protected pA = new Float64Array(0);
  protected pB = new Float64Array(0);
  protected pP = new Float64Array(0);
  protected pQ = new Float64Array(0);
  /** a*sigma of the two-body entry per type pair (the two-body term is zero beyond it). */
  protected cut2 = new Float64Array(0);
  // screening leg (center I -> neighbor J): gamma, sigma, a of entry (EI,EJ,EJ), ordered
  protected gam = new Float64Array(0);
  protected sigS = new Float64Array(0);
  protected aS = new Float64Array(0);
  // three-body parameters of the (I,J,K) entry, tables of size nt^3
  protected lam3 = new Float64Array(0);
  protected eps3 = new Float64Array(0);
  protected c03 = new Float64Array(0);

  // per-center scratch for the gathered neighbors
  private cap = 0;
  private sIdx = new Int32Array(0);
  private sDx = new Float64Array(0);
  private sDy = new Float64Array(0);
  private sDz = new Float64Array(0);
  private sR = new Float64Array(0);
  private sInvR = new Float64Array(0);
  private sES = new Float64Array(0);
  private sW = new Float64Array(0);

  override settings(args: string[], _ctx: StyleContext): void {
    for (let k = 0; k < args.length; k++) {
      const key = args[k];
      if (key === 'threebody' && !this.isMod) {
        const v = args[++k];
        if (v === 'on') this.threebody = true;
        else if (v === 'off') this.threebody = false;
        else throw new StyleError(`usage: pair_style ${this.name} threebody on|off (got '${v}')`);
      } else if (key === 'maxdelcs' && this.isMod) {
        const d1 = parseNum(args[++k], 'delta1');
        const d2 = parseNum(args[++k], 'delta2');
        if (!(d1 >= 0) || !(d2 > d1)) throw new StyleError(`usage: pair_style ${this.name} maxdelcs delta1 delta2 with 0 <= delta1 < delta2 (got ${d1} ${d2})`);
        this.setDeltas(d1, d2);
      } else if (key === 'threebody') {
        throw new StyleError(`pair_style ${this.name} does not support the threebody keyword (only the plain sw style does)`);
      } else if (key === 'maxdelcs') {
        throw new StyleError(`pair_style ${this.name} does not support the maxdelcs keyword (only sw/mod does)`);
      } else {
        throw new StyleError(`pair_style ${this.name} keyword '${key}' is not supported (keywords: ${this.isMod ? 'maxdelcs' : 'threebody'})`);
      }
    }
  }

  protected setDeltas(_d1: number, _d2: number): void {}

  /** f_C(delta) of sw/mod; the plain sw style returns 1 (no angle scaling). */
  protected angleScale(_dc: number): number {
    return 1;
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.elemOf = new Int32Array(nt).fill(-1);
    this.elemNames = [];
    this.mappingSet = false;
    this.active = new Uint8Array(nt * nt);
    this.eps2 = new Float64Array(nt * nt);
    this.sigma2 = new Float64Array(nt * nt);
    this.a2 = new Float64Array(nt * nt);
    this.pA = new Float64Array(nt * nt);
    this.pB = new Float64Array(nt * nt);
    this.pP = new Float64Array(nt * nt);
    this.pQ = new Float64Array(nt * nt);
    this.cut2 = new Float64Array(nt * nt);
    this.gam = new Float64Array(nt * nt);
    this.sigS = new Float64Array(nt * nt);
    this.aS = new Float64Array(nt * nt);
    this.lam3 = new Float64Array(nt * nt * nt);
    this.eps3 = new Float64Array(nt * nt * nt);
    this.c03 = new Float64Array(nt * nt * nt);
    this.fileRead = false;
    this.entries.clear();
    this.fileElems.clear();
    this.unitScale = 1;
    this.fileName = '';
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError(`usage: pair_coeff I J filename elem1 ... elemN (style ${this.name})`);
    const filename = args[2];
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError(`pair_coeff for style ${this.name} needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    }
    if (this.threebody) {
      if (args[0] !== '*' || args[1] !== '*') {
        throw new StyleError(`the first 2 arguments of pair_coeff for style ${this.name} must be * *`);
      }
    }
    if (!this.fileRead) {
      this.readSWFile(filename, ctx);
    } else if (filename !== this.fileName) {
      throw new StyleError(`all pair_coeff commands for style ${this.name} must use the same potential file (first: ${this.fileName})`);
    }
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      let idx: number;
      if (name === 'NULL') {
        idx = -1;
      } else {
        if (!this.fileElems.has(name)) {
          throw new StyleError(`element '${name}' is not in SW potential file ${this.fileName} (elements: ${[...this.fileElems].sort().join(' ')})`);
        }
        idx = this.elemNames.indexOf(name);
        if (idx < 0) {
          idx = this.elemNames.length;
          this.elemNames.push(name);
        }
      }
      if (this.mappingSet && !this.threebody && this.elemOf[t] !== idx) {
        throw new StyleError(`the element to atom type mappings must be consistent across all pair_coeff statements (type ${t}: '${this.elemOf[t] < 0 ? 'NULL' : this.elemNames[this.elemOf[t]]}' vs '${name}')`);
      }
      this.elemOf[t] = idx;
    }
    this.mappingSet = true;
    if (!this.threebody) {
      const nt = this.ntypes + 1;
      const [ilo, ihi] = typeBounds(args[0], this.ntypes);
      const [jlo, jhi] = typeBounds(args[1], this.ntypes);
      for (let i = ilo; i <= ihi; i++) {
        for (let j = jlo; j <= jhi; j++) {
          this.active[i * nt + j] = 1;
          this.active[j * nt + i] = 1;
        }
      }
    }
  }

  private readSWFile(filename: string, ctx: StyleContext): void {
    const text = ctx.readFile(filename);
    const parsed = parseSWFile(text, filename);
    const scale = unitScaleOf(parsed.unitTag, ctx, filename);
    if (scale !== 1) ctx.log(`WARNING: converting SW potential file ${filename} from ${parsed.unitTag} to ${ctx.s?.units.style} units`);
    this.entries = parsed.entries;
    this.fileElems = parsed.elems;
    this.unitScale = scale;
    this.fileName = filename;
    this.fileRead = true;
  }

  protected entry(k: string, what: string): SWEntry {
    const e = this.entries.get(k);
    if (!e) {
      throw new StyleError(`SW potential file ${this.fileName} has no entry for elements ${k} (${what})`);
    }
    return e;
  }

  protected checkTol(e: SWEntry, k: string): void {
    if (e.tol !== 0) {
      throw new StyleError(`SW potential file ${this.fileName} entry ${k} has tol = ${e.tol}; the tol-based virtual cutoff is not supported by this engine (only tol = 0.0, the standard cutoff)`);
    }
  }

  override initStyle(ctx: StyleContext): void {
    if (this.shift || this.tail) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify shift and tail options`);
    }
    if (this.table !== 12) {
      throw new StyleError(`pair_style ${this.name} does not support the pair_modify table option`);
    }
    if (!this.fileRead) {
      throw new StyleError(`pair_style ${this.name} needs a pair_coeff command with a Stillinger-Weber potential file`);
    }
    if (!this.threebody) return;
    const nt = this.ntypes + 1;
    for (let ti = 1; ti <= this.ntypes; ti++) {
      if (this.elemOf[ti] < 0) continue;
      for (let tj = 1; tj <= this.ntypes; tj++) {
        if (this.elemOf[tj] < 0) continue;
        for (let tk = 1; tk <= this.ntypes; tk++) {
          if (this.elemOf[tk] < 0) continue;
          const k = key3(this.elemNames[this.elemOf[ti]], this.elemNames[this.elemOf[tj]], this.elemNames[this.elemOf[tk]]);
          const e = this.entry(k, `needed by the three-body combination of types ${ti} ${tj} ${tk}`);
          this.checkTol(e, k);
          const t3 = (ti * nt + tj) * nt + tk;
          this.lam3[t3] = e.lambda;
          this.eps3[t3] = e.eps * this.unitScale;
          this.c03[t3] = e.costheta0;
        }
      }
    }
  }

  override initOne(i: number, j: number): number {
    const ei = this.elemOf[i];
    const ej = this.elemOf[j];
    if (ei < 0 || ej < 0) return 0;
    const nt = this.ntypes + 1;
    if (!this.threebody && this.active[i * nt + j] === 0) return 0;
    const e1 = this.elemNames[ei];
    const e2 = this.elemNames[ej];
    const k = key3(e1, e2, e2);
    const e = this.entry(k, `needed by the two-body interaction of types ${i} ${j}`);
    this.checkTol(e, k);
    const scale = this.unitScale;
    // two-body pair parameters (both orders) from the (i,j,j) entry
    this.eps2[i * nt + j] = this.eps2[j * nt + i] = e.eps * scale;
    this.sigma2[i * nt + j] = this.sigma2[j * nt + i] = e.sigma;
    this.a2[i * nt + j] = this.a2[j * nt + i] = e.a;
    this.pA[i * nt + j] = this.pA[j * nt + i] = e.A;
    this.pB[i * nt + j] = this.pB[j * nt + i] = e.B;
    this.pP[i * nt + j] = this.pP[j * nt + i] = e.p;
    this.pQ[i * nt + j] = this.pQ[j * nt + i] = e.q;
    // the two-body term is zero beyond its own entry's a*sigma
    this.cut2[i * nt + j] = this.cut2[j * nt + i] = e.a * e.sigma;
    // screening leg center i -> neighbor j from the same entry
    this.gam[i * nt + j] = e.gamma;
    this.sigS[i * nt + j] = e.sigma;
    this.aS[i * nt + j] = e.a;
    if (!this.threebody) return e.a * e.sigma;
    const kr = key3(e2, e1, e1);
    const er = this.entry(kr, `needed by the screening of the leg from type ${j} to type ${i}`);
    this.checkTol(er, kr);
    this.gam[j * nt + i] = er.gamma;
    this.sigS[j * nt + i] = er.sigma;
    this.aS[j * nt + i] = er.a;
    // the neighbor list must cover both legs' three-body screening cutoffs
    return Math.max(e.a * e.sigma, er.a * er.sigma);
  }

  /** Two-body energy phi2(r) and radial force coefficient fpair = -dphi2/dr / r (> 0 repulsive). */
  private twoBody(t: number, r: number, rsq: number): { phi2: number; fpair: number } {
    const eps = this.eps2[t], sig = this.sigma2[t], a = this.a2[t];
    const A = this.pA[t], B = this.pB[t], p = this.pP[t], q = this.pQ[t];
    const u = sig / r;
    const den = r - a * sig;
    const es = Math.exp(sig / den);
    const f2 = B * Math.pow(u, p) - Math.pow(u, q);
    const dfdu = (p > 0 ? B * p * Math.pow(u, p - 1) : 0) - (q > 0 ? q * Math.pow(u, q - 1) : 0);
    // dphi2/dr = A*eps*exp(sigma/(r-a*sigma)) * (-dfdu*sigma/r^2 - f2*sigma/(r-a*sigma)^2)
    const dphi2 = A * eps * es * (-dfdu * sig / rsq - f2 * sig / (den * den));
    return { phi2: A * eps * f2 * es, fpair: -dphi2 / r };
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const elemOf = this.elemOf;
    const three = this.threebody;
    const eatom = pc.eatom;
    const vatom = pc.vatom;
    let evdwl = 0;

    let maxn = 0;
    for (let i = 0; i < list.inum; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    if (maxn > this.cap) {
      this.cap = maxn;
      this.sIdx = new Int32Array(maxn);
      this.sDx = new Float64Array(maxn);
      this.sDy = new Float64Array(maxn);
      this.sDz = new Float64Array(maxn);
      this.sR = new Float64Array(maxn);
      this.sInvR = new Float64Array(maxn);
      this.sES = new Float64Array(maxn);
      this.sW = new Float64Array(maxn);
    }
    const nIdx = this.sIdx, nDx = this.sDx, nDy = this.sDy, nDz = this.sDz;
    const nR = this.sR, nInvR = this.sInvR, nES = this.sES, nW = this.sW;

    for (let i = 0; i < nlocal; i++) {
      const tiv = type[i];
      if (elemOf[tiv] < 0) continue;
      const ti = tiv * nt;
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      let m = 0;
      let fxi = 0, fyi = 0, fzi = 0;
      // pass 1: two-body terms (half per visit; the pair is seen again from the other end) and gather
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dxx = x[3 * j] - x[3 * i];
        const dyy = x[3 * j + 1] - x[3 * i + 1];
        const dzz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dxx * dxx + dyy * dyy + dzz * dzz;
        const tj = type[j];
        if (rsq >= cutsq[ti + tj]) continue;
        const r = Math.sqrt(rsq);
        const c2 = this.cut2[ti + tj];
        if (rsq < c2 * c2) {
          const { phi2, fpair } = this.twoBody(ti + tj, r, rsq);
          const hx = 0.5 * fpair * dxx;
          const hy = 0.5 * fpair * dyy;
          const hz = 0.5 * fpair * dzz;
          fxi -= hx;
          fyi -= hy;
          fzi -= hz;
          f[3 * j] += hx;
          f[3 * j + 1] += hy;
          f[3 * j + 2] += hz;
          evdwl += 0.5 * phi2;
          if (eatom) {
            const e4 = 0.25 * phi2;
            eatom[i] += e4;
            eatom[j] += e4;
          }
          if (vatom) {
            const vc = 0.25 * fpair;
            const v0 = vc * dxx * dxx, v1 = vc * dyy * dyy, v2 = vc * dzz * dzz;
            const v3 = vc * dxx * dyy, v4 = vc * dxx * dzz, v5 = vc * dyy * dzz;
            vatom[6 * i] += v0; vatom[6 * i + 1] += v1; vatom[6 * i + 2] += v2;
            vatom[6 * i + 3] += v3; vatom[6 * i + 4] += v4; vatom[6 * i + 5] += v5;
            vatom[6 * j] += v0; vatom[6 * j + 1] += v1; vatom[6 * j + 2] += v2;
            vatom[6 * j + 3] += v3; vatom[6 * j + 4] += v4; vatom[6 * j + 5] += v5;
          }
        }
        if (three) {
          // screening of the leg center i -> j: exp(gamma*sigma/(r - a*sigma)) of
          // entry (EI,EJ,EJ); it is zero beyond that entry's own a*sigma (beyond
          // it the exponent would be positive and overflow, so the leg is dead)
          const g = this.gam[ti + tj], sgs = this.sigS[ti + tj], sa = this.aS[ti + tj];
          const legCut = sa * sgs;
          if (r < legCut) {
            const d2 = r - legCut;
            nIdx[m] = j;
            nDx[m] = dxx; nDy[m] = dyy; nDz[m] = dzz;
            nR[m] = r;
            nInvR[m] = 1 / r;
            nES[m] = Math.exp((g * sgs) / d2);
            nW[m] = (g * sgs) / (d2 * d2);
            m++;
          }
        }
      }
      // pass 2: three-body terms, once per (center i; unordered neighbor pair)
      if (three && m > 1) {
        const t3base = tiv * nt * nt;
        for (let p1 = 0; p1 < m; p1++) {
          const j = nIdx[p1];
          const tjv = type[j];
          const d1x = nDx[p1], d1y = nDy[p1], d1z = nDz[p1];
          const invrj = nInvR[p1], esj = nES[p1], wj = nW[p1];
          const base1 = t3base + tjv * nt;
          for (let p2 = p1 + 1; p2 < m; p2++) {
            const k = nIdx[p2];
            const t3 = base1 + type[k];
            const lam = this.lam3[t3];
            if (lam === 0) continue;
            const d2x = nDx[p2], d2y = nDy[p2], d2z = nDz[p2];
            const invrk = nInvR[p2], esk = nES[p2], wk = nW[p2];
            const c = (d1x * d2x + d1y * d2y + d1z * d2z) * invrj * invrk;
            const dc = c - this.c03[t3];
            const fc = this.angleScale(dc);
            const zeta = fc * dc;
            const zeta2 = zeta * zeta;
            const P = lam * this.eps3[t3] * esj * esk;
            const e3v = P * zeta2;
            evdwl += e3v;
            // fj = aj*ej - bk*ek, fk = ak*ek - bk*ej, fi = -(fj + fk)
            const zf = 2 * zeta * fc;
            const bk = P * zf * invrj * invrk;
            const aj = P * (zf * c * invrj * invrj + zeta2 * wj * invrj);
            const ak = P * (zf * c * invrk * invrk + zeta2 * wk * invrk);
            const fxj = aj * d1x - bk * d2x;
            const fyj = aj * d1y - bk * d2y;
            const fzj = aj * d1z - bk * d2z;
            const fxk = ak * d2x - bk * d1x;
            const fyk = ak * d2y - bk * d1y;
            const fzk = ak * d2z - bk * d1z;
            f[3 * j] += fxj;
            f[3 * j + 1] += fyj;
            f[3 * j + 2] += fzj;
            f[3 * k] += fxk;
            f[3 * k + 1] += fyk;
            f[3 * k + 2] += fzk;
            fxi -= fxj + fxk;
            fyi -= fyj + fyk;
            fzi -= fzj + fzk;
            if (eatom) {
              const e9 = e3v / 3;
              eatom[i] += e9;
              eatom[j] += e9;
              eatom[k] += e9;
            }
            if (vatom) {
              const va = vatom;
              const w0 = (d1x * fxj + d2x * fxk) / 3;
              const w1 = (d1y * fyj + d2y * fyk) / 3;
              const w2 = (d1z * fzj + d2z * fzk) / 3;
              const w3 = (d1x * fyj + d2x * fyk) / 3;
              const w4 = (d1x * fzj + d2x * fzk) / 3;
              const w5 = (d1y * fzj + d2y * fzk) / 3;
              va[6 * i] += w0; va[6 * i + 1] += w1; va[6 * i + 2] += w2;
              va[6 * i + 3] += w3; va[6 * i + 4] += w4; va[6 * i + 5] += w5;
              va[6 * j] += w0; va[6 * j + 1] += w1; va[6 * j + 2] += w2;
              va[6 * j + 3] += w3; va[6 * j + 4] += w4; va[6 * j + 5] += w5;
              va[6 * k] += w0; va[6 * k + 1] += w1; va[6 * k + 2] += w2;
              va[6 * k + 3] += w3; va[6 * k + 4] += w4; va[6 * k + 5] += w5;
            }
          }
        }
      }
      f[3 * i] += fxi;
      f[3 * i + 1] += fyi;
      f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  /**
   * "The single() function of the sw pair style is only enabled and supported
   * for the case of the threebody off setting" — the two-body term alone.
   */
  override single(_i: number, _j: number, itype: number, jtype: number, rsq: number): { eng: number; fforce: number } {
    if (this.threebody && !this.isMod) {
      throw new StyleError(`pair style ${this.name}: single() is only supported with threebody off`);
    }
    const t = itype * (this.ntypes + 1) + jtype;
    if (rsq >= this.cutsq[t]) return { eng: 0, fforce: 0 };
    const r = Math.sqrt(rsq);
    const { phi2, fpair } = this.twoBody(t, r, rsq);
    return { eng: phi2, fforce: fpair };
  }
}

/**
 * pair_style sw — the Stillinger-Weber potential. "The default value for the
 * *threebody* setting of the "sw" pair style is "on"".
 */
export class PairSW extends PairSWBase {
  readonly name = 'sw';

  constructor() {
    super(false);
  }
}

/**
 * pair_style sw/mod — the Stillinger-Weber potential with the three-body
 * angle difference delta scaled by the switching factor f_C(delta)
 * ("maxdelcs" delta1 delta2, defaults 0.25 and 0.35).
 */
export class PairSWMod extends PairSWBase {
  readonly name = 'sw/mod';
  private delta1 = 0.25;
  private delta2 = 0.35;

  constructor() {
    super(true);
  }

  protected override setDeltas(d1: number, d2: number): void {
    this.delta1 = d1;
    this.delta2 = d2;
  }

  /** f_C(delta) of the doc page (verbatim formula in the file header). */
  protected override angleScale(dc: number): number {
    const ad = Math.abs(dc);
    if (ad <= this.delta1) return 1;
    if (ad >= this.delta2) return 0;
    return 0.5 + 0.5 * Math.cos((Math.PI * (ad - this.delta1)) / (this.delta2 - this.delta1));
  }
}
