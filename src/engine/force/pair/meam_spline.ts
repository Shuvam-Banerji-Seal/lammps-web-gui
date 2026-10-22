import { Pair, StyleError, type PairCompute, type StyleContext, potentialFile } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum } from '../util';

/*
 * pair_style meam/spline and meam/sw/spline: docs.lammps.org/pair_meam_spline.html and
 * docs.lammps.org/pair_meam_sw_spline.html (sources: plans/lammps-docs-live/pair_meam_spline.rst and
 * plans/lammps-docs-live/pair_meam_sw_spline.rst). The energy forms are written from the documentation,
 * the Lenosky et al. (2000) spline MEAM form and black-box measurements with native LAMMPS listed below.
 *
 * Doc sentences quoted here (verbatim from the pages):
 * "cutoffs for spline-based MEAM potentials are not set in the pair_style or pair_coeff command; they are specified in the potential files themselves."
 * "This pair style does not support mapping multiple atom types to the same element."
 * "The meam/sw/spline style currently supports only single-element MEAM+SW potentials."
 * "If a mapping value is specified as NULL, the mapping is not performed."
 * "All of these files are parameterized in terms of LAMMPS metal units."
 *
 * Energy of the single-element (old style) potential, written here in our own notation:
 *   E = sum_{i<j} phi(r_ij) + sum_i [U(n_i) - U(0)]
 *   n_i = sum_j rho(r_ij) + sum_{j<k} f(r_ij) f(r_ik) g(cos theta_jik)
 * The -U(0) per atom is measured (see below). The multicomponent form uses phi_ab and g_ab for each element
 * pair, rho_b and f_b for each neighbour element b and U_a for the centre atom a. meam/sw/spline adds
 * E_SW = sum_i sum_{j<k} F(r_ij) F(r_ik) G(cos theta_jik), with the same j<k convention, and its file holds
 * F and G before rho, U, f and g.
 *
 * File layout (read from the distributed files):
 *   - line 1 is a comment;
 *   - old style: each spline is N (the knot count), a line with the two end slopes d0 dN, four 0/1 flags,
 *     and N lines x y y2; the blocks follow in the order phi, rho, U, f, g (meam/spline) or phi, F, G,
 *     rho, U, f, g (meam/sw/spline);
 *   - new style: line 2 starts with meam/spline and gives the element count and names; each spline is a
 *     spline3eq line, N, d0 dN and N knot lines (no flags); the blocks follow in the order phi, rho, U, f,
 *     g with N(N+1)/2 phi and g functions and N rho, U and f functions.
 * The cutoff of each spline is its last knot x.
 *
 * Measured with native LAMMPS (black box), on the distributed files and on constructed probes:
 *   - Clamped splines. The third column of a knot line is not used: recomputing the cubic spline from
 *     (x, y, d0, dN) reproduces every energy. Ti.meam.sw.spline has a first-knot third column of 0, where the
 *     clamped value is 99.49, and the energies follow the clamped spline. The four flags have no effect:
 *     Si_1.meam.spline with all flags set to 0 or to 1 gives the identical energy.
 *   - Beyond the last knot a function is zero (its cutoff). Below the first knot, and for densities and cos
 *     values outside the knot range (U and g), the spline continues linearly with its end slope. Dimer and
 *     cluster probes on TiO.meam.spline agree to 1e-14.
 *   - Embedding shift: an isolated atom has energy 0 natively, and a dimer is lower by 2 U(0) than
 *     phi + 2 U(rho) with the file's U. Each atom therefore contributes U(n_i) - U(0).
 *   - Angular sum: over unordered neighbour pairs j<k. Ordered j != k sums differ from native by 0.07 to 1.7
 *     on the clusters tested. The same j<k sum is used by the F G term of meam/sw/spline (Si.b.meam.sw.spline
 *     and Ti.meam.sw.spline agree with native to 1e-14).
 *   - Multicomponent indexing: phi and g use the pair index (0,0), (0,1), (1,1). rho and f come from the
 *     neighbour's element, U from the centre atom's element; swapping rho and f changes the TiO energies by
 *     1 to 3 eV. Atom type t is the t-th element of the file whatever the order of the names in pair_coeff
 *     (TiO.meam.spline with the names O Ti and with Ti O give the same energies, and a renamed copy with the
 *     names Al Zr maps type 1 to the first element of the file). Only two-element new-style files are
 *     accepted, since the pair order of three or more elements is not measured.
 *   - Refusals natively: a units style other than metal is an error (requires metal units); two atom types on
 *     one element is an error; a type count different from the element count is an error (one atom type per
 *     element); an element name missing from a new-style file is an error (No matching element); a
 *     meam/sw/spline file with two different element names is an error (only supports single element
 *     potentials); a meam/sw/spline file read as a meam/spline file is accepted natively, here it is refused
 *     because the extra spline blocks cannot be assigned. An old-style meam/spline file accepts any element name.
 *
 * Forces and virial: analytic (spline slopes and the derivative of the cosine). The global virial comes from
 * the dot product of positions and forces (virialFdotr). Per-atom energy and virial are split as in the sw style.
 *
 * Neighbour lists: a full list. The pair term is summed over the full list with half the energy and force per
 * visit. Each owned atom evaluates its embedding energy from its own neighbours, and the forces on ghost
 * neighbours are folded back by the force field.
 */

/** Clamped cubic spline through the knots of one file block. */
export class CubicSpline {
  readonly n: number;
  readonly x: Float64Array;
  readonly y: Float64Array;
  /** Second derivatives recomputed from the end slopes (clamped). */
  readonly y2: Float64Array;
  readonly xmin: number;
  readonly xmax: number;

  constructor(readonly label: string, knots: number[][], readonly d0: number, readonly dN: number) {
    this.n = knots.length;
    this.x = Float64Array.from(knots, (k) => k[0]);
    this.y = Float64Array.from(knots, (k) => k[1]);
    this.xmin = this.x[0];
    this.xmax = this.x[this.n - 1];
    this.y2 = clampedSecondDerivatives(this.x, this.y, d0, dN);
  }

  /** Value at v; linear with the end slope outside the knots. */
  value(v: number): number {
    const { x, y, y2, n } = this;
    if (v < this.xmin) return y[0] + this.d0 * (v - x[0]);
    if (v > this.xmax) return y[n - 1] + this.dN * (v - x[n - 1]);
    const i = this.interval(v);
    const h = x[i + 1] - x[i];
    const A = (x[i + 1] - v) / h, B = (v - x[i]) / h;
    return A * y[i] + B * y[i + 1] + ((A * A * A - A) * y2[i] + (B * B * B - B) * y2[i + 1]) * (h * h) / 6;
  }

  /** Derivative dy/dv; the end slope outside the knots. */
  slope(v: number): number {
    const { x, y, y2 } = this;
    if (v < this.xmin) return this.d0;
    if (v > this.xmax) return this.dN;
    const i = this.interval(v);
    const h = x[i + 1] - x[i];
    const A = (x[i + 1] - v) / h, B = (v - x[i]) / h;
    return (y[i + 1] - y[i]) / h - ((3 * A * A - 1) / 6) * h * y2[i] + ((3 * B * B - 1) / 6) * h * y2[i + 1];
  }

  private interval(v: number): number {
    const x = this.x;
    let lo = 0, hi = this.n - 1;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (x[m] > v) hi = m;
      else lo = m;
    }
    return lo;
  }
}

/** Second derivatives of the cubic spline with first derivatives d0 and dN at the ends (tridiagonal solve). */
const clampedSecondDerivatives = (x: Float64Array, y: Float64Array, d0: number, dN: number): Float64Array => {
  const n = x.length;
  const h = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) h[i] = x[i + 1] - x[i];
  const a = new Float64Array(n), b = new Float64Array(n), c = new Float64Array(n), r = new Float64Array(n);
  b[0] = 2 * h[0];
  c[0] = h[0];
  r[0] = 6 * ((y[1] - y[0]) / h[0] - d0);
  for (let i = 1; i < n - 1; i++) {
    a[i] = h[i - 1];
    b[i] = 2 * (h[i - 1] + h[i]);
    c[i] = h[i];
    r[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
  }
  a[n - 1] = h[n - 2];
  b[n - 1] = 2 * h[n - 2];
  r[n - 1] = 6 * (dN - (y[n - 1] - y[n - 2]) / h[n - 2]);
  const cp = new Float64Array(n), dp = new Float64Array(n);
  cp[0] = c[0] / b[0];
  dp[0] = r[0] / b[0];
  for (let i = 1; i < n; i++) {
    const m = b[i] - a[i] * cp[i - 1];
    cp[i] = c[i] / m;
    dp[i] = (r[i] - a[i] * dp[i - 1]) / m;
  }
  const y2 = new Float64Array(n);
  y2[n - 1] = dp[n - 1];
  for (let i = n - 2; i >= 0; i--) y2[i] = dp[i] - cp[i] * y2[i + 1];
  return y2;
};

/** Parsed potential file. */
export interface MeamSplineFile {
  /** Element names (new style) or [] for the single-element old style. */
  readonly names: readonly string[];
  readonly multi: boolean;
  readonly phi: CubicSpline[];
  readonly rho: CubicSpline[];
  readonly U: CubicSpline[];
  readonly f: CubicSpline[];
  readonly g: CubicSpline[];
  /** meam/sw/spline only. */
  readonly F: CubicSpline | null;
  readonly G: CubicSpline | null;
}

/** Index of the unordered element pair (a, b) in the phi and g lists: (0,0),(0,1),(1,1),(0,2),... */
export const pairIndex = (a: number, b: number, nel: number): number => {
  const lo = Math.min(a, b), hi = Math.max(a, b);
  return lo * nel - (lo * (lo - 1)) / 2 + (hi - lo);
};

/** Line cursor that skips blank lines and reports the line number in errors. */
class Lines {
  private i = 0;
  constructor(private readonly lines: string[], private readonly fileName: string) {}
  next(what: string): { words: string[]; no: number } {
    while (this.i < this.lines.length && this.lines[this.i].trim() === '') this.i++;
    if (this.i >= this.lines.length) throw new StyleError(`potential file ${this.fileName} ends early (expected ${what})`);
    const no = this.i + 1;
    const words = this.lines[this.i].trim().split(/\s+/);
    this.i++;
    return { words, no };
  }
  rest(): boolean {
    for (let k = this.i; k < this.lines.length; k++) if (this.lines[k].trim() !== '') return true;
    return false;
  }
}

const readKnots = (cur: Lines, n: number, label: string, fileName: string): number[][] => {
  const knots: number[][] = [];
  for (let k = 0; k < n; k++) {
    const { words, no } = cur.next(`knot ${k + 1} of ${label}`);
    if (words.length < 3) throw new StyleError(`potential file ${fileName} line ${no}: a knot of ${label} needs x y y2, got ${words.length} values`);
    const v = [0, 1, 2].map((c) => parseNum(words[c], `knot ${k + 1} of ${label} in ${fileName}`));
    if (knots.length > 0 && !(v[0] > knots[knots.length - 1][0])) {
      throw new StyleError(`potential file ${fileName} line ${no}: the knots of ${label} must increase in x`);
    }
    knots.push(v);
  }
  return knots;
};

/** One spline block: knot count, end slopes, optional flags (old style), knots. */
const readBlock = (cur: Lines, label: string, fileName: string, flags: boolean, eq: boolean): CubicSpline => {
  if (eq) {
    const { words, no } = cur.next(`spline3eq line of ${label}`);
    if (words[0] !== 'spline3eq') throw new StyleError(`potential file ${fileName} line ${no}: expected spline3eq before ${label}, got '${words[0]}'`);
  }
  const { words: nw, no: nno } = cur.next(`knot count of ${label}`);
  const n = Number(nw[0]);
  if (!Number.isInteger(n) || n < 2) throw new StyleError(`potential file ${fileName} line ${nno}: ${label} needs an integer knot count of at least 2`);
  const { words: dw, no: dno } = cur.next(`end slopes of ${label}`);
  if (dw.length < 2) throw new StyleError(`potential file ${fileName} line ${dno}: ${label} needs the two end slopes`);
  const d0 = parseNum(dw[0], `start slope of ${label} in ${fileName}`);
  const dN = parseNum(dw[1], `end slope of ${label} in ${fileName}`);
  if (flags) {
    const { words: fw, no: fno } = cur.next(`flags of ${label}`);
    if (fw.length !== 4 || fw.some((w) => !/^[01]$/.test(w))) {
      throw new StyleError(`potential file ${fileName} line ${fno}: ${label} needs four 0/1 flags`);
    }
  }
  const knots = readKnots(cur, n, label, fileName);
  return new CubicSpline(label, knots, d0, dN);
};

/**
 * Parses a meam/spline (old or new style) or meam/sw/spline file. The block count is fixed by the style and by
 * the element count, and trailing blocks are refused rather than read.
 */
export const parseMeamSplineFile = (text: string, fileName: string, sw: boolean): MeamSplineFile => {
  const lines = text.split(/\r?\n/);
  const second = (lines[1] ?? '').trim();
  const cur = new Lines(lines.slice(2), fileName);
  if (second.startsWith('meam/spline')) {
    if (sw) throw new StyleError(`pair_style meam/sw/spline needs a single-element file; ${fileName} is a multicomponent meam/spline file`);
    const hw = second.split(/\s+/);
    const nel = Number(hw[1]);
    const names = hw.slice(2);
    if (!Number.isInteger(nel) || nel < 1 || names.length !== nel) {
      throw new StyleError(`potential file ${fileName} line 2: expected 'meam/spline N El1 ... ElN' with N names`);
    }
    if (new Set(names).size !== nel) throw new StyleError(`potential file ${fileName} line 2: element names must be distinct`);
    if (nel > 2) {
      throw new StyleError(`potential file ${fileName} has ${nel} elements; this engine only supports two-element multicomponent meam/spline files (the pair index order of three or more elements is not measured)`);
    }
    const npair = (nel * (nel + 1)) / 2;
    const phi = [], rho = [], U = [], f = [], g = [];
    for (let t = 0; t < npair; t++) phi.push(readBlock(cur, `phi block ${t + 1}`, fileName, false, true));
    for (let t = 0; t < nel; t++) rho.push(readBlock(cur, `rho block ${t + 1}`, fileName, false, true));
    for (let t = 0; t < nel; t++) U.push(readBlock(cur, `U block ${t + 1}`, fileName, false, true));
    for (let t = 0; t < nel; t++) f.push(readBlock(cur, `f block ${t + 1}`, fileName, false, true));
    for (let t = 0; t < npair; t++) g.push(readBlock(cur, `g block ${t + 1}`, fileName, false, true));
    if (cur.rest()) throw new StyleError(`potential file ${fileName} has more spline blocks than its ${nel} elements need`);
    return { names, multi: true, phi, rho, U, f, g, F: null, G: null };
  }
  // old style: the first line of the file is a comment and the first block starts at line 2
  const oldCur = new Lines(lines.slice(1), fileName);
  if (sw) {
    const phi = readBlock(oldCur, 'phi', fileName, true, false);
    const F = readBlock(oldCur, 'F', fileName, true, false);
    const G = readBlock(oldCur, 'G', fileName, true, false);
    const rho = readBlock(oldCur, 'rho', fileName, true, false);
    const U = readBlock(oldCur, 'U', fileName, true, false);
    const f = readBlock(oldCur, 'f', fileName, true, false);
    const g = readBlock(oldCur, 'g', fileName, true, false);
    if (oldCur.rest()) throw new StyleError(`potential file ${fileName} has more spline blocks than the seven of meam/sw/spline`);
    return { names: [], multi: false, phi: [phi], rho: [rho], U: [U], f: [f], g: [g], F, G };
  }
  const phi = readBlock(oldCur, 'phi', fileName, true, false);
  const rho = readBlock(oldCur, 'rho', fileName, true, false);
  const U = readBlock(oldCur, 'U', fileName, true, false);
  const f = readBlock(oldCur, 'f', fileName, true, false);
  const g = readBlock(oldCur, 'g', fileName, true, false);
  if (oldCur.rest()) {
    throw new StyleError(`potential file ${fileName} has more spline blocks than the five of meam/spline (use pair_style meam/sw/spline for a MEAM+SW file)`);
  }
  return { names: [], multi: false, phi: [phi], rho: [rho], U: [U], f: [f], g: [g], F: null, G: null };
};

/** Shared implementation of meam/spline and meam/sw/spline. */
abstract class PairMeamSplineBase extends Pair {
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  protected fileName = '';
  protected fileRead = false;
  protected file: MeamSplineFile | null = null;
  /** Element index of each atom type, or -1 when unmapped. */
  protected elemOf = new Int32Array(0);
  /** Largest cutoff of the file's functions (the neighbour list cutoff of every type pair). */
  protected globalCut = 0;
  /** U(0) per element, subtracted from each embedding energy. */
  protected u0 = new Float64Array(0);
  protected mappingSet = false;

  protected abstract readonly sw: boolean;

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length > 0) throw new StyleError(`pair_style ${this.name} takes no arguments (got '${args[0]}')`);
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.elemOf = new Int32Array(ntypes + 1).fill(-1);
    this.fileRead = false;
    this.file = null;
    this.fileName = '';
    this.mappingSet = false;
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError(`usage: pair_coeff * * filename elem1 ... elemN (style ${this.name})`);
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError(`the first 2 arguments of pair_coeff for style ${this.name} must be * *`);
    const filename = args[2];
    const names = args.slice(3);
    if (names.length !== this.ntypes) {
      throw new StyleError(this.name === 'meam/spline'
        // As native LAMMPS does (src/MANYBODY/pair_meam_spline.cpp coeff, its own count check; pair_meam_sw_spline.cpp
        // calls Pair::map_element2type, at b332f28)
        ? 'Incorrect args for pair coefficients'
        : 'Number of element to type mappings does not match number of atom types');
    }
    if (names.includes('NULL')) {
      throw new StyleError(`pair_style ${this.name} does not support NULL mappings in this engine (the NULL placeholder of the documentation is refused, not approximated)`);
    }
    if (!this.fileRead) {
      const text = ctx.readFile(filename, potentialFile(this.name));
      this.file = parseMeamSplineFile(text, filename, this.sw);
      this.fileName = filename;
      this.fileRead = true;
    } else if (filename !== this.fileName) {
      throw new StyleError(`all pair_coeff commands for style ${this.name} must use the same potential file (first: ${this.fileName})`);
    }
    const file = this.file as MeamSplineFile;
    if (file.multi) {
      // "Systems where there would be multiple atom types assigned to the same element are not supported"
      const nel = file.names.length;
      // native text (src/MANYBODY/pair_meam_spline.cpp coeff at b332f28)
      if (this.ntypes !== nel) throw new StyleError('Pair style meam/spline requires one atom type per element');
      for (const name of names) {
        if (file.names.indexOf(name) < 0) throw new StyleError(`element '${name}' is not in potential file ${filename} (elements: ${file.names.join(' ')})`);
      }
      if (new Set(names).size !== nel) throw new StyleError(`pair style ${this.name} requires one atom type per element`);
      // Measured with native LAMMPS (black box): the names after the file only have to exist; atom type t uses the
      // t-th element of the file whatever the order of the names (TiO.meam.spline with the names O Ti and with the
      // names Ti O give the same energies, and a renamed copy with the names Al Zr also maps type 1 to the first
      // element of the file).
      for (let t = 1; t <= this.ntypes; t++) this.elemOf[t] = t - 1;
    } else if (this.sw) {
      const distinct = new Set(names);
      if (distinct.size > 1) throw new StyleError(`pair_style ${this.name} only supports single element potentials (got ${[...distinct].join(' ')})`);
      for (let t = 1; t <= this.ntypes; t++) this.elemOf[t] = 0;
    } else {
      if (this.ntypes !== 1) throw new StyleError(`pair style ${this.name} requires one atom type per element (1 element, ${this.ntypes} types)`);
      this.elemOf[1] = 0;
    }
    this.mappingSet = true;
  }

  override initStyle(ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError(`pair_style ${this.name} does not support the pair_modify shift and tail options`);
    if (this.table !== 12) throw new StyleError(`pair_style ${this.name} does not support the pair_modify table option`);
    if (!this.fileRead || !this.file || !this.mappingSet) {
      throw new StyleError(`pair_style ${this.name} needs a pair_coeff command with a potential file`);
    }
    const style = ctx.s?.units.style;
    if (style !== undefined && style !== 'metal') {
      throw new StyleError(`potential file ${this.fileName} requires metal units but ${style} units are in use`);
    }
    const file = this.file;
    let cut = 0;
    const scan = (s: CubicSpline) => { if (s.xmax > cut) cut = s.xmax; };
    file.phi.forEach(scan);
    file.rho.forEach(scan);
    file.f.forEach(scan);
    if (file.F) scan(file.F);
    this.globalCut = cut;
    const nel = file.multi ? file.names.length : 1;
    this.u0 = new Float64Array(nel);
    for (let e = 0; e < nel; e++) this.u0[e] = file.U[e].value(0);
  }

  override initOne(i: number, j: number): number {
    if (this.elemOf[i] < 0 || this.elemOf[j] < 0) return 0;
    return this.globalCut;
  }

  // scratch for the gathered neighbours of one owned atom and for the angular pairs (a < b)
  private cap = 0;
  private nJ = new Int32Array(0);
  private nDx = new Float64Array(0);
  private nDy = new Float64Array(0);
  private nDz = new Float64Array(0);
  private nR = new Float64Array(0);
  private nE = new Int32Array(0);
  private nFlag = new Uint8Array(0);
  private nRv = new Float64Array(0);
  private nRd = new Float64Array(0);
  private nFv = new Float64Array(0);
  private nFd = new Float64Array(0);
  private nSv = new Float64Array(0);
  private nSd = new Float64Array(0);
  private pA = new Int32Array(0);
  private pB = new Int32Array(0);
  private pC = new Float64Array(0);
  private pG = new Float64Array(0);
  private pGd = new Float64Array(0);
  private sA = new Int32Array(0);
  private sB = new Int32Array(0);
  private sC = new Float64Array(0);
  private sG = new Float64Array(0);
  private sGd = new Float64Array(0);

  private grow(maxn: number): void {
    if (maxn <= this.cap) return;
    this.cap = maxn;
    this.nJ = new Int32Array(maxn);
    this.nDx = new Float64Array(maxn);
    this.nDy = new Float64Array(maxn);
    this.nDz = new Float64Array(maxn);
    this.nR = new Float64Array(maxn);
    this.nE = new Int32Array(maxn);
    this.nFlag = new Uint8Array(maxn);
    this.nRv = new Float64Array(maxn);
    this.nRd = new Float64Array(maxn);
    this.nFv = new Float64Array(maxn);
    this.nFd = new Float64Array(maxn);
    this.nSv = new Float64Array(maxn);
    this.nSd = new Float64Array(maxn);
    const np = (maxn * (maxn - 1)) / 2 + 1;
    this.pA = new Int32Array(np);
    this.pB = new Int32Array(np);
    this.pC = new Float64Array(np);
    this.pG = new Float64Array(np);
    this.pGd = new Float64Array(np);
    this.sA = new Int32Array(np);
    this.sB = new Int32Array(np);
    this.sC = new Float64Array(np);
    this.sG = new Float64Array(np);
    this.sGd = new Float64Array(np);
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const file = this.file;
    if (!file) throw new Error(`pair style ${this.name} has no potential`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const cutsq = this.globalCut * this.globalCut;
    const nel = file.multi ? file.names.length : 1;
    const eatom = pc.eatom, vatom = pc.vatom;
    let maxn = 0;
    for (let i = 0; i < list.inum; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    this.grow(maxn);
    const { nJ, nDx, nDy, nDz, nR, nE, nFlag, nRv, nRd, nFv, nFd, nSv, nSd } = this;
    const { pA, pB, pC, pG, pGd, sA, sB, sC, sG, sGd } = this;
    const SWF = file.F, SWG = file.G;
    const multi = file.multi;
    let evdwl = 0;

    for (let i = 0; i < nlocal; i++) {
      const ei = this.elemOf[type[i]];
      if (ei < 0) continue;
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      let m = 0;
      let fxi = 0, fyi = 0, fzi = 0;
      // pass 1: pair term (half per visit); gather the neighbours that enter the density, the angular sums or
      // the three-body term, with their spline values and slopes (each spline is evaluated once per neighbour)
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = x[3 * j] - xi, dy = x[3 * j + 1] - yi, dz = x[3 * j + 2] - zi;
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq) continue;
        const ej = this.elemOf[type[j]];
        if (ej < 0) continue;
        const r = Math.sqrt(rsq);
        const phi = file.phi[multi ? pairIndex(ei, ej, nel) : 0];
        if (r < phi.xmax) {
          const v = phi.value(r);
          const fpair = -phi.slope(r) / r;
          const hx = 0.5 * fpair * dx, hy = 0.5 * fpair * dy, hz = 0.5 * fpair * dz;
          fxi -= hx; fyi -= hy; fzi -= hz;
          f[3 * j] += hx; f[3 * j + 1] += hy; f[3 * j + 2] += hz;
          evdwl += 0.5 * v;
          if (eatom) { eatom[i] += 0.25 * v; eatom[j] += 0.25 * v; }
          if (vatom) {
            const vc = 0.25 * fpair;
            const v0 = vc * dx * dx, v1 = vc * dy * dy, v2 = vc * dz * dz;
            const v3 = vc * dx * dy, v4 = vc * dx * dz, v5 = vc * dy * dz;
            vatom[6 * i] += v0; vatom[6 * i + 1] += v1; vatom[6 * i + 2] += v2;
            vatom[6 * i + 3] += v3; vatom[6 * i + 4] += v4; vatom[6 * i + 5] += v5;
            vatom[6 * j] += v0; vatom[6 * j + 1] += v1; vatom[6 * j + 2] += v2;
            vatom[6 * j + 3] += v3; vatom[6 * j + 4] += v4; vatom[6 * j + 5] += v5;
          }
        }
        const rhoS = file.rho[multi ? ej : 0];
        const fS = file.f[multi ? ej : 0];
        const inRho = r < rhoS.xmax;
        const inAng = r < fS.xmax;
        const inSW = SWF !== null && r < SWF.xmax;
        if (inRho || inAng || inSW) {
          nJ[m] = j; nDx[m] = dx; nDy[m] = dy; nDz[m] = dz; nR[m] = r; nE[m] = ej;
          nFlag[m] = (inRho ? 1 : 0) | (inAng ? 2 : 0) | (inSW ? 4 : 0);
          if (inRho) { nRv[m] = rhoS.value(r); nRd[m] = rhoS.slope(r); }
          if (inAng) { nFv[m] = fS.value(r); nFd[m] = fS.slope(r); }
          if (inSW) { nSv[m] = SWF!.value(r); nSd[m] = SWF!.slope(r); }
          m++;
        }
      }
      // angular pairs a < b of the embedding sum (f_a f_b g(cos)) and of the three-body term (F_a F_b G(cos))
      let np = 0, ns = 0;
      for (let a = 0; a < m; a++) {
        const fa = nFlag[a];
        if ((fa & 6) === 0) continue;
        for (let b = a + 1; b < m; b++) {
          const fb = nFlag[b];
          if ((fb & 6) === 0) continue;
          const c = (nDx[a] * nDx[b] + nDy[a] * nDy[b] + nDz[a] * nDz[b]) / (nR[a] * nR[b]);
          if (fa & 2 && fb & 2) {
            const gs = file.g[multi ? pairIndex(nE[a], nE[b], nel) : 0];
            pA[np] = a; pB[np] = b; pC[np] = c; pG[np] = gs.value(c); pGd[np] = gs.slope(c);
            np++;
          }
          if (fa & 4 && fb & 4 && SWG) {
            sA[ns] = a; sB[ns] = b; sC[ns] = c; sG[ns] = SWG.value(c); sGd[ns] = SWG.slope(c);
            ns++;
          }
        }
      }
      // embedding energy U(n_i) - U(0) with n_i = sum rho + sum_{a<b} f_a f_b g
      let n = 0;
      for (let a = 0; a < m; a++) if (nFlag[a] & 1) n += nRv[a];
      for (let p = 0; p < np; p++) n += nFv[pA[p]] * nFv[pB[p]] * pG[p];
      const U = file.U[ei];
      const eU = U.value(n) - this.u0[ei];
      const w = U.slope(n);
      evdwl += eU;
      if (eatom) eatom[i] += eU;
      // forces of the density terms (weight w = U'(n_i)) and virial of this atom's embedding cluster
      let W0 = 0, W1 = 0, W2 = 0, W3 = 0, W4 = 0, W5 = 0;
      for (let a = 0; a < m; a++) {
        if (!(nFlag[a] & 1)) continue;
        const dr = (w * nRd[a]) / nR[a];
        const fx = -dr * nDx[a], fy = -dr * nDy[a], fz = -dr * nDz[a];
        f[3 * nJ[a]] += fx; f[3 * nJ[a] + 1] += fy; f[3 * nJ[a] + 2] += fz;
        fxi -= fx; fyi -= fy; fzi -= fz;
        W0 += nDx[a] * fx; W1 += nDy[a] * fy; W2 += nDz[a] * fz;
        W3 += nDx[a] * fy; W4 += nDx[a] * fz; W5 += nDy[a] * fz;
      }
      // angular forces: T = f_a f_b g(c), E = w T; grad_a = w (f'_a f_b g) d_a/r_a + w f_a f_b g' dc/d_a
      for (let p = 0; p < np; p++) {
        const a = pA[p], b = pB[p];
        const ra = nR[a], rb = nR[b];
        const ax = nDx[a], ay = nDy[a], az = nDz[a];
        const bx = nDx[b], by = nDy[b], bz = nDz[b];
        const c = pC[p], g = pG[p], gp = pGd[p];
        const fa = nFv[a], fb = nFv[b], fap = nFd[a], fbp = nFd[b];
        const K = w * fa * fb * gp;
        const inv = 1 / (ra * rb), ira2 = 1 / (ra * ra), irb2 = 1 / (rb * rb);
        // dc/da = b/(ra rb) - c a/ra^2 ; dc/db = a/(ra rb) - c b/rb^2
        const gax = (w * fap * fb * g * ax) / ra + K * (bx * inv - c * ax * ira2);
        const gay = (w * fap * fb * g * ay) / ra + K * (by * inv - c * ay * ira2);
        const gaz = (w * fap * fb * g * az) / ra + K * (bz * inv - c * az * ira2);
        const gbx = (w * fa * fbp * g * bx) / rb + K * (ax * inv - c * bx * irb2);
        const gby = (w * fa * fbp * g * by) / rb + K * (ay * inv - c * by * irb2);
        const gbz = (w * fa * fbp * g * bz) / rb + K * (az * inv - c * bz * irb2);
        const ja = nJ[a], jb = nJ[b];
        f[3 * ja] -= gax; f[3 * ja + 1] -= gay; f[3 * ja + 2] -= gaz;
        f[3 * jb] -= gbx; f[3 * jb + 1] -= gby; f[3 * jb + 2] -= gbz;
        fxi += gax + gbx; fyi += gay + gby; fzi += gaz + gbz;
        W0 -= ax * gax + bx * gbx; W1 -= ay * gay + by * gby; W2 -= az * gaz + bz * gbz;
        W3 -= ax * gay + bx * gby; W4 -= ax * gaz + bx * gbz; W5 -= ay * gaz + by * gbz;
      }
      if (vatom) {
        // the cluster of this embedding energy (atom i and its gathered neighbours) shares its virial evenly
        const cnt = 1 + m;
        const q0 = W0 / cnt, q1 = W1 / cnt, q2 = W2 / cnt, q3 = W3 / cnt, q4 = W4 / cnt, q5 = W5 / cnt;
        vatom[6 * i] += q0; vatom[6 * i + 1] += q1; vatom[6 * i + 2] += q2;
        vatom[6 * i + 3] += q3; vatom[6 * i + 4] += q4; vatom[6 * i + 5] += q5;
        for (let a = 0; a < m; a++) {
          const q = nJ[a];
          vatom[6 * q] += q0; vatom[6 * q + 1] += q1; vatom[6 * q + 2] += q2;
          vatom[6 * q + 3] += q3; vatom[6 * q + 4] += q4; vatom[6 * q + 5] += q5;
        }
      }
      // Stillinger-Weber three-body term (meam/sw/spline only): E = sum_{a<b} F_a F_b G(c), once per triple
      for (let p = 0; p < ns; p++) {
        const a = sA[p], b = sB[p];
        const ra = nR[a], rb = nR[b];
        const ax = nDx[a], ay = nDy[a], az = nDz[a];
        const bx = nDx[b], by = nDy[b], bz = nDz[b];
        const c = sC[p], g = sG[p], gp = sGd[p];
        const Fa = nSv[a], Fb = nSv[b], Fap = nSd[a], Fbp = nSd[b];
        const e3 = Fa * Fb * g;
        evdwl += e3;
        const K = Fa * Fb * gp;
        const inv = 1 / (ra * rb), ira2 = 1 / (ra * ra), irb2 = 1 / (rb * rb);
        const gax = (Fap * Fb * g * ax) / ra + K * (bx * inv - c * ax * ira2);
        const gay = (Fap * Fb * g * ay) / ra + K * (by * inv - c * ay * ira2);
        const gaz = (Fap * Fb * g * az) / ra + K * (bz * inv - c * az * ira2);
        const gbx = (Fa * Fbp * g * bx) / rb + K * (ax * inv - c * bx * irb2);
        const gby = (Fa * Fbp * g * by) / rb + K * (ay * inv - c * by * irb2);
        const gbz = (Fa * Fbp * g * bz) / rb + K * (az * inv - c * bz * irb2);
        const ja = nJ[a], jb = nJ[b];
        f[3 * ja] -= gax; f[3 * ja + 1] -= gay; f[3 * ja + 2] -= gaz;
        f[3 * jb] -= gbx; f[3 * jb + 1] -= gby; f[3 * jb + 2] -= gbz;
        fxi += gax + gbx; fyi += gay + gby; fzi += gaz + gbz;
        if (eatom) {
          const e9 = e3 / 3;
          eatom[i] += e9; eatom[ja] += e9; eatom[jb] += e9;
        }
        if (vatom) {
          const w0 = -(ax * gax + bx * gbx) / 3, w1 = -(ay * gay + by * gby) / 3, w2 = -(az * gaz + bz * gbz) / 3;
          const w3 = -(ax * gay + bx * gby) / 3, w4 = -(ax * gaz + bx * gbz) / 3, w5 = -(ay * gaz + by * gbz) / 3;
          for (const q of [i, ja, jb]) {
            vatom[6 * q] += w0; vatom[6 * q + 1] += w1; vatom[6 * q + 2] += w2;
            vatom[6 * q + 3] += w3; vatom[6 * q + 4] += w4; vatom[6 * q + 5] += w5;
          }
        }
      }
      f[3 * i] += fxi;
      f[3 * i + 1] += fyi;
      f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }
}

/** pair_style meam/spline: "The meam/spline style computes pairwise interactions for metals". */
export class PairMeamSpline extends PairMeamSplineBase {
  readonly name = 'meam/spline';
  protected readonly sw = false;
}

/** pair_style meam/sw/spline: spline MEAM plus the Stillinger-Weber-like three-body term. */
export class PairMeamSWSpline extends PairMeamSplineBase {
  readonly name = 'meam/sw/spline';
  protected readonly sw = true;
}
