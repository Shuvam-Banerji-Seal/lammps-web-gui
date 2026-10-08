import { FixWallBase } from './walls';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import type { System } from '../system';

/*
 * fix wall/table — docs.lammps.org/fix_wall.html (source: plans/lammps-docs/fix_wall.rst)
 *
 * Syntax (verbatim):
 *
 *   fix ID group-ID style [tabstyle] [N] face args ... keyword value ...
 *
 * * tabstyle = *linear* or *spline* = method of table interpolation (only applies to *wall/table*)
 * * N = use N values in *linear* or *spline* interpolation (only applies to *wall/table*)
 *
 * * args for style *table*
 *
 *   .. parsed-literal::
 *
 *          args = coord filename keyword cutoff
 *          coord = position of wall = EDGE or constant or variable
 *            EDGE = current lo or hi edge of simulation box
 *            constant = number like 0.0 or -30.0 (distance units)
 *            variable = :doc:`equal-style variable <variable>` like v_x or v_wiggle
 *          filename = file containing tabulated energy and force values
 *          keyword = section identifier to select a specific table in table file
 *          cutoff = distance from wall at which wall-particle interactions are cut off (distance units)
 *
 * "For style *wall/table*, the energy E and forces are determined from
 * interpolation tables listed in one or more files as a function of
 * distance.  The interpolation tables are used to evaluate energy and
 * forces between particles and the wall similar to how analytic formulas
 * are used for the other wall styles."
 *
 * "The interpolation tables are created as a pre-computation by fitting
 * cubic splines to the file values and interpolating energy and force
 * values at each of *N* distances.  During a simulation, the tables are
 * used to interpolate energy and force values as needed for each wall and
 * particle separated by a distance *R*\ .  The interpolation is done in
 * one of two styles: *linear* or *spline*."
 *
 * "For the *linear* style, the distance *R* is used to find the 2
 * surrounding table values from which an energy or force is computed by
 * linear interpolation."
 *
 * "For the *spline* style, cubic spline coefficients are computed and
 * stored for each of the *N* values in the table, one set of splines for
 * energy, another for force.  Note that these splines are different than
 * the ones used to pre-compute the *N* values.  Those splines were fit
 * to the *Nfile* values in the tabulated file, where often *Nfile* <
 * *N*\ .  The distance *R* is used to find the appropriate set of spline
 * coefficients which are used to evaluate a cubic polynomial which
 * computes the energy or force."
 *
 * "For each wall a filename and a keyword must be provided as in the
 * examples above.  The filename specifies a file containing tabulated
 * energy and force values.  The keyword specifies a section of the file.
 * The format of this file is described below."
 *
 * "In all cases, *r* is the distance from the particle to the wall at
 * position *coord*, and :math:`r_c` is the *cutoff* distance at which the
 * particle and wall no longer interact.  The energy of the wall
 * potential is shifted so that the wall-particle interaction energy is
 * 0.0 at the cutoff distance."
 *
 * Table file format (verbatim):
 *
 *   # Tabulated wall potential UNITS: real
 *
 *   HARMONIC                       (keyword is the first text on a line)
 *   N 100 FP 200 200
 *                                  (blank line)
 *   1  0.04    1568.16    792.00   (index, distance to wall, energy, force)
 *
 * "A section begins with a non-blank line whose first character is not a
 * "#"; blank lines or lines starting with "#" can be used as comments
 * between sections.  The first line begins with a keyword which identifies
 * the section.  The line can contain additional text, but the initial text
 * must match the argument specified in the fix *wall/table* command.  The
 * next line lists (in any order) one or more parameters for the table.
 * Each parameter is a keyword followed by one or more numeric values."
 *
 * "The parameter "N" is required and its value is the number of table
 * entries that follow.  Note that this may be different than the *N*
 * specified in the fix *wall/table* command.  Let Ntable = *N* in the fix
 * command, and Nfile = "N" in the tabulated file.  What LAMMPS does is a
 * preliminary interpolation by creating splines using the Nfile tabulated
 * values as nodal points.  It uses these to interpolate as needed to
 * generate energy and force values at Ntable different points.  The
 * resulting tables of length Ntable are then used as described above, when
 * computing energy and force for wall-particle interactions."
 *
 * Measured with native LAMMPS (black box) on synthetic tables:
 *
 * - *tabstyle* and *N* are both required (their absence is an error, not a
 *   default), *tabstyle* is one of the two documented words, and N must be
 *   >= 2 (N = 1 is an error). The parameter line accepts only N and the
 *   optional FP; R, RSQ, BITMAP and EQ are each rejected by name.
 * - The Nfile tabulated r values are the abscissae of the preliminary
 *   splines in r, which use all Nfile nodes ("the Nfile tabulated values as
 *   nodal points"; the cutoff does not drop nodes). The energy spline is a
 *   complete cubic with first derivatives -F at the innermost and outermost
 *   nodes (the force column is F = -dE/dr); the force spline is a complete
 *   cubic whose end first derivatives are FP fplo/fphi, or, when FP is
 *   absent, "estimated ... by the first two and last two force values in
 *   the table": (F1-F0)/(r1-r0) and (F_{n}-F_{n-1})/(r_n-r_{n-1}).
 * - The internal table has Ntable entries uniform in r from the innermost
 *   to the outermost tabulated r (not just to the cutoff; the cutoff only
 *   bounds the interaction and the energy shift). Each entry is the
 *   preliminary spline evaluated there.
 * - linear evaluation interpolates both columns linearly in r; spline
 *   evaluation is a complete cubic in r through the Ntable entries whose end
 *   first derivatives are the preliminary spline's derivative at the two
 *   grid ends (measured agreement 5e-13 energy, 5e-15 force).
 * - The energy is shifted by the table value at the cutoff, so it is 0 at r
 *   = r_c; atoms with r >= cutoff do not interact. FP changes only the force
 *   column, never the energy.
 * - Nfile = 2 is allowed and the preliminary splines reduce to the cubic
 *   Hermite polynomial fixed by the node values and the end derivatives.
 * - A cutoff above the outer tabulated r is an error; a cutoff below the
 *   inner tabulated distance is an error; a particle at r below the inner
 *   tabulated distance is an error (measured: the native lookup tolerates
 *   one bin below the inner distance, an implementation detail not
 *   reproduced here).
 */

const FACES = ['xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi'];
const isFace = (w: string): boolean => FACES.includes(w);

/** One parsed table section: Nfile nodes plus FP if given. */
interface FileSection {
  n: number;
  fplo: number | null;
  fphi: number | null;
  r: Float64Array;
  e: Float64Array;
  f: Float64Array;
}

/** One wall's parsed file/keyword/cutoff, in the order the base parsed faces. */
interface FaceSpec {
  filename: string;
  keyword: string;
  cutoff: number;
}

/** A built interpolation table for one wall. */
interface Table {
  /** Innermost tabulated distance. */
  rInner: number;
  /** Outermost tabulated distance (the internal grid's upper end). */
  rOuter: number;
  /** Interaction cutoff (<= rOuter); energy is shifted to 0 here. */
  cutoff: number;
  /** Ntable grid points, uniform in r from rInner to rOuter. */
  g: Float64Array;
  /** 1 / (grid spacing). */
  invDelta: number;
  /** Energy at the grid points. */
  e: Float64Array;
  /** Force F = -dE/dr at the grid points. */
  f: Float64Array;
  /** spline only: second derivatives of e and f on the grid. */
  e2?: Float64Array;
  f2?: Float64Array;
}

/**
 * Complete cubic spline through y at the strictly increasing nodes x with end
 * first derivatives d0, dn; returns the second derivatives M at the nodes.
 * For two nodes this is the cubic Hermite interpolant fixed by y, d0, dn.
 * Textbook tridiagonal (Thomas) solve.
 */
const buildSpline = (x: Float64Array, y: Float64Array, d0: number, dn: number): Float64Array => {
  const n = x.length;
  const M = new Float64Array(n);
  if (n === 2) {
    const h = x[1] - x[0];
    const s = (y[1] - y[0]) / h;
    const u = (3 * s - 2 * d0 - dn) / 3;
    const v = s - d0 - 2 * u;
    M[0] = (6 * u) / h;
    M[1] = (6 * v) / h;
    return M;
  }
  const h = new Float64Array(n - 1);
  for (let k = 0; k < n - 1; k++) h[k] = x[k + 1] - x[k];
  const lower = new Float64Array(n), diag = new Float64Array(n), upper = new Float64Array(n), rhs = new Float64Array(n);
  diag[0] = 2 * h[0]; upper[0] = h[0];
  rhs[0] = 6 * ((y[1] - y[0]) / h[0] - d0);
  for (let k = 1; k < n - 1; k++) {
    lower[k] = h[k - 1];
    diag[k] = 2 * (h[k - 1] + h[k]);
    upper[k] = h[k];
    rhs[k] = 6 * ((y[k + 1] - y[k]) / h[k] - (y[k] - y[k - 1]) / h[k - 1]);
  }
  lower[n - 1] = h[n - 2]; diag[n - 1] = 2 * h[n - 2];
  rhs[n - 1] = 6 * (dn - (y[n - 1] - y[n - 2]) / h[n - 2]);
  for (let k = 1; k < n; k++) {
    const w = lower[k] / diag[k - 1];
    diag[k] -= w * upper[k - 1];
    rhs[k] -= w * rhs[k - 1];
  }
  M[n - 1] = rhs[n - 1] / diag[n - 1];
  for (let k = n - 2; k >= 0; k--) M[k] = (rhs[k] - upper[k] * M[k + 1]) / diag[k];
  return M;
};

/** Interval index k with x[k] <= q < x[k+1], clamped to [0, n-2]. */
const interval = (x: Float64Array, q: number): number => {
  let lo = 0, hi = x.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (x[mid] > q) hi = mid; else lo = mid;
  }
  return lo;
};

/** Evaluates the cubic spline (nodes x, values y, second derivatives M) at q. */
const evalSpline = (x: Float64Array, y: Float64Array, M: Float64Array, q: number): number => {
  if (q <= x[0]) return y[0];
  const last = x.length - 1;
  if (q >= x[last]) return y[last];
  const lo = interval(x, q), h = x[lo + 1] - x[lo];
  const a = (x[lo + 1] - q) / h, b = (q - x[lo]) / h;
  return a * y[lo] + b * y[lo + 1] + ((a * a * a - a) * M[lo] + (b * b * b - b) * M[lo + 1]) * (h * h) / 6;
};

/** Derivative dS/dr of the cubic spline (x, y, M) at q. */
const evalSplineDeriv = (x: Float64Array, y: Float64Array, M: Float64Array, q: number): number => {
  const last = x.length - 1;
  const lo = interval(x, q <= x[0] ? x[0] + 1e-15 : q >= x[last] ? x[last] - 1e-15 : q);
  const h = x[lo + 1] - x[lo];
  const a = (x[lo + 1] - q) / h, b = (q - x[lo]) / h;
  return (y[lo + 1] - y[lo]) / h + ((3 * b * b - 1) * M[lo + 1] - (3 * a * a - 1) * M[lo]) * h / 6;
};

/** Parses the parameter line of one section (keyword + numeric values). */
const parseParams = (line: string, filename: string, keyword: string): { n: number; fplo: number | null; fphi: number | null } => {
  const t = line.split(/\s+/);
  let n = 0, fplo: number | null = null, fphi: number | null = null;
  let k = 0;
  while (k < t.length) {
    const key = t[k++];
    if (key === 'N') n = parseNum(t[k++], 'table parameter N');
    else if (key === 'FP') {
      fplo = parseNum(t[k++], 'table parameter FP fplo');
      fphi = parseNum(t[k++], 'table parameter FP fphi');
    } else throw new StyleError(`fix wall/table: unknown table parameter '${key}' in section ${keyword} of ${filename} (N and FP only)`);
  }
  if (!Number.isInteger(n) || n < 2) throw new StyleError(`fix wall/table: section ${keyword} of ${filename} needs an integer N >= 2 table entries, got N = ${n}`);
  return { n, fplo, fphi };
};

/** Reads the file and returns the section whose initial word matches keyword. */
const readSection = (text: string, filename: string, keyword: string): FileSection => {
  const lines = text.split('\n');
  let k = 0;
  while (k < lines.length) {
    const header = lines[k].trim();
    k++;
    // "A section begins with a non-blank line whose first character is not a "#""
    if (header === '' || header.startsWith('#')) continue;
    const kw = header.split(/\s+/)[0];
    let params = '';
    while (k < lines.length) {
      const l = lines[k].trim();
      k++;
      if (l !== '' && !l.startsWith('#')) { params = l; break; }
    }
    if (params === '') throw new StyleError(`fix wall/table: section ${kw} of ${filename} has no parameter line`);
    const { n, fplo, fphi } = parseParams(params, filename, kw);
    const r = new Float64Array(n), e = new Float64Array(n), f = new Float64Array(n);
    let got = 0;
    while (got < n && k < lines.length) {
      const l = lines[k].trim();
      k++;
      if (l === '' || l.startsWith('#')) continue;
      const t = l.split(/\s+/);
      if (t.length < 4) throw new StyleError(`fix wall/table: section ${kw} of ${filename} data line '${l}' needs index, distance, energy, force`);
      // "the first value is the index from 1 to N" (native requires an integer token)
      const idx = parseNum(t[0], 'table index');
      if (!Number.isInteger(idx)) throw new StyleError(`fix wall/table: section ${kw} of ${filename} data line '${l}' has a non-integer index '${t[0]}'`);
      r[got] = parseNum(t[1], 'table distance');
      e[got] = parseNum(t[2], 'table energy');
      f[got] = parseNum(t[3], 'table force');
      got++;
    }
    if (got < n) throw new StyleError(`fix wall/table: section ${kw} of ${filename} has ${got} of ${n} data lines`);
    for (let m = 1; m < n; m++) {
      if (!(r[m] > r[m - 1])) throw new StyleError(`fix wall/table: section ${kw} of ${filename} distances must increase from one line to the next`);
    }
    if (kw === keyword) return { n, fplo, fphi, r, e, f };
  }
  throw new StyleError(`fix wall/table: file ${filename} has no section with keyword '${keyword}'`);
};

/**
 * Parses the tokens after the style: tabstyle, N, then face/arg quadruples and
 * the units/fld/pbc keywords. Builds the numeric substitute arguments the base
 * wall parser accepts (filename and keyword are not numbers there).
 */
const parseTableArgs = (id: string, args: string[]): { tabStyle: 'linear' | 'spline'; ntable: number; specs: FaceSpec[]; fakeArgs: string[] } => {
  if (args.length < 2) throw new StyleError(`fix ${id} (wall/table): usage fix ID group-ID wall/table linear|spline N face args ... [keyword value ...]`);
  const tabStyle = args[0];
  if (tabStyle !== 'linear' && tabStyle !== 'spline') {
    throw new StyleError(`fix ${id} (wall/table): unknown table style '${tabStyle}' (linear or spline)`);
  }
  const ntable = parseNum(args[1], `fix ${id} (wall/table) table entries N`);
  if (!Number.isInteger(ntable) || ntable < 2) {
    throw new StyleError(`fix ${id} (wall/table): N must be an integer >= 2, got '${args[1]}'`);
  }
  const specs: FaceSpec[] = [];
  const fakeArgs: string[] = [];
  for (let k = 2; k < args.length;) {
    const w = args[k];
    if (isFace(w)) {
      const words = args.slice(k + 1, k + 5);
      if (words.length < 4) throw new StyleError(`fix ${id} (wall/table): face ${w} needs coord filename keyword cutoff`);
      const coord = words[0], filename = words[1], keyword = words[2], cutW = words[3];
      const cutoff = parseNum(cutW, `fix ${id} (wall/table) cutoff`);
      if (!(cutoff > 0)) throw new StyleError(`fix ${id} (wall/table): cutoff must be > 0, got '${cutW}'`);
      specs.push({ filename, keyword, cutoff });
      // the base wall parser reads coord, two numeric params, cutoff
      fakeArgs.push(w, coord, '1', '1', cutW);
      k += 5;
    } else if (w === 'units' || w === 'fld' || w === 'pbc') {
      if (args[k + 1] === undefined) throw new StyleError(`fix ${id} (wall/table): keyword ${w} needs a value`);
      fakeArgs.push(w, args[k + 1]);
      k += 2;
    } else {
      throw new StyleError(`fix ${id} (wall/table): unknown argument '${w}' (expected a face xlo..zhi or the keyword units, fld or pbc)`);
    }
  }
  if (!specs.length) throw new StyleError(`fix ${id} (wall/table): no wall face (xlo, xhi, ylo, yhi, zlo or zhi) was specified`);
  return { tabStyle, ntable, specs, fakeArgs };
};

/**
 * fix wall/table — tabulated wall-particle energy and force with linear or
 * cubic-spline interpolation (docs.lammps.org/fix_wall.html).
 */
export class FixWallTable extends FixWallBase {
  readonly style = 'wall/table';
  private readonly tabStyle: 'linear' | 'spline';
  private readonly ntable: number;
  private readonly specs: FaceSpec[];
  private tables: Table[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    const parsed = parseTableArgs(id, args);
    super(sys, id, group, parsed.fakeArgs, 'wall/table', 4);
    this.tabStyle = parsed.tabStyle;
    this.ntable = parsed.ntable;
    this.specs = parsed.specs;
  }

  init(): void {
    super.init();
    this.tables = this.specs.map((s) => this.buildTable(s));
  }

  /** Preliminary splines, Ntable grid, optional internal splines for one wall. */
  private buildTable(s: FaceSpec): Table {
    const sec = readSection(this.sys.readFile(s.filename), s.filename, s.keyword);
    const { n, r, e, f } = sec;
    const rInner = r[0], rOuter = r[n - 1];
    const cutoff = s.cutoff;
    if (cutoff < rInner) throw new StyleError(`fix ${this.id} (wall/table): cutoff ${cutoff} is below the table inner distance ${rInner}`);
    if (cutoff > rOuter) throw new StyleError(`fix ${this.id} (wall/table): cutoff ${cutoff} exceeds the table outer distance ${rOuter}`);
    // preliminary splines through the Nfile nodes (all of them, not just up to the cutoff)
    const me = buildSpline(r, e, -f[0], -f[n - 1]);
    const d0f = sec.fplo !== null ? sec.fplo : (f[1] - f[0]) / (r[1] - r[0]);
    const dnf = sec.fphi !== null ? sec.fphi : (f[n - 1] - f[n - 2]) / (r[n - 1] - r[n - 2]);
    const mf = buildSpline(r, f, d0f, dnf);
    // internal grid: Ntable points uniform in r from the inner to the outer node
    const g = new Float64Array(this.ntable);
    for (let k = 0; k < this.ntable; k++) g[k] = rInner + ((rOuter - rInner) * k) / (this.ntable - 1);
    const eT = new Float64Array(this.ntable), fT = new Float64Array(this.ntable);
    for (let k = 0; k < this.ntable; k++) {
      eT[k] = evalSpline(r, e, me, g[k]);
      fT[k] = evalSpline(r, f, mf, g[k]);
    }
    const tab: Table = { rInner, rOuter, cutoff, g, invDelta: (this.ntable - 1) / (rOuter - rInner), e: eT, f: fT };
    if (this.tabStyle === 'spline') {
      // internal cubic splines in r through the Ntable entries, end first
      // derivatives from the preliminary splines at the two grid ends
      const last = this.ntable - 1;
      tab.e2 = buildSpline(g, eT, evalSplineDeriv(r, e, me, g[0]), evalSplineDeriv(r, e, me, g[last]));
      tab.f2 = buildSpline(g, fT, evalSplineDeriv(r, f, mf, g[0]), evalSplineDeriv(r, f, mf, g[last]));
    }
    return tab;
  }

  /** Wall index of w (the base passes one of its own Wall objects). */
  private indexOfWall(w: unknown): number {
    const k = this.walls.indexOf(w as never);
    if (k < 0) throw new StyleError(`fix ${this.id} (wall/table): internal wall lookup failed`);
    return k;
  }

  protected energyAt(r: number, w: unknown): number {
    const t = this.tables[this.indexOfWall(w)];
    if (r < t.rInner) throw new StyleError(`fix ${this.id} (wall/table): particle/wall distance ${r} is below the table inner cutoff ${t.rInner}`);
    return this.interp(t, r, true);
  }

  protected forceAt(r: number, w: unknown): number {
    const t = this.tables[this.indexOfWall(w)];
    if (r < t.rInner) throw new StyleError(`fix ${this.id} (wall/table): particle/wall distance ${r} is below the table inner cutoff ${t.rInner}`);
    if (r >= t.cutoff) return 0;
    return this.interp(t, r, false);
  }

  /** Interpolates the energy (wantEnergy) or force column of table t at r. */
  private interp(t: Table, r: number, wantEnergy: boolean): number {
    const y = wantEnergy ? t.e : t.f;
    const m = wantEnergy ? t.e2 : t.f2;
    let p = (r - t.rInner) * t.invDelta;
    const last = t.g.length - 1;
    if (p < 0) p = 0;
    else if (p > last) p = last;
    if (this.tabStyle === 'spline') {
      // cubic in r: evaluate from the internal second derivatives at p
      let lo = Math.floor(p);
      if (lo >= last) lo = last - 1;
      const h = t.g[lo + 1] - t.g[lo];
      const a = (t.g[lo + 1] - r) / h, b = (r - t.g[lo]) / h;
      return a * y[lo] + b * y[lo + 1] + ((a * a * a - a) * m![lo] + (b * b * b - b) * m![lo + 1]) * (h * h) / 6;
    }
    // "the distance R is used to find the 2 surrounding table values ...
    // computed by linear interpolation"
    let lo = Math.floor(p), frac = p - lo;
    if (lo >= last) { lo = last - 1; frac = 1; }
    return y[lo] + frac * (y[lo + 1] - y[lo]);
  }
}
