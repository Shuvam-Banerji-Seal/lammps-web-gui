import { Pair, PairParams, StyleError, typeBounds, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { parseInt_, parseNum } from '../util';

/*
 * pair_style table — docs.lammps.org/pair_table.html
 * (source: plans/lammps-docs/pair_table.rst)
 *
 * Syntax (verbatim):
 *    pair_style table style N keyword ...
 *
 * * style = *lookup* or *linear* or *spline* or *bitmap* = method of interpolation
 * * N = use N values in *lookup*, *linear*, *spline* tables
 * * N = use 2\^N values in *bitmap* tables
 * * keyword = *ewald* or *pppm* or *msm* or *dispersion* or *tip4p*
 *
 * Interpolation (verbatim):
 * "For the *lookup* style, the distance *R* is used to find the nearest
 * table entry, which is the energy or force."
 * "For the *linear* style, the distance *R* is used to find the 2
 * surrounding table values from which an energy or force is computed by
 * linear interpolation."
 *
 * Coefficients (verbatim):
 * "The following coefficients must be defined for each pair of atoms
 * types via the :doc:`pair_coeff <pair_coeff>` command as in the examples
 * above."
 * * filename
 * * keyword
 * * cutoff (distance units)
 * "The keyword specifies a section of the file.  The cutoff is
 * an optional coefficient.  If not specified, the outer cutoff in the
 * table itself (see below) will be used to build an interpolation table
 * that extend to the largest tabulated distance.  If specified, only
 * file values up to the cutoff are used to create the interpolation
 * table.  The format of this file is described below."
 *
 * File format (verbatim):
 * "A section begins with a non-blank line whose first character is not a
 * "#"; blank lines or lines starting with "#" can be used as comments
 * between sections.  The first line begins with a keyword which
 * identifies the section.  The line can contain additional text, but the
 * initial text must match the argument specified in the pair_coeff
 * command.  The next line lists (in any order) one or more parameters
 * for the table.  Each parameter is a keyword followed by one or more
 * numeric values."
 * "The parameter "N" is required and its value is the number of table
 * entries that follow."
 * "1 1.0 25.5 102.34          (index, r, energy, force)"
 * "Following a blank line, the next N lines list the tabulated values.
 * On each line, the first value is the index from 1 to N, the second value is
 * r (in distance units), the third value is the energy (in energy units),
 * and the fourth is the force (in force units).  The r values must increase
 * from one line to the next (unless the BITMAP parameter is specified)."
 * "If used, the parameters "R" or "RSQ" are followed by 2 values *rlo*
 * and *rhi*\ .  If specified, the distance associated with each energy and
 * force value is computed from these 2 values (at high accuracy), rather
 * than using the (low-accuracy) value listed in each line of the table.
 * The distance values in the table file are ignored in this case.
 * For "R", distances uniformly spaced between *rlo* and *rhi* are
 * computed; for "RSQ", squared distances uniformly spaced between
 * *rlo\*rlo* and *rhi\*rhi* are computed."
 * "If used, the parameter "FPRIME" is followed by 2 values *fplo* and
 * *fphi* which are the derivative of the force at the innermost and
 * outermost distances listed in the table.  These values are needed by
 * the spline construction routines.  If not specified by the "FPRIME"
 * parameter, they are estimated (less accurately) by the first 2 and
 * last 2 force values in the table.  This parameter is not used by
 * BITMAP tables."
 *
 * Ntable vs Nfile (verbatim):
 * "Let
 * Ntable = *N* in the pair_style command, and Nfile = "N" in the
 * tabulated file.  What LAMMPS does is a preliminary interpolation by
 * creating splines using the Nfile tabulated values as nodal points.  It
 * uses these to interpolate energy and force values at Ntable different
 * points.  The resulting tables of length Ntable are then used as
 * described above, when computing energy and force for individual pair
 * distances.  This means that if you want the interpolation tables of
 * length Ntable to match exactly what is in the tabulated file (with
 * effectively no preliminary interpolation), you should set Ntable =
 * Nfile, and use the "RSQ" or "BITMAP" parameter.  This is because the
 * internal table abscissa is always RSQ (separation distance squared),
 * for efficient lookup."
 *
 * The doc does not spell out the sample points, the end conditions of
 * the preliminary splines, or the exact quantity interpolated. Measured
 * against native LAMMPS (/home/roy/.local/bin/lmp) with dimers and
 * pair_write on tests/oracle/w3table_lj.table:
 *
 * - The preliminary splines are complete cubic splines in r through ALL
 *   Nfile tabulated values ("using the Nfile tabulated values as nodal
 *   points" — the cutoff does not drop nodes, it only sets the extent of
 *   the internal table). The energy spline's end first derivatives are
 *   -F at the first and last tabulated distances (dE/dr = -F); the force
 *   spline's are the FPRIME values, or the slopes "estimated ... by the
 *   first 2 and last 2 force values in the table".
 * - The internal table holds Ntable points uniform in r^2 from the first
 *   tabulated distance to the pair_coeff cutoff (linear style: points
 *   k/(Ntable-1); lookup style: the bin midpoints (k+1/2)/(Ntable-1),
 *   because a lookup entry represents the whole bin between two node
 *   distances). For linear with Ntable = Nfile, RSQ spacing and a cutoff
 *   reaching the last tabulated distance, the sample points coincide
 *   with the tabulated values and the file values are used directly
 *   ("effectively no preliminary interpolation").
 * - The energy column stores E; the force column stores the pair force
 *   scalar F/r ("fpair"). Linear evaluation interpolates both columns
 *   linearly in r^2 and the radial force is fpair*r; lookup evaluation
 *   takes the single nearest entry (the bin containing r^2) and the
 *   radial force is again fpair*r.
 */

/** One parsed section of a table file (docs.lammps.org/pair_table.html). */
interface FileSection {
  /** The required "N" parameter: number of table entries that follow. */
  n: number;
  /** Spacing parameter: none (file r used as-is), R, RSQ or BITMAP. */
  spacing: 'none' | 'R' | 'RSQ';
  rlo: number;
  rhi: number;
  /** FPRIME values, or null to estimate from the first/last 2 forces. */
  fplo: number | null;
  fphi: number | null;
  /** The N (index, r, energy, force) lines; r is replaced for R/RSQ. */
  r: Float64Array;
  e: Float64Array;
  f: Float64Array;
}

/** The per type pair interpolation table (uniform in rsq). */
interface Tab {
  rsq1: number;
  /** 1 / (rsq spacing between entries). */
  invDelta: number;
  e: Float64Array;
  /** F/r at the table points (the pair force scalar). */
  g: Float64Array;
}

/** Parses the parameter line of one section (keyword + numeric values). */
const parseParams = (line: string, filename: string, keyword: string): FileSection => {
  const t = line.split(/\s+/);
  const sec: FileSection = { n: 0, spacing: 'none', rlo: 0, rhi: 0, fplo: null, fphi: null, r: new Float64Array(0), e: new Float64Array(0), f: new Float64Array(0) };
  let k = 0;
  while (k < t.length) {
    const key = t[k++];
    if (key === 'N') sec.n = parseInt_(t[k++], 'table parameter N');
    else if (key === 'R' || key === 'RSQ') {
      sec.spacing = key;
      sec.rlo = parseNum(t[k++], `table parameter ${key} rlo`);
      sec.rhi = parseNum(t[k++], `table parameter ${key} rhi`);
    } else if (key === 'BITMAP') {
      throw new StyleError(`pair_style table: BITMAP table format is not supported (file ${filename}, section ${keyword})`);
    } else if (key === 'FPRIME') {
      sec.fplo = parseNum(t[k++], 'table parameter FPRIME fplo');
      sec.fphi = parseNum(t[k++], 'table parameter FPRIME fphi');
    } else throw new StyleError(`table file ${filename}: unknown table parameter '${key}' in section ${keyword}`);
  }
  if (sec.n < 2) throw new StyleError(`table file ${filename}: section ${keyword} needs N >= 2 table entries`);
  if (sec.spacing !== 'none' && !(sec.rhi > sec.rlo)) {
    throw new StyleError(`table file ${filename}: section ${keyword} ${sec.spacing} needs rhi > rlo`);
  }
  return sec;
};

/** Reads the file and returns the section whose keyword matches pair_coeff. */
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
    if (params === '') throw new StyleError(`table file ${filename}: section ${kw} has no parameter line`);
    const sec = parseParams(params, filename, kw);
    const r = new Float64Array(sec.n), e = new Float64Array(sec.n), f = new Float64Array(sec.n);
    let got = 0;
    while (got < sec.n && k < lines.length) {
      const l = lines[k].trim();
      k++;
      if (l === '' || l.startsWith('#')) continue;
      const t = l.split(/\s+/);
      if (t.length < 4) throw new StyleError(`table file ${filename}: section ${kw} data line '${l}' needs index, r, energy, force`);
      r[got] = parseNum(t[1], 'table r');
      e[got] = parseNum(t[2], 'table energy');
      f[got] = parseNum(t[3], 'table force');
      got++;
    }
    if (got < sec.n) throw new StyleError(`table file ${filename}: section ${kw} has ${got} of ${sec.n} data lines`);
    if (kw === keyword) {
      if (sec.spacing === 'none') {
        // "The r values must increase from one line to the next"
        for (let m = 1; m < sec.n; m++) {
          if (!(r[m] > r[m - 1])) throw new StyleError(`table file ${filename}: section ${kw} r values must increase from one line to the next`);
        }
      }
      sec.r = r; sec.e = e; sec.f = f;
      return sec;
    }
  }
  throw new StyleError(`table file ${filename} has no section with keyword '${keyword}'`);
};

/**
 * Complete cubic spline (fixed first derivatives d0, dn at both ends)
 * through y at the strictly increasing nodes x; returns the second
 * derivatives M for evalSpline(). Textbook tridiagonal (Thomas) solve of
 *   2*h[0]*M0 + h[0]*M1 = 6*((y1-y0)/h0 - d0)
 *   h[k-1]*M[k-1] + 2*(h[k-1]+h[k])*M[k] + h[k]*M[k+1] = 6*d2y[k]
 *   h[n-2]*M[n-2] + 2*h[n-2]*M[n-1] = 6*(dn - (y[n-1]-y[n-2])/h[n-2])
 */
const completeSpline = (x: Float64Array, y: Float64Array, d0: number, dn: number): Float64Array => {
  const n = x.length;
  const M = new Float64Array(n);
  if (n < 3) return M;
  const h = new Float64Array(n - 1);
  for (let k = 0; k < n - 1; k++) h[k] = x[k + 1] - x[k];
  // tridiagonal system: lower[i]*M[i-1] + diag[i]*M[i] + upper[i]*M[i+1] = rhs[i]
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

/** Evaluates the spline (completeSpline coefficients M) at xq. */
const evalSpline = (x: Float64Array, y: Float64Array, M: Float64Array, xq: number): number => {
  // binary search for the interval x[k] <= xq <= x[k+1]
  let lo = 0, hi = x.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (x[mid] > xq) hi = mid; else lo = mid;
  }
  const h = x[hi] - x[lo];
  const a = (x[hi] - xq) / h, b = (xq - x[lo]) / h;
  return a * y[lo] + b * y[hi] + ((a * a * a - a) * M[lo] + (b * b * b - b) * M[hi]) * h * h / 6;
};

/**
 * pair_style table — tabulated pair potentials with lookup or linear
 * interpolation (spline and bitmap are rejected by name).
 */
export class PairTable extends Pair {
  readonly name: string = 'table';
  virialFdotr = true;
  /** Interpolation style from pair_style: lookup or linear. */
  mode: 'lookup' | 'linear' = 'linear';
  /** N from pair_style: the length of the internal interpolation table. */
  ntable = 0;
  p!: PairParams;
  /** Per type pair file section + cutoff, stored by pair_coeff (i*nt+j, both orders). */
  private inputs = new Map<number, { sec: FileSection; cut: number }>();
  private tRsq1 = new Float64Array(0);
  private tInvDelta = new Float64Array(0);
  private tE: Float64Array[] = [];
  private tG: Float64Array[] = [];

  settings(args: string[]): void {
    if (args.length < 2) throw new StyleError('usage: pair_style table style N [keyword ...]');
    const style = args[0];
    if (style === 'spline' || style === 'bitmap') {
      throw new StyleError(`pair_style table interpolation style '${style}' is not supported (lookup and linear only)`);
    }
    if (style !== 'lookup' && style !== 'linear') {
      throw new StyleError(`invalid pair_style table interpolation style '${style}' (lookup, linear, spline or bitmap)`);
    }
    this.mode = style;
    const n = parseInt_(args[1], 'N');
    if (n < 2) throw new StyleError('pair_style table N must be >= 2');
    this.ntable = n;
    // "keyword = *ewald* or *pppm* or *msm* or *dispersion* or *tip4p*" —
    // long-range solver coupling is not implemented; reject by name.
    if (args.length > 2) throw new StyleError(`pair_style table keyword '${args[2]}' is not supported (long-range solver coupling is not implemented)`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['cut']);
    this.inputs.clear();
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (args.length < 4 || args.length > 5) throw new StyleError('usage: pair_coeff I J filename keyword [cutoff]');
    const filename = args[2], keyword = args[3];
    const sec = readSection(ctx.readFile(filename), filename, keyword);
    // "The cutoff is an optional coefficient.  If not specified, the outer cutoff in the
    // table itself ... will be used"
    const cut = args[4] !== undefined ? parseNum(args[4], 'cutoff') : (sec.spacing === 'none' ? sec.r[sec.n - 1] : sec.rhi);
    if (!(cut > 0)) throw new StyleError('pair_coeff cutoff must be > 0');
    // measured with native LAMMPS: a cutoff at or below the table's inner distance, or beyond its
    // outer one, is an error (the outer distance itself is accepted)
    const inner = sec.spacing === 'none' ? sec.r[0] : sec.rlo;
    const outer = sec.spacing === 'none' ? sec.r[sec.n - 1] : sec.rhi;
    if (cut <= inner || cut > outer) throw new StyleError('Pair table cutoff outside of table');
    // "This pair style does not support mixing.  Thus, coefficients for all
    // I,J pairs must be specified explicitly." — setRange marks them set.
    this.p.setRange(args[0], args[1], [cut]);
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const nt = this.ntypes + 1;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.inputs.set(i * nt + j, { sec, cut });
        count++;
      }
    }
    if (count === 0 && ilo === ihi && jlo === jhi) this.inputs.set(jlo * nt + ilo, { sec, cut });
    if (count === 0) throw new StyleError(`pair coefficients: no type pairs with I <= J in ${args[0]} ${args[1]}`);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.ntypes + 1;
    if (this.tRsq1.length !== nt * nt) {
      this.tRsq1 = new Float64Array(nt * nt);
      this.tInvDelta = new Float64Array(nt * nt);
      this.tE = new Array(nt * nt);
      this.tG = new Array(nt * nt);
    }
    const k1 = i * nt + j, k2 = j * nt + i;
    const input = this.inputs.get(k1)!;
    const tab = this.buildTab(input);
    this.tRsq1[k1] = this.tRsq1[k2] = tab.rsq1;
    this.tInvDelta[k1] = this.tInvDelta[k2] = tab.invDelta;
    this.tE[k1] = this.tE[k2] = tab.e;
    this.tG[k1] = this.tG[k2] = tab.g;
    return input.cut;
  }

  /**
   * Builds the internal interpolation table for one type pair: complete
   * cubic splines through all Nfile tabulated values, sampled at Ntable
   * distances uniform in rsq from the first tabulated distance to the
   * cutoff (lookup: the bin midpoints) — or the file values directly for
   * linear when the sample grid coincides with the RSQ node grid.
   */
  private buildTab(input: { sec: FileSection; cut: number }): Tab {
    const sec = input.sec, cut = input.cut, n = sec.n;
    // distances: "For "R", distances uniformly spaced between *rlo* and *rhi* are
    // computed; for "RSQ", squared distances uniformly spaced between
    // *rlo\*rlo* and *rhi\*rhi* are computed."
    let r: Float64Array;
    if (sec.spacing === 'R') {
      r = new Float64Array(n);
      for (let k = 0; k < n; k++) r[k] = sec.rlo + ((sec.rhi - sec.rlo) * k) / (n - 1);
    } else if (sec.spacing === 'RSQ') {
      r = new Float64Array(n);
      const lo = sec.rlo * sec.rlo, hi = sec.rhi * sec.rhi;
      for (let k = 0; k < n; k++) r[k] = Math.sqrt(lo + ((hi - lo) * k) / (n - 1));
    } else r = sec.r;
    const rLast = r[n - 1];
    if (cut > rLast * (1 + 1e-12) + 1e-12) {
      throw new StyleError(`pair_coeff cutoff ${cut} exceeds the outer table cutoff ${rLast}`);
    }
    if (cut < r[0]) throw new StyleError(`pair_coeff cutoff ${cut} is below the inner table distance ${r[0]}`);
    const rsq1 = r[0] * r[0];
    // direct copy for linear: "you should set Ntable = Nfile, and use the "RSQ"
    // or "BITMAP" parameter" — the Ntable sample points coincide with the
    // tabulated values ("effectively no preliminary interpolation").
    if (this.mode === 'linear' && sec.spacing === 'RSQ' && this.ntable === n && cut >= rLast - 1e-12) {
      const rsqN = sec.rhi * sec.rhi;
      const g = new Float64Array(n);
      for (let k = 0; k < n; k++) g[k] = sec.f[k] / r[k];
      return { rsq1, invDelta: (n - 1) / (rsqN - rsq1), e: sec.e, g };
    }
    // preliminary interpolation: splines through the Nfile tabulated values
    // as nodal points, sampled at Ntable points; the internal table abscissa
    // is RSQ. End derivatives measured against native LAMMPS (see header):
    // the energy spline gets -F at both ends, the force spline the slopes
    // estimated by the first 2 and last 2 force values (FPRIME if given).
    const d0e = -sec.f[0], dne = -sec.f[n - 1];
    const d0f = sec.fplo !== null ? sec.fplo : (sec.f[1] - sec.f[0]) / (r[1] - r[0]);
    const dnf = sec.fphi !== null ? sec.fphi : (sec.f[n - 1] - sec.f[n - 2]) / (r[n - 1] - r[n - 2]);
    const me = completeSpline(r, sec.e, d0e, dne);
    const mf = completeSpline(r, sec.f, d0f, dnf);
    const rsqN = cut * cut;
    const eT = new Float64Array(this.ntable), gT = new Float64Array(this.ntable);
    const half = this.mode === 'lookup' ? 0.5 : 0;
    for (let k = 0; k < this.ntable; k++) {
      const rsq = rsq1 + ((rsqN - rsq1) * (k + half)) / (this.ntable - 1);
      const rq = Math.sqrt(rsq);
      eT[k] = evalSpline(r, sec.e, me, rq);
      gT[k] = evalSpline(r, sec.f, mf, rq) / rq;
    }
    return { rsq1, invDelta: (this.ntable - 1) / (rsqN - rsq1), e: eT, g: gT };
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq } = this;
    const sLJ = pc.specialLJ;
    const lookup = this.mode === 'lookup';
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    const { tRsq1, tInvDelta, tE, tG } = this;
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
        if (rsq >= cutsq[t]) continue;
        const eT = tE[t]!, gT = tG[t]!;
        const rsq1 = tRsq1[t];
        if (rsq < rsq1) throw new Error(`pair distance ${Math.sqrt(rsq)} below the table inner cutoff ${Math.sqrt(rsq1)}`);
        const p = (rsq - rsq1) * tInvDelta[t];
        let e: number, fpair: number;
        if (lookup) {
          // "the distance R is used to find the nearest table entry"
          let m = Math.floor(p);
          const last = eT.length - 1;
          if (m > last) m = last; else if (m < 0) m = 0;
          e = eT[m]; fpair = gT[m];
        } else {
          // "find the 2 surrounding table values from which an energy or force
          // is computed by linear interpolation" (uniform rsq abscissa)
          let m = Math.floor(p), frac = p - m;
          const last = eT.length - 1;
          if (m >= last) { m = last - 1; frac = 1; } else if (m < 0) { m = 0; frac = 0; }
          e = eT[m] + frac * (eT[m + 1] - eT[m]);
          fpair = gT[m] + frac * (gT[m + 1] - gT[m]);
        }
        const factor = sLJ[jj >>> SBBITS];
        // fpair is F/r: the scalar multiplying (dx,dy,dz); the radial force is fpair*r
        const fs = factor * fpair;
        const fx = dx * fs, fy = dy * fs, fz = dz * fs;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const ev = factor * e;
        evdwl += ev;
        if (tally) tallyAtom(pc, i, j, ev, fs, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number): { eng: number; fforce: number } {
    const t = itype * (this.ntypes + 1) + jtype;
    const eT = this.tE[t]!, gT = this.tG[t]!;
    const rsq1 = this.tRsq1[t];
    if (rsq < rsq1) throw new Error(`pair distance ${Math.sqrt(rsq)} below the table inner cutoff ${Math.sqrt(rsq1)}`);
    const p = (rsq - rsq1) * this.tInvDelta[t];
    let e: number, fpair: number;
    if (this.mode === 'lookup') {
      let m = Math.floor(p);
      const last = eT.length - 1;
      if (m > last) m = last; else if (m < 0) m = 0;
      e = eT[m]; fpair = gT[m];
    } else {
      let m = Math.floor(p), frac = p - m;
      const last = eT.length - 1;
      if (m >= last) { m = last - 1; frac = 1; } else if (m < 0) { m = 0; frac = 0; }
      e = eT[m] + frac * (eT[m + 1] - eT[m]);
      fpair = gT[m] + frac * (gT[m + 1] - gT[m]);
    }
    return { eng: factorLJ * e, fforce: factorLJ * fpair };
  }
}
