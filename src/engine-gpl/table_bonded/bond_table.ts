/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MOLECULE/bond_table.cpp and
 * src/MOLECULE/bond_table.h (BondTable: settings, coeff, read_table, param_extract, spline_table, compute_table,
 * uf_lookup, compute, single). Contributing author: Chuanfu Luo (luochuanfu@gmail.com). Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/bond_table.html: "bond_style table style N", where "style = linear or spline" and "N = use N
 * values to interpolate the table". The table file is read by bond_coeff "N file keyword"; its section has the
 * parameters "N value [FP fplo fphi] [EQ r0]" and then N lines "index r e f".
 */
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { atomIndex, bondDelta } from '../../engine/force/bonded_util';
import { spline, splint } from './splines';
import { TableFileReader, TokenError, WordStream, fmtG, readTableText, splitWords } from './tablefile';

const BIGNUM = 1.0e300;

/** A table section as read (Table in bond_table.h, the file arrays and the splined tables). */
interface BondTab {
  ninput: number;
  rfile: Float64Array;
  efile: Float64Array;
  ffile: Float64Array;
  e2file: Float64Array;
  f2file: Float64Array;
  fpflag: boolean;
  fplo: number;
  fphi: number;
  r0: number;
  lo: number;
  hi: number;
  delta: number;
  invdelta: number;
  deltasq6: number;
  r: Float64Array;
  e: Float64Array;
  de: Float64Array;
  f: Float64Array;
  df: Float64Array;
  e2: Float64Array;
  f2: Float64Array;
}

/** Native parameter line: "N value", optional "FP fplo fphi" and "EQ r0" (bond_table.cpp param_extract). */
const paramExtract = (line: string): { ninput: number; fpflag: boolean; fplo: number; fphi: number; r0: number } => {
  let ninput = 0, fpflag = false, fplo = 0, fphi = 0, r0 = 0.0;
  try {
    const w = new WordStream(splitWords(line));
    while (w.hasNext()) {
      const word = w.nextString();
      if (word === 'N') {
        ninput = w.nextInt();
      } else if (word === 'FP') {
        fpflag = true;
        fplo = w.nextDouble();
        fphi = w.nextDouble();
      } else if (word === 'EQ') {
        r0 = w.nextDouble();
      } else {
        throw new StyleError(`Unknown keyword ${word} in bond table parameters`);
      }
    }
  } catch (e) {
    if (e instanceof TokenError) throw new StyleError(e.message);
    throw e;
  }
  if (ninput === 0) throw new StyleError('Bond table parameters did not set N');
  return { ninput, fpflag, fplo, fphi, r0 };
};

export class BondTable extends Bonded {
  readonly name = 'table';
  readonly kind = 'bond' as const;

  /** 'linear' or 'spline' (tabstyle), set by bond_style table. */
  private tabstyle: 'linear' | 'spline' = 'linear';
  private tablength = 0;
  private tables: BondTab[] = [];
  private tabindex = new Int32Array(0);
  private r0 = new Float64Array(0);
  private setflag = new Uint8Array(0);

  /** bond_style table style N (BondTable::settings): drops every table read so far. */
  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 2) throw new StyleError('Illegal bond_style command: must have 2 arguments');
    if (args[0] === 'linear') this.tabstyle = 'linear';
    else if (args[0] === 'spline') this.tabstyle = 'spline';
    else throw new StyleError(`Unknown table style ${args[0]} in bond style table`);
    this.tablength = intArg(args[1]);
    if (this.tablength < 2) throw new StyleError(`Illegal number of bond table entries: ${args[1]}`);
    // delete old tables, since cannot just change settings (the next coeff allocates again)
    this.tables = [];
    this.allocate(this.ntypes);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.tabindex = new Int32Array(ntypes + 1);
    this.r0 = new Float64Array(ntypes + 1);
    this.setflag = new Uint8Array(ntypes + 1);
  }

  /** bond_coeff N-range file keyword (BondTable::coeff). */
  coeff(args: string[], ctx: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Illegal bond_coeff command: must have 3 arguments');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const text = readTableText(ctx, args[1], 'bond');
    const tb = this.readTable(text, args[1], args[2], ctx);
    // error check on table parameters
    if (tb.ninput <= 1) throw new StyleError(`Invalid bond table length: ${tb.ninput}`);
    tb.lo = tb.rfile[0];
    tb.hi = tb.rfile[tb.ninput - 1];
    if (tb.lo >= tb.hi) throw new StyleError('Bond table values are not increasing');
    // spline read-in and compute r,e,f vectors within table
    this.splineTable(tb);
    this.computeTable(tb);
    // store ptr to table in tabindex
    const index = this.tables.length;
    this.tables.push(tb);
    for (let i = ilo; i <= ihi; i++) {
      this.tabindex[i] = index;
      this.r0[i] = tb.r0;
      this.setflag[i] = 1;
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.setflag[t]) throw new StyleError(`All bond coeffs are not set. Status:\n(type ${t} has no table)`);
    }
  }

  /** Equilibrium bond length (BondTable::equilibrium_distance): the EQ value, or the file's minimum-energy r. */
  equilibrium(type: number): number { return this.r0[type]; }

  /**
   * read_table (bond_table.cpp): the section is found, its parameter line read, one line is skipped (raw, whatever it
   * holds), then N data lines are read. r0 is the file's r at the lowest energy when EQ is not given.
   */
  private readTable(text: string, file: string, keyword: string, ctx: StyleContext): BondTab {
    const reader = new TableFileReader(text, file, 'bond');
    const start = reader.findSectionStart(keyword);
    if (start === null) throw new StyleError(`Did not find keyword ${keyword} in table file`);
    // read args on 2nd line of section
    const pl = reader.nextLine();
    if (pl === null) throw new StyleError(`Did not find keyword ${keyword} in table file`);
    const p = paramExtract(pl);
    const n = p.ninput;
    const tb: BondTab = {
      ninput: n, rfile: new Float64Array(Math.max(n, 0)), efile: new Float64Array(Math.max(n, 0)), ffile: new Float64Array(Math.max(n, 0)),
      e2file: new Float64Array(0), f2file: new Float64Array(0), fpflag: p.fpflag, fplo: p.fplo, fphi: p.fphi, r0: p.r0,
      lo: 0, hi: 0, delta: 0, invdelta: 0, deltasq6: 0,
      r: new Float64Array(0), e: new Float64Array(0), de: new Float64Array(0), f: new Float64Array(0), df: new Float64Array(0),
      e2: new Float64Array(0), f2: new Float64Array(0),
    };
    // read r,e,f table values from file
    let r0idx = -1;
    reader.skipLine();
    let emin = BIGNUM;
    for (let i = 0; i < n; i++) {
      const line = reader.nextLine();
      if (line === null) throw new StyleError(`Data missing when parsing bond table '${keyword}' line ${i + 1} of ${n}.`);
      try {
        const w = new WordStream(splitWords(line));
        w.nextInt();
        tb.rfile[i] = w.nextDouble();
        tb.efile[i] = w.nextDouble();
        tb.ffile[i] = w.nextDouble();
      } catch (e) {
        if (e instanceof TokenError) throw new StyleError(`Error parsing bond table '${keyword}' line ${i + 1} of ${n}. ${e.message}\nLine was: ${line}`);
        throw e;
      }
      if (tb.efile[i] < emin) {
        emin = tb.efile[i];
        r0idx = i;
      }
    }
    // infer r0 from minimum of potential, if not given explicitly
    if (tb.r0 === 0.0 && r0idx >= 0) tb.r0 = tb.rfile[r0idx];
    // warn if force != dE/dr at any point that is not an inflection point (secant check; the end points are skipped)
    let ferror = 0;
    for (let i = 1; i < n - 1; i++) {
      const r = tb.rfile[i], rprev = tb.rfile[i - 1], rnext = tb.rfile[i + 1];
      const e = tb.efile[i], eprev = tb.efile[i - 1], enext = tb.efile[i + 1];
      const f = tb.ffile[i];
      const fleft = -(e - eprev) / (r - rprev);
      const fright = -(enext - e) / (rnext - r);
      if (f < fleft && f < fright) ferror++;
      if (f > fleft && f > fright) ferror++;
    }
    if (ferror) {
      ctx.log(`WARNING: ${ferror} of ${n} force values in table are inconsistent with -dE/dr.\nWARNING:  Should only be flagged at inflection points`);
    }
    return tb;
  }

  /** build spline representation of e,f over the read-in table (BondTable::spline_table). */
  private splineTable(tb: BondTab): void {
    const n = tb.ninput;
    tb.e2file = new Float64Array(n);
    tb.f2file = new Float64Array(n);
    const ep0 = -tb.ffile[0];
    const epn = -tb.ffile[n - 1];
    spline(tb.rfile, tb.efile, n, ep0, epn, tb.e2file);
    if (!tb.fpflag) {
      tb.fplo = (tb.ffile[1] - tb.ffile[0]) / (tb.rfile[1] - tb.rfile[0]);
      tb.fphi = (tb.ffile[n - 1] - tb.ffile[n - 2]) / (tb.rfile[n - 1] - tb.rfile[n - 2]);
    }
    const fp0 = tb.fplo;
    const fpn = tb.fphi;
    spline(tb.rfile, tb.ffile, n, fp0, fpn, tb.f2file);
  }

  /** compute r,e,f vectors (N evenly spaced bins from lo to hi) from the splined values (BondTable::compute_table). */
  private computeTable(tb: BondTab): void {
    const N = this.tablength;
    const tlm1 = N - 1;
    tb.delta = (tb.hi - tb.lo) / tlm1;
    tb.invdelta = 1.0 / tb.delta;
    tb.deltasq6 = tb.delta * tb.delta / 6.0;
    tb.r = new Float64Array(N);
    tb.e = new Float64Array(N);
    tb.de = new Float64Array(N);
    tb.f = new Float64Array(N);
    tb.df = new Float64Array(N);
    tb.e2 = new Float64Array(N);
    tb.f2 = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const a = tb.lo + i * tb.delta;
      tb.r[i] = a;
      tb.e[i] = splint(tb.rfile, tb.efile, tb.e2file, tb.ninput, a);
      tb.f[i] = splint(tb.rfile, tb.ffile, tb.f2file, tb.ninput, a);
    }
    for (let i = 0; i < tlm1; i++) {
      tb.de[i] = tb.e[i + 1] - tb.e[i];
      tb.df[i] = tb.f[i + 1] - tb.f[i];
    }
    // get final elements from linear extrapolation
    tb.de[tlm1] = 2.0 * tb.de[tlm1 - 1] - tb.de[tlm1 - 2];
    tb.df[tlm1] = 2.0 * tb.df[tlm1 - 1] - tb.df[tlm1 - 2];
    const ep0 = -tb.f[0];
    const epn = -tb.f[tlm1];
    spline(tb.r, tb.e, N, ep0, epn, tb.e2);
    spline(tb.r, tb.f, N, tb.fplo, tb.fphi, tb.f2);
  }

  /**
   * uf_lookup (BondTable::uf_lookup): the energy and -dE/dr at distance x, with the bin index from the truncated
   * (x - lo) / delta. Outside the table it is an error, named as the native message gives it.
   */
  private ufLookup(type: number, x: number): { u: number; f: number } {
    if (!Number.isFinite(x)) throw new StyleError('Illegal bond in bond style table');
    const tb = this.tables[this.tabindex[type]];
    const itable = Math.trunc((x - tb.lo) * tb.invdelta);
    if (itable < 0) throw new StyleError(`Bond length < table inner cutoff: type ${type} length ${fmtG(x, 8)}`);
    if (itable >= this.tablength) throw new StyleError(`Bond length > table outer cutoff: type ${type} length ${fmtG(x, 8)}`);
    if (this.tabstyle === 'linear') {
      const fraction = (x - tb.r[itable]) * tb.invdelta;
      return { u: tb.e[itable] + fraction * tb.de[itable], f: tb.f[itable] + fraction * tb.df[itable] };
    }
    // Spline: the native code reads entry itable + 1, one past the end at the last node (an undefined read in C++);
    // the value of the last node is used there (the weight of that entry is then b, below rounding at the last node).
    const i1 = itable + 1 < this.tablength ? itable + 1 : itable;
    const b = (x - tb.r[itable]) * tb.invdelta;
    const a = 1.0 - b;
    const u = a * tb.e[itable] + b * tb.e[i1] + ((a * a * a - a) * tb.e2[itable] + (b * b * b - b) * tb.e2[i1]) * tb.deltasq6;
    const f = a * tb.f[itable] + b * tb.f[i1] + ((a * a * a - a) * tb.f2[itable] + (b * b * b - b) * tb.f2[i1]) * tb.deltasq6;
    return { u, f };
  }

  /** BondTable::compute: fbond = mdu / r, forces on the two atoms, virial and energy tallies in list order. */
  compute(bc: BondedCompute): void {
    const B = bc.s.topo.bonds;
    const f = bc.f;
    const d = [0, 0, 0];
    for (let k = 0; k < B.n; k++) {
      const i1 = atomIndex(bc, B.atoms[2 * k], 'bond');
      const i2 = atomIndex(bc, B.atoms[2 * k + 1], 'bond');
      const type = B.type[k];
      bondDelta(bc, i1, i2, d); // delx = x[i1] - x[i2] (the copy of i2 nearest to i1)
      const delx = d[0], dely = d[1], delz = d[2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const r = Math.sqrt(rsq);
      const { u, f: mdu } = this.ufLookup(type, r);
      const fbond = mdu / r;
      const ebond = u;
      f[3 * i1] += delx * fbond;
      f[3 * i1 + 1] += dely * fbond;
      f[3 * i1 + 2] += delz * fbond;
      f[3 * i2] -= delx * fbond;
      f[3 * i2 + 1] -= dely * fbond;
      f[3 * i2 + 2] -= delz * fbond;
      bc.acc.ebond += ebond;
      const v = bc.virial;
      v[0] += delx * delx * fbond;
      v[1] += dely * dely * fbond;
      v[2] += delz * delz * fbond;
      v[3] += delx * dely * fbond;
      v[4] += delx * delz * fbond;
      v[5] += dely * delz * fbond;
      if (bc.eatom) {
        bc.eatom[i1] += 0.5 * ebond;
        bc.eatom[i2] += 0.5 * ebond;
      }
      if (bc.vatom) {
        const w = [delx * delx * fbond, dely * dely * fbond, delz * delz * fbond, delx * dely * fbond, delx * delz * fbond, dely * delz * fbond];
        for (let c = 0; c < 6; c++) {
          bc.vatom[6 * i1 + c] += 0.5 * w[c];
          bc.vatom[6 * i2 + c] += 0.5 * w[c];
        }
      }
    }
  }
}

/** utils::inumeric: an integer argument, or the native error. */
const intArg = (w: string): number => {
  if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
  return Number(w);
};

