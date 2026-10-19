/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MOLECULE/angle_table.cpp and
 * src/MOLECULE/angle_table.h (AngleTable: settings, coeff, read_table, param_extract, spline_table, compute_table,
 * uf_lookup, u_lookup, compute). Contributing author: Chuanfu Luo (luochuanfu@gmail.com). Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/angle_table.html: "angle_style table style N", where "style = linear or spline" and "N = use N
 * values to interpolate the table". The angle_coeff line names a section "N value [FP fplo fphi] [EQ theta0]" and N
 * lines "index angle energy force", the angles in degrees, from 0 to 180. The angles are converted to radians here,
 * as the native code does.
 */
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { spline, splint } from './splines';
import { TableFileReader, TokenError, WordStream, readTableText, splitWords } from './tablefile';
import { acos as libmAcos } from '../libm/acos';
import { tallyAngle } from '../../engine/force/angle/harmonic';

const MY_PI = 3.14159265358979323846;
const DEG2RAD = MY_PI / 180.0;
const RAD2DEG = 180.0 / MY_PI;
const SMALL = 0.001;
const TINY = 1.0e-10;

/** A table section (Table in angle_table.h). */
interface AngleTab {
  ninput: number;
  afile: Float64Array;
  efile: Float64Array;
  ffile: Float64Array;
  e2file: Float64Array;
  f2file: Float64Array;
  fpflag: boolean;
  fplo: number;
  fphi: number;
  theta0: number;
  delta: number;
  invdelta: number;
  deltasq6: number;
  ang: Float64Array;
  e: Float64Array;
  de: Float64Array;
  f: Float64Array;
  df: Float64Array;
  e2: Float64Array;
  f2: Float64Array;
}

/** Native parameter line (angle_table.cpp param_extract): theta0 defaults to pi, FP values are scaled to degrees. */
const paramExtract = (line: string): { ninput: number; fpflag: boolean; fplo: number; fphi: number; theta0: number } => {
  let ninput = 0, fpflag = false, fplo = 0, fphi = 0, theta0 = MY_PI;
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
        fplo *= RAD2DEG * RAD2DEG;
        fphi *= RAD2DEG * RAD2DEG;
      } else if (word === 'EQ') {
        theta0 = DEG2RAD * w.nextDouble();
      } else {
        throw new StyleError(`Unknown keyword ${word} in angle table parameters`);
      }
    }
  } catch (e) {
    if (e instanceof TokenError) throw new StyleError(e.message);
    throw e;
  }
  if (ninput === 0) throw new StyleError('Angle table parameters did not set N');
  return { ninput, fpflag, fplo, fphi, theta0 };
};

export class AngleTable extends Bonded {
  readonly name = 'table';
  readonly kind = 'angle' as const;

  private tabstyle: 'linear' | 'spline' = 'linear';
  private tablength = 0;
  private tables: AngleTab[] = [];
  private tabindex = new Int32Array(0);
  private theta0 = new Float64Array(0);
  private setflag = new Uint8Array(0);

  /** angle_style table style N (AngleTable::settings): drops every table read so far. */
  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 2) throw new StyleError('Illegal angle_style command: must have 2 arguments');
    if (args[0] === 'linear') this.tabstyle = 'linear';
    else if (args[0] === 'spline') this.tabstyle = 'spline';
    else throw new StyleError(`Unknown table style ${args[0]} in angle style table`);
    this.tablength = intArg(args[1]);
    if (this.tablength < 2) throw new StyleError(`Illegal number of angle table entries: ${args[1]}`);
    // delete old tables, since cannot just change settings (the next coeff allocates again)
    this.tables = [];
    this.allocate(this.ntypes);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.tabindex = new Int32Array(ntypes + 1);
    this.theta0 = new Float64Array(ntypes + 1);
    this.setflag = new Uint8Array(ntypes + 1);
  }

  /** angle_coeff N-range file keyword (AngleTable::coeff). */
  coeff(args: string[], ctx: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Illegal angle_coeff command: must have 3 arguments');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const text = readTableText(ctx, args[1], 'angle');
    const tb = this.readTable(text, args[1], args[2]);
    // error check on table parameters
    if (tb.ninput <= 1) throw new StyleError(`Invalid angle table length: ${tb.ninput}`);
    const alo = tb.afile[0];
    const ahi = tb.afile[tb.ninput - 1];
    if (Math.abs(alo - 0.0) > TINY || Math.abs(ahi - 180.0) > TINY) {
      throw new StyleError('Angle table must range from 0 to 180 degrees');
    }
    // convert theta from degrees to radians
    for (let i = 0; i < tb.ninput; i++) {
      tb.afile[i] *= DEG2RAD;
      tb.ffile[i] *= RAD2DEG;
    }
    // spline read-in and compute a,e,f vectors within table
    this.splineTable(tb);
    this.computeTable(tb);
    const index = this.tables.length;
    this.tables.push(tb);
    for (let i = ilo; i <= ihi; i++) {
      this.tabindex[i] = index;
      this.setflag[i] = 1;
      this.theta0[i] = tb.theta0;
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.setflag[t]) throw new StyleError(`All angle coeffs are not set. Status:\n(type ${t} has no table)`);
    }
  }

  /** The equilibrium angle in degrees (Angle::equilibrium: RAD2DEG * theta0). */
  equilibrium(type: number): number { return RAD2DEG * this.theta0[type]; }
  /** The equilibrium angle in radians, as the style stores it (AngleTable::equilibrium_angle). */
  equilibriumRadians(type: number): number { return this.theta0[type]; }

  /** read_table (angle_table.cpp): parameter line, one raw line skipped, N data lines "index angle energy force". */
  private readTable(text: string, file: string, keyword: string): AngleTab {
    const reader = new TableFileReader(text, file, 'angle');
    const start = reader.findSectionStart(keyword);
    if (start === null) throw new StyleError(`Did not find keyword ${keyword} in table file`);
    const pl = reader.nextLine();
    if (pl === null) throw new StyleError(`Did not find keyword ${keyword} in table file`);
    const p = paramExtract(pl);
    const n = p.ninput;
    const len = Math.max(n, 0);
    const tb: AngleTab = {
      ninput: n, afile: new Float64Array(len), efile: new Float64Array(len), ffile: new Float64Array(len),
      e2file: new Float64Array(0), f2file: new Float64Array(0), fpflag: p.fpflag, fplo: p.fplo, fphi: p.fphi, theta0: p.theta0,
      delta: 0, invdelta: 0, deltasq6: 0,
      ang: new Float64Array(0), e: new Float64Array(0), de: new Float64Array(0), f: new Float64Array(0), df: new Float64Array(0),
      e2: new Float64Array(0), f2: new Float64Array(0),
    };
    reader.skipLine();
    for (let i = 0; i < n; i++) {
      const line = reader.nextLine();
      if (line === null) throw new StyleError(`Data missing when parsing angle table '${keyword}' line ${i + 1} of ${n}.`);
      try {
        const w = new WordStream(splitWords(line));
        w.nextInt();
        tb.afile[i] = w.nextDouble();
        tb.efile[i] = w.nextDouble();
        tb.ffile[i] = w.nextDouble();
      } catch (e) {
        if (e instanceof TokenError) throw new StyleError(`Error parsing angle table '${keyword}' line ${i + 1} of ${n}. ${e.message}\nLine was: ${line}`);
        throw e;
      }
    }
    return tb;
  }

  /** build spline representation of e,f over the read-in table (AngleTable::spline_table). */
  private splineTable(tb: AngleTab): void {
    const n = tb.ninput;
    tb.e2file = new Float64Array(n);
    tb.f2file = new Float64Array(n);
    const ep0 = -tb.ffile[0];
    const epn = -tb.ffile[n - 1];
    spline(tb.afile, tb.efile, n, ep0, epn, tb.e2file);
    if (!tb.fpflag) {
      tb.fplo = (tb.ffile[1] - tb.ffile[0]) / (tb.afile[1] - tb.afile[0]);
      tb.fphi = (tb.ffile[n - 1] - tb.ffile[n - 2]) / (tb.afile[n - 1] - tb.afile[n - 2]);
    }
    const fp0 = tb.fplo;
    const fpn = tb.fphi;
    spline(tb.afile, tb.ffile, n, fp0, fpn, tb.f2file);
  }

  /** a,e,f vectors at N evenly spaced angles from 0 to pi (AngleTable::compute_table). */
  private computeTable(tb: AngleTab): void {
    const N = this.tablength;
    const tlm1 = N - 1;
    tb.delta = MY_PI / tlm1;
    tb.invdelta = 1.0 / tb.delta;
    tb.deltasq6 = tb.delta * tb.delta / 6.0;
    tb.ang = new Float64Array(N);
    tb.e = new Float64Array(N);
    tb.de = new Float64Array(N);
    tb.f = new Float64Array(N);
    tb.df = new Float64Array(N);
    tb.e2 = new Float64Array(N);
    tb.f2 = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const a = i * tb.delta;
      tb.ang[i] = a;
      tb.e[i] = splint(tb.afile, tb.efile, tb.e2file, tb.ninput, a);
      tb.f[i] = splint(tb.afile, tb.ffile, tb.f2file, tb.ninput, a);
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
    spline(tb.ang, tb.e, N, ep0, epn, tb.e2);
    spline(tb.ang, tb.f, N, tb.fplo, tb.fphi, tb.f2);
  }

  /** Bin index of angle x, clamped to the table (AngleTable::uf_lookup / u_lookup). */
  private index(tb: AngleTab, x: number): number {
    let itable = Math.trunc(x * tb.invdelta);
    if (itable < 0) itable = 0;
    if (itable >= this.tablength) itable = this.tablength - 1;
    return itable;
  }

  /**
   * uf_lookup (angle_table.cpp): energy and -dE/dtheta at angle x (radians). The spline reads entry itable + 1; at the
   * last node that entry is past the end in C++ (an undefined read), so the last node's value is used there.
   */
  private ufLookup(type: number, x: number): { u: number; f: number } {
    if (!Number.isFinite(x)) throw new StyleError('Illegal angle in angle style table');
    const tb = this.tables[this.tabindex[type]];
    const itable = this.index(tb, x);
    if (this.tabstyle === 'linear') {
      const fraction = (x - tb.ang[itable]) * tb.invdelta;
      return { u: tb.e[itable] + fraction * tb.de[itable], f: tb.f[itable] + fraction * tb.df[itable] };
    }
    const i1 = itable + 1 < this.tablength ? itable + 1 : itable;
    const b = (x - tb.ang[itable]) * tb.invdelta;
    const a = 1.0 - b;
    const u = a * tb.e[itable] + b * tb.e[i1] + ((a * a * a - a) * tb.e2[itable] + (b * b * b - b) * tb.e2[i1]) * tb.deltasq6;
    const f = a * tb.f[itable] + b * tb.f[i1] + ((a * a * a - a) * tb.f2[itable] + (b * b * b - b) * tb.f2[i1]) * tb.deltasq6;
    return { u, f };
  }

  /** AngleTable::compute: the angle at the middle atom, the tabulated force along each bond, tallies in list order. */
  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    for (let n = 0; n < A.n; n++) {
      const i1 = atomIndex(bc, A.atoms[3 * n], 'angle');
      const i2 = atomIndex(bc, A.atoms[3 * n + 1], 'angle');
      const i3 = atomIndex(bc, A.atoms[3 * n + 2], 'angle');
      const type = A.type[n];
      // 1st bond: x[i1] - x[i2]; 2nd bond: x[i3] - x[i2] (copies nearest to the middle atom)
      delta(bc, i2, i1, d1);
      const delx1 = d1[0], dely1 = d1[1], delz1 = d1[2];
      const rsq1 = delx1 * delx1 + dely1 * dely1 + delz1 * delz1;
      const r1 = Math.sqrt(rsq1);
      delta(bc, i2, i3, d2);
      const delx2 = d2[0], dely2 = d2[1], delz2 = d2[2];
      const rsq2 = delx2 * delx2 + dely2 * dely2 + delz2 * delz2;
      const r2 = Math.sqrt(rsq2);
      // angle (cos and sin)
      let c = delx1 * delx2 + dely1 * dely2 + delz1 * delz2;
      c /= r1 * r2;
      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;
      let s = Math.sqrt(1.0 - c * c);
      if (s < SMALL) s = SMALL;
      s = 1.0 / s;
      // tabulated force & energy
      const theta = libmAcos(c);
      const { u, f: mdu } = this.ufLookup(type, theta);
      const eangle = u;
      const a = mdu * s;
      const a11 = a * c / rsq1;
      const a12 = -a / (r1 * r2);
      const a22 = a * c / rsq2;
      const f1x = a11 * delx1 + a12 * delx2, f1y = a11 * dely1 + a12 * dely2, f1z = a11 * delz1 + a12 * delz2;
      const f3x = a22 * delx2 + a12 * delx1, f3y = a22 * dely2 + a12 * dely1, f3z = a22 * delz2 + a12 * delz1;
      // apply force to each of 3 atoms, and Angle::ev_tally (the virial, a third of the energy and of the virial to each
      // atom, the centroid virial)
      bc.acc.eangle += eangle;
      tallyAngle(bc, i1, i2, i3, eangle, [f1x, f1y, f1z], [f3x, f3y, f3z], [delx1, dely1, delz1], [delx2, dely2, delz2]);
    }
  }
}

/** utils::inumeric: an integer argument, or the native error. */
const intArg = (w: string): number => {
  if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
  return Number(w);
};
