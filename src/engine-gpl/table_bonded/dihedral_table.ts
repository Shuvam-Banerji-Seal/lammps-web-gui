/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MOLECULE/dihedral_table.cpp and
 * src/MOLECULE/dihedral_table.h (DihedralTable: settings, coeff, read_table, param_extract, spline_table,
 * compute_table, uf_lookup, Phi, compute). Contributing author: Andrew Jewett (jewett.aij at gmail); the cyclic
 * tridiagonal solver was borrowed from the GSL tridiag.c by Gerard Jungman. Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dihedral_table.html: "dihedral_style table style N", where "style = linear or spline" and "N = use
 * N values to interpolate the table". The coefficients are "dihedral_coeff N file keyword", the section has the
 * parameters "N value [NOF] [DEGREES or RADIANS]" and N lines "index angle energy [force]". The angles must be
 * increasing and their range less than 360 degrees (2*PI radians). The "NOF" keyword omits the force column.
 * CHECKU and CHECKF write check files; the notebook cannot write them, so those keywords are refused by name.
 *
 * The dihedral angle is computed here, not with the dihedral harmonic formula: Phi() of dihedral_table.cpp, with the
 * gradients of its compute(). The bond vectors are the copies nearest to the second atom (bonded_util.ts delta),
 * then minimum-imaged as Phi() does.
 */
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { cycLin, cycSpline, cycSplint } from './splines';
import { TableFileReader, TokenError, WordStream, readTableText, splitWords } from './tablefile';

const MY_PI = 3.14159265358979323846;
const MY_2PI = 6.28318530717958647692;

/** A table section (Table in dihedral_table.h). */
export interface DihedralTab {
  ninput: number;
  fUnspecified: boolean;
  useDegrees: boolean;
  phifile: Float64Array;
  efile: Float64Array;
  ffile: Float64Array;
  e2file: Float64Array;
  f2file: Float64Array;
  delta: number;
  invdelta: number;
  deltasq6: number;
  phi: Float64Array;
  e: Float64Array;
  de: Float64Array;
  f: Float64Array;
  df: Float64Array;
  e2: Float64Array;
  f2: Float64Array;
}

const numArg = (w: string): number => {
  const v = Number(w);
  if (w.trim() === '' || Number.isNaN(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  return v;
};

/** The table parameters of one section; checkU/checkF are refused (they write files). */
const paramExtract = (line: string): { ninput: number; fUnspecified: boolean; useDegrees: boolean } => {
  let ninput = 0, fUnspecified = false, useDegrees = true;
  try {
    const w = new WordStream(splitWords(line));
    while (w.hasNext()) {
      const word = w.nextString();
      if (word === 'N') {
        ninput = w.nextInt();
      } else if (word === 'NOF') {
        fUnspecified = true;
      } else if (word === 'DEGREES' || word === 'degrees') {
        useDegrees = true;
      } else if (word === 'RADIANS' || word === 'radians') {
        useDegrees = false;
      } else if (word === 'CHECKU' || word === 'CHECKF') {
        const file = w.nextString();
        throw new StyleError(`dihedral table keyword ${word} ${file} is not supported: it writes a check file, and the notebook's styles write no files`);
      } else {
        throw new StyleError(`Unknown keyword ${word} in dihedral table parameters`);
      }
    }
  } catch (e) {
    if (e instanceof TokenError) throw new StyleError(e.message);
    throw e;
  }
  if (ninput === 0) throw new StyleError('Dihedral table parameters did not set N');
  return { ninput, fUnspecified, useDegrees };
};

/** MathExtra helpers of dihedral_table.cpp (cross3, dot3, norm3 of math_extra.h). */
const cross3 = (a: number[], b: number[], out: number[]): void => {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
};
const dot3 = (a: number[], b: number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm3 = (v: number[]): void => {
  const val = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (val > 0.0) {
    const scale = 1.0 / Math.sqrt(val);
    v[0] *= scale;
    v[1] *= scale;
    v[2] *= scale;
  }
};

export class DihedralTable extends Bonded {
  readonly name: string = 'table';
  readonly kind = 'dihedral' as const;

  protected tabstyle: 'linear' | 'spline' = 'linear';
  protected tablength = 0;
  protected tables: DihedralTab[] = [];
  protected tabindex = new Int32Array(0);
  protected setflag = new Uint8Array(0);

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 2) throw new StyleError('Illegal dihedral_style command: must have 2 arguments');
    if (args[0] === 'linear') this.tabstyle = 'linear';
    else if (args[0] === 'spline') this.tabstyle = 'spline';
    else throw new StyleError(`Unknown table style ${args[0]} in dihedral style table`);
    this.tablength = intArg(args[1]);
    if (this.tablength < 3) throw new StyleError(`Illegal number of dihedral table entries: ${args[1]}`);
    // delete old tables, since cannot just change settings (the next coeff allocates again)
    this.tables = [];
    this.allocate(this.ntypes);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.tabindex = new Int32Array(ntypes + 1);
    this.setflag = new Uint8Array(ntypes + 1);
  }

  /** dihedral_coeff N-range file keyword (DihedralTable::coeff). */
  coeff(args: string[], ctx: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Illegal dihedral_coeff command: must have 3 arguments');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    this.addTable(ilo, ihi, args[1], args[2], ctx, false);
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.setflag[t]) throw new StyleError(`All dihedral coeffs are not set. Status:\n(type ${t} has no table)`);
    }
  }

  /**
   * Reads one table section and stores it for the types ilo..ihi (the body of DihedralTable::coeff; the cut style
   * shares it). cut selects the messages of dihedral_table_cut.cpp, which name the file where the base style names
   * the keyword, and have no period after the range messages.
   */
  protected addTable(ilo: number, ihi: number, file: string, keyword: string, ctx: StyleContext, cut: boolean): void {
    const text = readTableText(ctx, file, 'dihedral');
    const tb = this.readTable(text, file, keyword);
    const label = cut ? file : keyword;
    // --- check the angle data for range errors ---
    // ---  and resolve issues with periodicity  ---
    if (tb.ninput < 2) throw new StyleError(`Invalid dihedral table length: ${label}`);
    if (tb.ninput === 2 && this.tabstyle === 'spline') throw new StyleError(`Invalid dihedral spline table length: ${label} (Try linear)`);
    // check for monotonicity
    for (let i = 0; i < tb.ninput - 1; i++) {
      if (tb.phifile[i] >= tb.phifile[i + 1]) {
        let msg = `Dihedral table values are not increasing (${label}, entry ${i + 1})`;
        if (i === 0) msg += '\n(This is probably a mistake with your table format.)\n';
        throw new StyleError(msg);
      }
    }
    // check the range of angles
    const philo = tb.phifile[0];
    const phihi = tb.phifile[tb.ninput - 1];
    const end = cut ? '' : '.';
    if (tb.useDegrees) {
      if (phihi - philo >= 360) throw new StyleError(`Dihedral table angle range must be < 360 degrees (${label})${end}`);
    } else if (phihi - philo >= MY_2PI) {
      throw new StyleError(`Dihedral table angle range must be < 2*PI radians (${label})${end}`);
    }
    // convert phi from degrees to radians
    if (tb.useDegrees) {
      for (let i = 0; i < tb.ninput; i++) {
        tb.phifile[i] *= MY_PI / 180.0;
        // I assume that if angles are in degrees, then the forces (f=dU/dphi) are specified with "phi" in degrees as well.
        tb.ffile[i] *= 180.0 / MY_PI;
      }
    }
    // We want all the phi dihedral angles to lie in the range from 0 to 2*PI, sorted in increasing order.
    this.reimage(tb);
    // spline read-in and compute r,e,f vectors within table
    this.splineTable(tb);
    this.computeTable(tb);
    // store ptr to table in tabindex
    const index = this.tables.length;
    this.tables.push(tb);
    for (let i = ilo; i <= ihi; i++) {
      this.tabindex[i] = index;
      this.setflag[i] = 1;
    }
  }

  /** Re-images the angles into [0, 2*PI) and rotates the table so that it starts at the first angle after the wrap. */
  private reimage(tb: DihedralTab): void {
    const n = tb.ninput;
    const phiTmp = new Float64Array(n), ffTmp = new Float64Array(n), efTmp = new Float64Array(n);
    // After re-imaging, does the range of angles cross the 0 or 2*PI boundary? If so, find the discontinuity.
    let iDisc = n;
    for (let i = 0; i < n; i++) {
      let phi = tb.phifile[i];
      // Add a multiple of 2*PI to phi until it lies in the range [0, 2*PI).
      phi -= MY_2PI * Math.floor(phi / MY_2PI);
      phiTmp[i] = phi;
      efTmp[i] = tb.efile[i];
      ffTmp[i] = tb.ffile[i];
      if (i > 0 && phiTmp[i] < phiTmp[i - 1]) {
        // at most one discontinuity: the data was sorted before imaging and spans less than 2*PI
        iDisc = i;
      }
    }
    let I = 0;
    for (let i = iDisc; i < n; i++) {
      tb.phifile[I] = phiTmp[i];
      tb.efile[I] = efTmp[i];
      tb.ffile[I] = ffTmp[i];
      I++;
    }
    for (let i = 0; i < iDisc; i++) {
      tb.phifile[I] = phiTmp[i];
      tb.efile[I] = efTmp[i];
      tb.ffile[I] = ffTmp[i];
      I++;
    }
  }

  /** read_table (dihedral_table.cpp): no raw line is skipped here; NOF sections have no force column. */
  protected readTable(text: string, file: string, keyword: string): DihedralTab {
    const reader = new TableFileReader(text, file, 'dihedral');
    const start = reader.findSectionStart(keyword);
    if (start === null) throw new StyleError(`Did not find keyword ${keyword} in table file`);
    const pl = reader.nextLine();
    if (pl === null) throw new StyleError(`Did not find keyword ${keyword} in table file`);
    const p = paramExtract(pl);
    const n = p.ninput;
    const len = Math.max(n, 0);
    const tb: DihedralTab = {
      ninput: n, fUnspecified: p.fUnspecified, useDegrees: p.useDegrees,
      phifile: new Float64Array(len), efile: new Float64Array(len), ffile: new Float64Array(len),
      e2file: new Float64Array(0), f2file: new Float64Array(0),
      delta: 0, invdelta: 0, deltasq6: 0,
      phi: new Float64Array(0), e: new Float64Array(0), de: new Float64Array(0), f: new Float64Array(0), df: new Float64Array(0),
      e2: new Float64Array(0), f2: new Float64Array(0),
    };
    for (let i = 0; i < n; i++) {
      const line = reader.nextLine();
      if (line === null) throw new StyleError(`Data missing when parsing dihedral table '${keyword}' line ${i + 1} of ${n}.`);
      try {
        const w = new WordStream(splitWords(line));
        w.nextInt();
        tb.phifile[i] = w.nextDouble();
        tb.efile[i] = w.nextDouble();
        if (!tb.fUnspecified) tb.ffile[i] = w.nextDouble();
      } catch (e) {
        if (e instanceof TokenError) throw new StyleError(`Error parsing dihedral table '${keyword}' line ${i + 1} of ${n}. ${e.message}\nLine was: ${line}`);
        throw e;
      }
    }
    return tb;
  }

  /**
   * Splines of e and f over the read-in table, with the crude force-energy consistency check of
   * DihedralTable::spline_table ("Try \"NOF\"").
   */
  private splineTable(tb: DihedralTab): void {
    const n = tb.ninput;
    tb.e2file = cycSpline(tb.phifile, tb.efile, n, MY_2PI);
    if (!tb.fUnspecified) tb.f2file = cycSpline(tb.phifile, tb.ffile, n, MY_2PI);
    if (tb.fUnspecified) return;
    let numDisagreements = 0;
    for (let i = 0; i < n; i++) {
      // Calculate what the force should be at the control points by linear interpolation of the derivatives of the energy.
      const phiI = tb.phifile[i];
      let phiIm1: number, phiIp1: number;
      let im1 = i - 1;
      if (im1 < 0) {
        im1 += n;
        phiIm1 = tb.phifile[im1] - MY_2PI;
      } else phiIm1 = tb.phifile[im1];
      let ip1 = i + 1;
      if (ip1 >= n) {
        ip1 -= n;
        phiIp1 = tb.phifile[ip1] + MY_2PI;
      } else phiIp1 = tb.phifile[ip1];
      // midpoints above and below phi_i
      const phiLo = 0.5 * (phiIm1 + phiI);
      const phiHi = 0.5 * (phiI + phiIp1);
      // linear approximation to the derivative at these two midpoints
      const dUdphiLo = (tb.efile[i] - tb.efile[im1]) / (phiI - phiIm1);
      const dUdphiHi = (tb.efile[ip1] - tb.efile[i]) / (phiIp1 - phiI);
      // the derivative at phi_i by linear interpolation
      const a = (phiI - phiLo) / (phiHi - phiLo);
      const b = (phiHi - phiI) / (phiHi - phiLo);
      const dUdphi = a * dUdphiLo + b * dUdphiHi;
      const f = -dUdphi;
      if (f !== 0.0 && tb.ffile[i] !== 0.0 && (f / tb.ffile[i] < 0.5 || f / tb.ffile[i] > 2.0)) numDisagreements++;
    }
    if (numDisagreements > Math.trunc(n / 2) && numDisagreements > 2) {
      throw new StyleError('Dihedral table has inconsistent forces and energies. (Try "NOF".)\n');
    }
  }

  /** a, e, f on the N = tablength cyclic bins of 2*PI (DihedralTable::compute_table). */
  private computeTable(tb: DihedralTab): void {
    const N = this.tablength;
    tb.delta = MY_2PI / N;
    tb.invdelta = 1.0 / tb.delta;
    tb.deltasq6 = tb.delta * tb.delta / 6.0;
    tb.phi = new Float64Array(N);
    tb.e = new Float64Array(N);
    tb.de = new Float64Array(N);
    tb.f = new Float64Array(N);
    tb.df = new Float64Array(N);
    tb.e2 = new Float64Array(N);
    tb.f2 = new Float64Array(N);
    if (this.tabstyle === 'spline') {
      // Cubic spline interpolation calculates the entries of the internal table.
      for (let i = 0; i < N; i++) {
        const phi = i * tb.delta;
        tb.phi[i] = phi;
        tb.e[i] = cycSplint(tb.phifile, tb.efile, tb.e2file, tb.ninput, MY_2PI, phi);
        if (!tb.fUnspecified) tb.f[i] = cycSplint(tb.phifile, tb.ffile, tb.f2file, tb.ninput, MY_2PI, phi);
      }
    } else {
      if (!tb.fUnspecified) {
        for (let i = 0; i < N; i++) {
          const phi = i * tb.delta;
          tb.phi[i] = phi;
          tb.e[i] = cycLin(tb.phifile, tb.efile, tb.ninput, MY_2PI, phi);
          tb.f[i] = cycLin(tb.phifile, tb.ffile, tb.ninput, MY_2PI, phi);
        }
      } else {
        for (let i = 0; i < N; i++) {
          const phi = i * tb.delta;
          tb.phi[i] = phi;
          tb.e[i] = cycLin(tb.phifile, tb.efile, tb.ninput, MY_2PI, phi);
        }
        // Without forces the f array is the average of the linear slopes on either side of each node (-dE/dphi).
        for (let i = 0; i < N; i++) {
          let im1 = i - 1;
          if (im1 < 0) im1 += N;
          let ip1 = i + 1;
          if (ip1 >= N) ip1 -= N;
          const dedx = (tb.e[ip1] - tb.e[im1]) / (2.0 * tb.delta);
          tb.f[i] = -dedx;
        }
      }
      // Fill the linear interpolation tables (de, df)
      for (let i = 0; i < N; i++) {
        let ip1 = i + 1;
        if (ip1 >= N) ip1 -= N;
        tb.de[i] = tb.e[ip1] - tb.e[i];
        tb.df[i] = tb.f[ip1] - tb.f[i];
      }
    }
    tb.e2 = cycSpline(tb.phi, tb.e, N, MY_2PI);
    if (!tb.fUnspecified) tb.f2 = cycSpline(tb.phi, tb.f, N, MY_2PI);
  }

  /**
   * uf_lookup (dihedral_table.h): the energy and -dE/dphi at angle x in [0, 2*PI); the bin is the truncated x / delta,
   * and entries past the end wrap to the start (the table is periodic).
   */
  protected ufLookup(type: number, x: number): { u: number; f: number } {
    const tb = this.tables[this.tabindex[type]];
    const N = this.tablength;
    const xOverDelta = x * tb.invdelta;
    let i = Math.trunc(xOverDelta);
    const b = xOverDelta - i;
    if (i >= N) i -= N;
    let ip1 = i + 1;
    if (ip1 >= N) ip1 -= N;
    if (this.tabstyle === 'linear') {
      return { u: tb.e[i] + b * tb.de[i], f: tb.f[i] + b * tb.df[i] }; // works even if f_unspecified (f holds -dE/dphi)
    }
    const a = 1.0 - b;
    const u = a * tb.e[i] + b * tb.e[ip1] + ((a * a * a - a) * tb.e2[i] + (b * b * b - b) * tb.e2[ip1]) * tb.deltasq6;
    let f: number;
    if (tb.fUnspecified) {
      f = (tb.e[i] - tb.e[ip1]) * tb.invdelta
        + ((3.0 * a * a - 1.0) * tb.e2[i] + (1.0 - 3.0 * b * b) * tb.e2[ip1]) * tb.delta / 6.0;
    } else {
      f = a * tb.f[i] + b * tb.f[ip1] + ((a * a * a - a) * tb.f2[i] + (b * b * b - b) * tb.f2[ip1]) * tb.deltasq6;
    }
    return { u, f };
  }

  /**
   * DihedralTable::compute: Phi() with the gradients of the dihedral angle with respect to each atom (the
   * Blondel-Karplus form of compute()), the force -dU/dphi times each gradient, and the tallies of ev_tally.
   */
  compute(bc: BondedCompute): void {
    const D = bc.s.topo.dihedrals;
    const f = bc.f;
    const g = bc.geom;
    for (let n = 0; n < D.n; n++) {
      const i1 = atomIndex(bc, D.atoms[4 * n], 'dihedral');
      const i2 = atomIndex(bc, D.atoms[4 * n + 1], 'dihedral');
      const i3 = atomIndex(bc, D.atoms[4 * n + 2], 'dihedral');
      const i4 = atomIndex(bc, D.atoms[4 * n + 3], 'dihedral');
      const type = D.type[n];
      // the bond vectors are the raw differences of the positions (copies nearest to the first atom of each bond)
      const pI = [0, 0, 0], pK = [0, 0, 0], pL = [0, 0, 0];
      delta(bc, i2, i1, pI); // x1 - x2
      delta(bc, i2, i3, pK); // x3 - x2
      delta(bc, i3, i4, pL); // x4 - x3
      const vb12 = [-pI[0], -pI[1], -pI[2]];
      const vb23 = [pK[0], pK[1], pK[2]];
      const vb34 = [pL[0], pL[1], pL[2]];
      // Phi(): minimum image of each bond vector, then the normals of the two planes
      g.minimumImage(vb12);
      g.minimumImage(vb23);
      g.minimumImage(vb34);
      const n123 = [0, 0, 0], n234 = [0, 0, 0];
      cross3(vb23, vb12, n123); // n123 = vb23 x vb12
      cross3(vb23, vb34, n234); // n234 = vb23 x vb34
      norm3(n123);
      norm3(n234);
      let cosPhi = -dot3(n123, n234);
      if (cosPhi > 1.0) cosPhi = 1.0;
      else if (cosPhi < -1.0) cosPhi = -1.0;
      let phi = Math.acos(cosPhi);
      if (dot3(n123, vb34) > 0.0) {
        phi = -phi; // negative dihedral angles are possible only in 3-D
        phi += MY_2PI; // ensure phi is always in the range 0 to 2*PI
      }
      // ------ gradients of phi with the atomic positions (compute()) ------
      const dot123 = dot3(vb12, vb23);
      const dot234 = dot3(vb23, vb34);
      const L23sqr = dot3(vb23, vb23);
      const L23 = Math.sqrt(L23sqr);
      let invL23sqr = 0.0, invL23 = 0.0;
      if (L23sqr !== 0.0) {
        invL23sqr = 1.0 / L23sqr;
        invL23 = 1.0 / L23;
      }
      const negInvL23 = -invL23;
      const dot123OverL23sqr = dot123 * invL23sqr;
      const dot234OverL23sqr = dot234 * invL23sqr;
      const proj12on23 = [0, 0, 0], proj34on23 = [0, 0, 0], perp12on23 = [0, 0, 0], perp34on23 = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        proj12on23[d] = vb23[d] * dot123OverL23sqr;
        proj34on23[d] = vb23[d] * dot234OverL23sqr;
        perp12on23[d] = vb12[d] - proj12on23[d];
        perp34on23[d] = vb34[d] - proj34on23[d];
      }
      const perp12Len = Math.sqrt(dot3(perp12on23, perp12on23));
      const perp34Len = Math.sqrt(dot3(perp34on23, perp34on23));
      let invPerp12 = 0.0;
      if (perp12Len !== 0.0) invPerp12 = 1.0 / perp12Len;
      let invPerp34 = 0.0;
      if (perp34Len !== 0.0) invPerp34 = 1.0 / perp34Len;
      const dphidx1 = [0, 0, 0], dphidx4 = [0, 0, 0], dphidx2 = [0, 0, 0], dphidx3 = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        dphidx1[d] = n123[d] * invPerp12;
        dphidx4[d] = n234[d] * invPerp34;
      }
      const proj12onLen = dot123 * invL23;
      const proj34onLen = dot234 * invL23;
      const dphi123dx2 = negInvL23 * (L23 + proj12onLen);
      const dphi234dx2 = invL23 * proj34onLen;
      const dphi234dx3 = negInvL23 * (L23 + proj34onLen);
      const dphi123dx3 = invL23 * proj12onLen;
      for (let d = 0; d < 3; d++) {
        dphidx2[d] = dphi123dx2 * dphidx1[d] + dphi234dx2 * dphidx4[d];
        dphidx3[d] = dphi123dx3 * dphidx1[d] + dphi234dx3 * dphidx4[d];
      }
      // ----- energy and force in the phi direction -----
      const { u, f: mDuDphi } = this.ufLookup(type, phi);
      const edihedral = u;
      const f1 = [0, 0, 0], f2 = [0, 0, 0], f3 = [0, 0, 0], f4 = [0, 0, 0];
      for (let d = 0; d < 3; d++) {
        f1[d] = mDuDphi * dphidx1[d];
        f2[d] = mDuDphi * dphidx2[d];
        f3[d] = mDuDphi * dphidx3[d];
        f4[d] = mDuDphi * dphidx4[d];
      }
      f[3 * i1] += f1[0]; f[3 * i1 + 1] += f1[1]; f[3 * i1 + 2] += f1[2];
      f[3 * i2] += f2[0]; f[3 * i2 + 1] += f2[1]; f[3 * i2 + 2] += f2[2];
      f[3 * i3] += f3[0]; f[3 * i3 + 1] += f3[1]; f[3 * i3 + 2] += f3[2];
      f[3 * i4] += f4[0]; f[3 * i4 + 1] += f4[1]; f[3 * i4 + 2] += f4[2];
      tallyDihedral(bc, [i1, i2, i3, i4], edihedral, f1, f3, f4,
        -vb12[0], -vb12[1], -vb12[2], vb23[0], vb23[1], vb23[2], vb34[0], vb34[1], vb34[2]);
    }
  }
}

/**
 * Dihedral::ev_tally (dihedral.cpp at b332f28) with newton_bond on: the energy and the per-atom quarters, and the
 * virial v = vb1 f1 + vb2 f3 + (vb3 + vb2) f4 with the six components in the order xx yy zz xy xz yz.
 */
export const tallyDihedral = (
  bc: BondedCompute, atoms: number[], edihedral: number, f1: number[], f3: number[], f4: number[],
  vb1x: number, vb1y: number, vb1z: number, vb2x: number, vb2y: number, vb2z: number, vb3x: number, vb3y: number, vb3z: number,
): void => {
  bc.acc.edihed += edihedral;
  if (bc.eatom) {
    const q = 0.25 * edihedral;
    for (const i of atoms) bc.eatom[i] += q;
  }
  const v = [
    vb1x * f1[0] + vb2x * f3[0] + (vb3x + vb2x) * f4[0],
    vb1y * f1[1] + vb2y * f3[1] + (vb3y + vb2y) * f4[1],
    vb1z * f1[2] + vb2z * f3[2] + (vb3z + vb2z) * f4[2],
    vb1x * f1[1] + vb2x * f3[1] + (vb3x + vb2x) * f4[1],
    vb1x * f1[2] + vb2x * f3[2] + (vb3x + vb2x) * f4[2],
    vb1y * f1[2] + vb2y * f3[2] + (vb3y + vb2y) * f4[2],
  ];
  for (let c = 0; c < 6; c++) bc.virial[c] += v[c];
  if (bc.vatom) {
    for (const i of atoms) for (let c = 0; c < 6; c++) bc.vatom[6 * i + c] += 0.25 * v[c];
  }
};

/** utils::inumeric: an integer argument, or the native error. */
const intArg = (w: string): number => {
  if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
  return Number(w);
};

export { numArg };
