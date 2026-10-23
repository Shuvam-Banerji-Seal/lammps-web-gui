/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DPD-REACT/pair_multi_lucy.cpp and pair_multi_lucy.h at commit b332f28 (James Larentzos and Joshua
 * Moore, U.S. Army Research Laboratory; the multi-body Lucy-weight density model of J. D. Moore, B. C. Barnes,
 * S. Izvekov, M. Lisal, M. S. Sellers, D. E. Taylor and J. K. Brennan). Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Non-rx pair_style multi/lucy (the reactive variant is not ported): each atom's local density is a sum of Lucy
 * weights of its neighbors inside the cutoff, and the pair force and the per-atom energy are read from a table
 * indexed by the squared density (lookup or linear table styles). Tables are read with the native table file
 * format (keyword section, "N n R|RSQ lo hi" parameter line, then index r E F lines).
 */
import { Pair, StyleError, typeBounds, type PairCompute, type PairConstructorWhere, type StyleContext, potentialFile } from '../../engine/force/types';
import { atomHasFlag } from '../atom_flags';
import { inumeric, parseNum } from '../../engine/force/util';
import { NEIGHMASK } from '../../engine/neighbor';

const PI = Math.PI;

export interface Table {
  ninput: number;
  /** 0 none (r from the file), 1 R (linear in r^2 ... r), 2 RSQ (linear in r^2) */
  rflag: 0 | 1 | 2;
  rlo: number;
  rhi: number;
  fpflag: number;
  fplo: number;
  fphi: number;
  rfile: Float64Array;
  efile: Float64Array;
  ffile: Float64Array;
  e2file: Float64Array;
  f2file: Float64Array;
  cut: number;
  innersq: number;
  delta: number;
  invdelta: number;
  rsq: Float64Array;
  e: Float64Array;
  f: Float64Array;
  de: Float64Array;
  df: Float64Array;
  match: boolean;
}

/** The native lookup of a keyword section: the line after the keyword is the parameter line, then ninput entries. */
const readSection = (text: string, keyword: string, file: string): { params: string[]; rows: string[][] } => {
  const lines = text.split(/\r?\n/).map((l) => l.replace(/#.*$/, '').trim()).filter((l) => l.length > 0);
  const k = lines.findIndex((l) => l === keyword);
  if (k < 0) throw new StyleError(`pair multi/lucy: did not find keyword '${keyword}' in '${file}'`);
  if (k + 1 >= lines.length) throw new StyleError(`pair multi/lucy: no parameter line after keyword '${keyword}' in '${file}'`);
  return { params: lines[k + 1].split(/\s+/), rows: lines.slice(k + 2).map((l) => l.split(/\s+/)) };
};

/** spline of pair_multi_lucy.cpp (natural cubic spline with end derivatives yp1, ypn; 0.99e30 = natural end). */
const spline = (x: Float64Array, y: Float64Array, n: number, yp1: number, ypn: number, y2: Float64Array): void => {
  const u = new Float64Array(n);
  if (yp1 > 0.99e30) { y2[0] = 0; u[0] = 0; } else {
    y2[0] = -0.5;
    u[0] = (3 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
  }
  for (let i = 1; i < n - 1; i++) {
    const sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    const p = sig * y2[i - 1] + 2;
    y2[i] = (sig - 1) / p;
    let ui = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    ui = (6 * ui / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    u[i] = ui;
  }
  let qn: number, un: number;
  if (ypn > 0.99e30) { qn = 0; un = 0; } else {
    qn = 0.5;
    un = (3 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
  }
  y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1);
  for (let k = n - 2; k >= 0; k--) y2[k] = y2[k] * y2[k + 1] + u[k];
};

/** splint of pair_multi_lucy.cpp: cubic spline value at x. */
const splint = (xa: Float64Array, ya: Float64Array, y2a: Float64Array, n: number, x: number): number => {
  let klo = 0, khi = n - 1;
  while (khi - klo > 1) {
    const k = (khi + klo) >> 1;
    if (xa[k] > x) khi = k; else klo = k;
  }
  const h = xa[khi] - xa[klo];
  const a = (xa[khi] - x) / h;
  const b = (x - xa[klo]) / h;
  return a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6;
};

/** A table with every field empty (the fields are filled by readTable, coeff and computeTable). */
export const newTable = (): Table => {
  return {
    ninput: 0, rflag: 0, rlo: 0, rhi: 0, fpflag: 0, fplo: 0, fphi: 0,
    rfile: new Float64Array(0), efile: new Float64Array(0), ffile: new Float64Array(0),
    e2file: new Float64Array(0), f2file: new Float64Array(0), cut: 0, innersq: 0, delta: 0, invdelta: 0,
    rsq: new Float64Array(0), e: new Float64Array(0), f: new Float64Array(0), de: new Float64Array(0), df: new Float64Array(0),
    match: false,
  };
};

/**
 * computeLocalDensity of pair_multi_lucy.cpp: the density of each owned atom, the sum of the Lucy weights of its
 * neighbors inside the cutoff (the pair's weight is added to both atoms; a ghost's weight goes to its owner).
 */
export const lucyLocalDensity = (pc: PairCompute, cutsq: Float64Array, nt: number): Float64Array => {
  const list = pc.half!;
  const { x, type } = pc;
  const owner = pc.nb.owner;
  const nlocal = pc.nlocal;
  const nbrs = list.neighbors;
  const rhoOwn = new Float64Array(nlocal);
  for (let i = 0; i < list.inum; i++) {
    const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
    for (let k = k0; k < k1; k++) {
      const j = nbrs[k] & NEIGHMASK;
      const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const ti = type[i], tj = type[j];
      if (!(rsq < cutsq[ti * nt + tj])) continue;
      const rcut = Math.sqrt(cutsq[ti * nt + tj]);
      const r = Math.sqrt(rsq);
      const factor = (84 / (5 * PI * rcut * rcut * rcut)) * (1 + (3 * r) / (2 * rcut)) * (1 - r / rcut) * (1 - r / rcut) * (1 - r / rcut) * (1 - r / rcut);
      rhoOwn[owner[i]] += factor;
      rhoOwn[owner[j]] += factor;
    }
  }
  return rhoOwn;
};

export class PairMultiLucy extends Pair {
  readonly name: string = 'multi/lucy';
  /** Global virial by the force dot product (native virial_fdotr_compute). */
  virialFdotr = true;
  protected tabstyle: 'lookup' | 'linear' = 'linear';
  protected tablength = 0;
  protected tables: Table[] = [];
  protected tabindex = new Int32Array(0);
  protected set = new Uint8Array(0);

  /** The constructor's check: the atom style must carry a density (atom->rho_flag). */
  override constructorCheck(where: PairConstructorWhere): void {
    if (!atomHasFlag(where.atomStyle, 'rho')) throw new StyleError(`Pair ${this.name} command requires atom_style with density (e.g. dpd, meso)`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.tabindex = new Int32Array(nt * nt).fill(-1);
    this.set = new Uint8Array(nt * nt);
  }

  settings(args: string[]): void {
    // As native LAMMPS does (src/DPD-REACT/pair_multi_lucy.cpp PairMultiLucy::settings at b332f28)
    if (args.length !== 2) throw new StyleError('Illegal pair_style command');
    this.tableSettings(args);
  }

  /** The table style and the table length (utils::inumeric, at least 2), as multi/lucy and multi/lucy/rx read them. */
  protected tableSettings(args: string[]): void {
    if (args[0] === 'lookup') this.tabstyle = 'lookup';
    else if (args[0] === 'linear') this.tabstyle = 'linear';
    else throw new StyleError('Unknown table style in pair_style command');
    const n = inumeric(args[1]);
    if (n < 2) throw new StyleError('Illegal number of pair table entries');
    this.tablength = n;
    this.tables = [];
    if (this.ntypes > 0) this.allocate(this.ntypes);
  }

  coeff(args: string[], ctx: StyleContext): void {
    // As native LAMMPS does (src/DPD-REACT/pair_multi_lucy.cpp PairMultiLucy::coeff at b332f28): a count other than 4 or 5 is "Illegal pair_coeff command".
    if (args.length !== 4 && args.length !== 5) throw new StyleError('Illegal pair_coeff command\nusage: pair_coeff I J file keyword [cutoff]');
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the atom types first');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    // RxTableFileReader opens the file as a "<style> table"
    const tb = this.readTable(ctx.readFile(args[2], potentialFile(`${this.name} table`)), args[3], args[2]);
    if (args.length === 5) tb.cut = parseNum(args[4], 'cutoff');
    else if (tb.rflag) tb.cut = tb.rhi;
    else tb.cut = tb.rfile[tb.ninput - 1];
    if (tb.ninput <= 1) throw new StyleError('Invalid pair table length');
    tb.match = this.tabstyle === 'linear' && tb.ninput === this.tablength && tb.rflag === 2;
    if (!tb.match) this.splineTable(tb);
    this.computeTable(tb);
    const nt = this.ntypes + 1;
    const id = this.tables.length;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.tabindex[i * nt + j] = id;
        this.set[i * nt + j] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Illegal pair_coeff command (no type pairs)');
    this.tables.push(tb);
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (!this.set[i * nt + j]) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    this.tabindex[j * nt + i] = this.tabindex[i * nt + j];
    return this.tables[this.tabindex[i * nt + j]].cut;
  }

  initStyle(ctx: StyleContext): void {
    if (!ctx.newtonPair) throw new StyleError('pair multi/lucy requires newton pair on');
  }

  /** read_table of pair_multi_lucy.cpp (param_extract and the entries of the keyword section). */
  protected readTable(text: string, keyword: string, file: string): Table {
    const { params, rows } = readSection(text, keyword, file);
    const tb = newTable();
    let nFound = -1;
    for (let k = 0; k < params.length; k++) {
      if (params[k] === 'N' && nFound < 0) {
        nFound = parseNum(params[k + 1], 'N');
        k++;
      }
    }
    if (nFound < 0) throw new StyleError(`pair multi/lucy: failed to read N from the parameter line of '${keyword}' in '${file}'`);
    tb.ninput = nFound;
    // the parameter tokens other than N (one N is consumed): R|RSQ lo hi, FP fplo fphi
    let consumed = false;
    for (let k = 0; k < params.length; k++) {
      const w = params[k];
      if (w === 'N' && !consumed) { consumed = true; k++; continue; }
      if (w[0] === 'R') {
        if (w === 'R') tb.rflag = 1;
        else if (w === 'RSQ') tb.rflag = 2;
        else throw new StyleError(`Invalid keyword in pair table parameters: ${w}`);
        tb.rlo = parseNum(params[++k], 'R lo');
        tb.rhi = parseNum(params[++k], 'R hi');
      } else if (w === 'FP') {
        tb.fpflag = 1;
        tb.fplo = parseNum(params[++k], 'FP lo');
        tb.fphi = parseNum(params[++k], 'FP hi');
      } else {
        throw new StyleError(`Invalid keyword in pair table parameters: ${w}`);
      }
    }
    if (rows.length < tb.ninput) throw new StyleError(`pair multi/lucy: table '${keyword}' in '${file}' has ${rows.length} entries, N is ${tb.ninput}`);
    tb.rfile = new Float64Array(tb.ninput);
    tb.efile = new Float64Array(tb.ninput);
    tb.ffile = new Float64Array(tb.ninput);
    for (let i = 0; i < tb.ninput; i++) {
      const v = rows[i];
      let r = parseNum(v[1], 'r');
      tb.efile[i] = parseNum(v[2], 'energy');
      tb.ffile[i] = parseNum(v[3], 'force');
      if (tb.rflag === 1) r = tb.rlo + ((tb.rhi - tb.rlo) * i) / (tb.ninput - 1);
      else if (tb.rflag === 2) {
        const rsq = tb.rlo * tb.rlo + ((tb.rhi * tb.rhi - tb.rlo * tb.rlo) * i) / (tb.ninput - 1);
        r = Math.sqrt(rsq);
      }
      tb.rfile[i] = r;
    }
    return tb;
  }

  /** spline_table of pair_multi_lucy.cpp. */
  private splineTable(tb: Table): void {
    const n = tb.ninput;
    tb.e2file = new Float64Array(n);
    tb.f2file = new Float64Array(n);
    spline(tb.rfile, tb.efile, n, -tb.ffile[0], -tb.ffile[n - 1], tb.e2file);
    if (tb.fpflag === 0) {
      tb.fplo = (tb.ffile[1] - tb.ffile[0]) / (tb.rfile[1] - tb.rfile[0]);
      tb.fphi = (tb.ffile[n - 1] - tb.ffile[n - 2]) / (tb.rfile[n - 1] - tb.rfile[n - 2]);
    }
    spline(tb.rfile, tb.ffile, n, tb.fplo, tb.fphi, tb.f2file);
  }

  /** compute_table of pair_multi_lucy.cpp: the tables of E and F against r^2 (lookup) or the linear table (linear). */
  private computeTable(tb: Table): void {
    const tlm1 = this.tablength - 1;
    const inner = tb.rflag ? tb.rlo : tb.rfile[0];
    tb.innersq = inner * inner;
    tb.delta = (tb.rhi * tb.rhi - tb.innersq) / tlm1;
    tb.invdelta = 1 / tb.delta;
    if (this.tabstyle === 'lookup') {
      tb.e = new Float64Array(tlm1);
      tb.f = new Float64Array(tlm1);
      for (let i = 0; i < tlm1; i++) {
        const rsq = tb.innersq + (i + 0.5) * tb.delta;
        const r = Math.sqrt(rsq);
        tb.e[i] = splint(tb.rfile, tb.efile, tb.e2file, tb.ninput, r);
        tb.f[i] = splint(tb.rfile, tb.ffile, tb.f2file, tb.ninput, r);
      }
      return;
    }
    const n = this.tablength;
    tb.rsq = new Float64Array(n);
    tb.e = new Float64Array(n);
    tb.f = new Float64Array(n);
    tb.de = new Float64Array(tlm1);
    tb.df = new Float64Array(tlm1);
    for (let i = 0; i < n; i++) {
      const rsq = tb.innersq + i * tb.delta;
      const r = Math.sqrt(rsq);
      tb.rsq[i] = rsq;
      if (tb.match) {
        tb.e[i] = tb.efile[i];
        tb.f[i] = tb.ffile[i];
      } else {
        tb.e[i] = splint(tb.rfile, tb.efile, tb.e2file, tb.ninput, r);
        tb.f[i] = splint(tb.rfile, tb.ffile, tb.f2file, tb.ninput, r);
      }
    }
    for (let i = 0; i < tlm1; i++) {
      tb.de[i] = tb.e[i + 1] - tb.e[i];
      tb.df[i] = tb.f[i + 1] - tb.f[i];
    }
  }

  /**
   * compute of pair_multi_lucy.cpp: the local densities (computeLocalDensity: Lucy weights of the neighbors inside
   * the cutoff, summed on the owners), then the pair forces from the tables of the two densities, and the per-atom
   * energy of each owned atom from its own density (the self table of its type, times pi rc^4 / 84).
   */
  compute(pc: PairCompute): void {
    if (pc.eatom || pc.vatom) throw new StyleError('pair multi/lucy: per-atom energy and virial are not supported by the browser engine');
    const list = pc.half!;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const nbrs = list.neighbors;
    const rhoOwn = lucyLocalDensity(pc, cutsq, nt);
    const rho = (k: number) => rhoOwn[owner[k]];
    const energyAt = (i: number, itype: number): number => {
      const tb = this.tables[this.tabindex[itype * nt + itype]];
      const rho2 = rho(i) * rho(i);
      if (rho2 < tb.innersq) throw new StyleError('Density < table inner cutoff');
      const itable = Math.trunc((rho2 - tb.innersq) * tb.invdelta);
      if (this.tabstyle === 'lookup') return tb.e[itable];
      if (itable >= this.tablength - 1) throw new StyleError('Density > table outer cutoff');
      const fraction = itable === 0 ? 0 : (rho2 - tb.rsq[itable]) * tb.invdelta;
      return tb.e[itable] + fraction * tb.de[itable];
    };
    for (let i = 0; i < list.inum; i++) {
      const itype = type[i];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbrs[k] & NEIGHMASK;
        const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const jtype = type[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        const tb = this.tables[this.tabindex[itype * nt + jtype]];
        const rhoI2 = rho(i) * rho(i), rhoJ2 = rho(j) * rho(j);
        if (rhoI2 < tb.innersq || rhoJ2 < tb.innersq) throw new StyleError('Density < table inner cutoff');
        const tlm1 = this.tablength - 1;
        const rc = Math.sqrt(cutsq[itype * nt + jtype]);
        const r = Math.sqrt(rsq);
        let Ai: number, Aj: number;
        const itable = Math.trunc((rhoI2 - tb.innersq) * tb.invdelta);
        const jtable = Math.trunc((rhoJ2 - tb.innersq) * tb.invdelta);
        if (this.tabstyle === 'lookup') {
          if (itable >= tlm1 || jtable >= tlm1) throw new StyleError('Density > table outer cutoff');
          Ai = tb.f[itable];
          Aj = tb.f[jtable];
        } else {
          if (itable >= tlm1 || jtable >= tlm1) throw new StyleError('Density > table outer cutoff');
          const fi = (rhoI2 - tb.rsq[itable]) * tb.invdelta;
          const fj = (rhoJ2 - tb.rsq[jtable]) * tb.invdelta;
          Ai = tb.f[itable] + fi * tb.df[itable];
          Aj = tb.f[jtable] + fj * tb.df[jtable];
        }
        let fpair = 0.5 * (Ai + Aj) * (1 + (3 * r) / rc) * (1 - r / rc) * (1 - r / rc) * (1 - r / rc);
        fpair = fpair / r;
        f[3 * i] += dx * fpair;
        f[3 * i + 1] += dy * fpair;
        f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair;
        f[3 * j + 1] -= dy * fpair;
        f[3 * j + 2] -= dz * fpair;
      }
    }
    // energies: one per owned atom, from its own density and its self table
    let evdwl = 0;
    for (let i = 0; i < nlocal; i++) {
      const itype = type[i];
      const e = energyAt(i, itype);
      const rc2 = cutsq[itype * nt + itype];
      evdwl += e * (PI * rc2 * rc2) / 84;
    }
    pc.acc.evdwl += evdwl;
  }
}
