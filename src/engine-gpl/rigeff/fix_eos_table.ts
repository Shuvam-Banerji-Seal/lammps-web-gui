/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/DPD-REACT/fix_eos_table.cpp and
 * fix_eos_table.h (class FixEOStable, fix eos/table): the equation of state of a DPD particle from a tabulated internal
 * energy against internal temperature. The table is read with the keyword reader of the reacting DPD tables
 * (RxTableFileReader, src/DPD-REACT/rx_table_file_reader.cpp), splined (Numerical Recipes, fix_eos_table.cpp spline and
 * splint), and resampled on tablength evenly spaced points; the lookups are linear. Contributing author (from the
 * source): James Larentzos (U.S. Army Research Laboratory). LAMMPS is Copyright (2003) Sandia Corporation and distributed
 * under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_eos_table.html: "fix ID group-ID eos/table linear table-file N keyword": the table file holds
 * "index T E" entries after the keyword section, and the internal energy is the uCond + uMech sum that gives dpdTheta.
 * The second, inverse table (E against T, native tables[1]) gives the temperature in post_integrate and end_of_step.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError, potentialFile } from '../../engine/force/types';
import { inumeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { DPD_THETA, DPD_UCOND, DPD_UMECH } from '../../engine/atoms';
import { RxTableReader } from '../dpdrx/rx_table_reader';

/** One table: the input (rfile, efile), its spline (e2file) and the resampled lookup (r, e, de). */
interface Table {
  ninput: number;
  lo: number;
  hi: number;
  rfile: Float64Array;
  efile: Float64Array;
  e2file: Float64Array;
  r: Float64Array;
  e: Float64Array;
  de: Float64Array;
  delta: number;
  invdelta: number;
}

/** fix_eos_table.cpp spline (Numerical Recipes), with the end derivatives yp1 and ypn (0.0 in the constructor). */
const spline = (x: Float64Array, y: Float64Array, n: number, yp1: number, ypn: number, y2: Float64Array): void => {
  const u = new Float64Array(n);
  if (yp1 > 0.99e30) {
    y2[0] = 0.0;
    u[0] = 0.0;
  } else {
    y2[0] = -0.5;
    u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
  }
  for (let i = 1; i < n - 1; i++) {
    const sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    const p = sig * y2[i - 1] + 2.0;
    y2[i] = (sig - 1.0) / p;
    u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
  }
  let qn: number, un: number;
  if (ypn > 0.99e30) {
    qn = 0.0;
    un = 0.0;
  } else {
    qn = 0.5;
    un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
  }
  y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
  for (let k = n - 2; k >= 0; k--) y2[k] = y2[k] * y2[k + 1] + u[k];
};

/** fix_eos_table.cpp splint: the spline value at x (bisection for the interval). */
const splint = (xa: Float64Array, ya: Float64Array, y2a: Float64Array, n: number, x: number): number => {
  let klo = 0;
  let khi = n - 1;
  while (khi - klo > 1) {
    const k = (khi + klo) >> 1;
    if (xa[k] > x) khi = k;
    else klo = k;
  }
  const h = xa[khi] - xa[klo];
  const a = (xa[khi] - x) / h;
  const b = (x - xa[klo]) / h;
  return a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6.0;
};

// As native LAMMPS does (src/DPD-REACT/fix_eos_table.cpp FixEOStable::FixEOStable at b332f28): utils::inumeric reads the table length.
const intArg = (w: string | undefined, _what: string): number => inumeric(w);

export class FixEOStable extends Fix {
  readonly style = 'eos/table';
  private readonly tablength: number;
  /** tables[0]: temperature -> energy (energy_lookup); tables[1]: energy -> temperature (temperature_lookup) */
  private readonly tables: Table[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_eos_table.cpp: narg must be 7, i.e. style linear, table file, N, keyword
    if (args.length !== 4) throw new StyleError('Illegal fix eos/table command');
    this.nevery = 1;
    if (args[0] !== 'linear') throw new StyleError('Unknown table style in fix eos/table');
    this.tablength = intArg(args[2], 'fix eos/table');
    if (this.tablength < 2) throw new StyleError('Illegal number of eos/table entries');
    const file = args[1];
    const reader = new RxTableReader(sys.readFile(file, potentialFile('eos/table table')), args[3], file, 'eos/table');
    const N = reader.N;
    const mk = (): Table => ({
      ninput: N, lo: 0, hi: 0, rfile: new Float64Array(N), efile: new Float64Array(N), e2file: new Float64Array(N),
      r: new Float64Array(this.tablength), e: new Float64Array(this.tablength), de: new Float64Array(this.tablength - 1),
      delta: 0, invdelta: 0,
    });
    const tb = mk(), tb2 = mk();
    // read_table: "index T E" per entry; tb maps T to E, tb2 maps E to T
    reader.readTable((i, values) => {
      values.int();
      tb.rfile[i] = tb2.efile[i] = values.double();
      tb.efile[i] = tb2.rfile[i] = values.double();
    });
    if (tb.ninput <= 1) throw new StyleError('Invalid eos/table length');
    tb.lo = tb.rfile[0];
    tb.hi = tb.rfile[tb.ninput - 1];
    if (tb.lo >= tb.hi) throw new StyleError('eos/table values are not increasing');
    if (tb2.ninput <= 1) throw new StyleError('Invalid eos/table length');
    tb2.lo = tb2.rfile[0];
    tb2.hi = tb2.rfile[tb2.ninput - 1];
    if (tb2.lo >= tb2.hi) throw new StyleError('eos/table values are not increasing');
    for (const t of [tb, tb2]) {
      spline(t.rfile, t.efile, t.ninput, 0.0, 0.0, t.e2file);
      this.computeTable(t);
    }
    this.tables.push(tb, tb2);
    if (!sys.state.custom.has(DPD_THETA)) throw new StyleError('FixEOStable requires atom_style with internal temperature and energies (e.g. dpd)');
  }

  /** compute_table: tablength evenly spaced points from lo to hi, e at the lower edge of each bin, de the bin differences. */
  private computeTable(tb: Table): void {
    const tlm1 = this.tablength - 1;
    tb.delta = (tb.hi - tb.lo) / tlm1;
    tb.invdelta = 1.0 / tb.delta;
    for (let i = 0; i < this.tablength; i++) {
      const a = tb.lo + i * tb.delta;
      tb.r[i] = a;
      tb.e[i] = splint(tb.rfile, tb.efile, tb.e2file, tb.ninput, a);
    }
    for (let i = 0; i < tlm1; i++) tb.de[i] = tb.e[i + 1] - tb.e[i];
  }

  /** energy_lookup: the internal energy at temperature t (linear in the table bin). */
  private energyLookup(t: number): number {
    const tb = this.tables[0];
    if (t < tb.lo || t > tb.hi) throw new StyleError(`Temperature ${t} is not within table cutoffs (${tb.lo}, ${tb.hi})`);
    const itable = Math.trunc((t - tb.lo) * tb.invdelta);
    const fraction = (t - tb.r[itable]) * tb.invdelta;
    return tb.e[itable] + fraction * tb.de[itable];
  }

  /** temperature_lookup: the internal temperature at energy u (linear in the table bin). */
  private temperatureLookup(u: number): number {
    const tb = this.tables[1];
    if (u < tb.lo || u > tb.hi) throw new StyleError(`Energy ${u} is not within table cutoffs (${tb.lo}, ${tb.hi})`);
    const itable = Math.trunc((u - tb.lo) * tb.invdelta);
    const fraction = (u - tb.r[itable]) * tb.invdelta;
    return tb.e[itable] + fraction * tb.de[itable];
  }

  /** fix_eos_table.cpp init (no restart): uMech = energy of the internal temperature, uCond = 0. */
  init(): void {
    const s = this.sys.state;
    const th = s.custom.get(DPD_THETA)!.data, uc = s.custom.get(DPD_UCOND)!.data, um = s.custom.get(DPD_UMECH)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (th[i] <= 0.0) throw new StyleError('Internal temperature <= zero');
      const tmp = this.energyLookup(th[i]);
      uc[i] = 0.0;
      um[i] = tmp;
    }
  }

  /** post_integrate and end_of_step: the internal temperature from uCond + uMech. */
  private update(): void {
    const s = this.sys.state;
    const th = s.custom.get(DPD_THETA)!.data, uc = s.custom.get(DPD_UCOND)!.data, um = s.custom.get(DPD_UMECH)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      th[i] = this.temperatureLookup(uc[i] + um[i]);
      if (th[i] <= 0.0) throw new StyleError('Internal temperature <= zero');
    }
  }

  postIntegrate(): void { this.update(); }
  endOfStep(): void { this.update(); }
}
