/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-REACT/fix_eos_table_rx.cpp and
 * fix_eos_table_rx.h (FixEOStableRX: fix eos/table/rx, the equation of state of a reacting DPD particle from tabulated
 * species energies), and src/DPD-REACT/fix_eos_table.cpp (the linear table lookups and the Numerical Recipes spline the
 * table is built with). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing author (from both files): James
 * Larentzos (U.S. Army Research Laboratory).
 *
 * docs.lammps.org/fix_eos_table_rx.html: "fix ID group-ID eos/table/rx linear table-file N keyword thermo-file" (with the
 * optional energy and temperature corrections). The table holds, for each species, the internal energy against the
 * internal temperature; the particle energy is the sum over its species of the molecule count times the species energy
 * (heat of formation, corrections, the molecule correction and the -(N + 1.5) kB T term), and the internal temperature
 * is found from the total internal energy by the secant method. The energy changes by reactions are carried in uChem:
 * the pair exp6/rx writes the old-concentration energy (uCG) and the new one (uCGnew), and their difference is added at
 * each end of step (native: duChem = uCG - uCGnew).
 *
 * Native's second table (the temperature-energy inverse, tables2) is only built and checked, never looked up, so it is
 * only checked here. The optional energy and temperature corrections of the command apply to species 1 only, as native
 * does (fix_eos_table_rx.cpp constructor: energyCorr[0], tempCorrCoeff[0]).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { inumeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import type { SimState } from '../../engine/types';
import { DPD_THETA, DPD_UCOND, DPD_UMECH, DPD_UCHEM, DPD_UCG, DPD_UCGNEW } from '../../engine/atoms';
import { FixRX, MY_EPSILON } from './fix_rx';
import { RxTableReader, TextLines, Tokens } from './rx_table_reader';

/** One tabulated species: the input table (rfile, efile), its spline (e2file) and the lookup table (r, e, de). */
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

/** Numerical Recipes spline with the end derivatives yp1 and ypn (fix_eos_table.cpp spline). */
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

/** fix_eos_table.cpp splint: the spline value at x. */
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

export class FixEOStableRX extends Fix {
  readonly style = 'eos/table/rx';
  private readonly rx: FixRX | null;
  private readonly nspecies: number;
  private readonly tablength: number;
  private readonly tables: Table[] = [];
  private readonly dHf: Float64Array;
  private readonly energyCorr: Float64Array;
  private readonly tempCorrCoeff: Float64Array;
  private readonly moleculeCorrCoeff: Float64Array;
  /** Species data of the state the current lookup uses (rx species vectors; null without fix rx). */
  private data: Float64Array[] | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // docs.lammps.org/fix_eos_table_rx.html: linear, table file, N, keyword, thermo file, and the two optional corrections
    if (args.length !== 5 && args.length !== 7) throw new StyleError('Illegal fix eos/table/rx command');
    this.nevery = 1;
    const rxs = sys.fixes.filter((f): f is FixRX => f.style === 'rx');
    if (rxs.length > 1) throw new StyleError('More than one fix rx instance defined');
    this.rx = rxs.length === 1 ? rxs[0] : null;
    this.nspecies = this.rx ? this.rx.nspecies : 1;
    if (args[0] !== 'linear') throw new StyleError('Unknown table style in fix eos/table/rx');
    this.tablength = intArg(args[2]);
    if (this.tablength < 2) throw new StyleError('Illegal number of eos/table/rx entries');
    const tableFile = args[1];
    const reader = new RxTableReader(sys.readFile(tableFile), args[3], tableFile, 'eos/table/rx');
    const N = reader.N;
    // param_extract: the species named on the keyword line, one per column (or species 0 without fix rx)
    const eosSpecies: number[] = [];
    if (this.rx) {
      for (const word of reader.params) {
        const k = this.rx.speciesIndexOf(word);
        if (k === undefined) throw new StyleError(`name=${word} not found in species list\nInvalid keyword in fix eos/table/rx parameters`);
        eosSpecies.push(k);
      }
      if (eosSpecies.length !== this.nspecies) {
        throw new StyleError(`ncolumns=${eosSpecies.length} nspecies=${this.nspecies}\nThe number of columns in fix eos/table/rx does not match the number of species`);
      }
    } else {
      eosSpecies.push(0);
    }
    for (let k = 0; k < this.nspecies; k++) {
      this.tables.push({
        ninput: N, lo: 0, hi: 0, rfile: new Float64Array(N), efile: new Float64Array(N), e2file: new Float64Array(N),
        r: new Float64Array(this.tablength), e: new Float64Array(this.tablength), de: new Float64Array(this.tablength - 1),
        delta: 0, invdelta: 0,
      });
    }
    // read_table: row i is "index T E_1 ... E_ncol"; the energies go to the species of their columns
    reader.readTable((i, values) => {
      values.int();
      const rtmp = values.double();
      const vals: number[] = [];
      while (values.hasNext()) {
        if (vals.length >= eosSpecies.length) {
          throw new StyleError(`Illegal fix eos/table/rx command: In file ${tableFile}, number of columns exceeds the number of species ${this.nspecies}`);
        }
        vals.push(values.double());
      }
      if (vals.length !== eosSpecies.length) throw new StyleError(`Missing energy in row ${i + 1} of table ${tableFile}`);
      for (let c = 0; c < vals.length; c++) {
        const tb = this.tables[eosSpecies[c]];
        tb.rfile[i] = rtmp;
        tb.efile[i] = vals[c];
      }
    });
    // bcast_table / the checks of the constructor; the second (inverse) table is checked but not looked up
    for (const tb of this.tables) {
      if (tb.ninput <= 1) throw new StyleError('Invalid eos/table/rx length');
      tb.lo = tb.rfile[0];
      tb.hi = tb.rfile[tb.ninput - 1];
      if (tb.lo >= tb.hi) throw new StyleError('eos/table/rx values are not increasing');
      const elo = tb.efile[0], ehi = tb.efile[tb.ninput - 1];
      if (elo >= ehi) throw new StyleError('eos/table/rx values are not increasing');
      this.buildTable(tb);
    }
    this.dHf = new Float64Array(this.nspecies);
    this.energyCorr = new Float64Array(this.nspecies);
    this.tempCorrCoeff = new Float64Array(this.nspecies);
    this.moleculeCorrCoeff = new Float64Array(this.nspecies);
    if (this.rx) this.readThermo(sys.readFile(args[4]), args[4]);
    else this.dHf[0] = numArg(args[4]);
    if (args.length === 7) {
      this.energyCorr[0] = numArg(args[5]);
      this.tempCorrCoeff[0] = numArg(args[6]);
    }
    if (!sys.state.custom.has(DPD_THETA)) throw new StyleError('FixEOStableRX requires atom_style with internal temperature and energies (e.g. dpd)');
  }

  /** fix_eos_table_rx.cpp read_file: "species dHf [energyCorr tempCorr moleculeCorr]" for the species of the rx fix. */
  private readThermo(text: string, file: string): void {
    const lines = new TextLines(text, file);
    for (let words = lines.next(2); words; words = lines.next(2)) {
      const tok = new Tokens(words, words.join(' '));
      const species = tok.next();
      const k = this.rx!.speciesIndexOf(species);
      if (k === undefined) continue;
      this.dHf[k] = tok.double();
      if (tok.hasNext()) {
        this.energyCorr[k] = tok.double();
        this.tempCorrCoeff[k] = tok.double();
        this.moleculeCorrCoeff[k] = tok.double();
      }
      if (tok.hasNext()) throw new StyleError(`Misplaced characters at end of line in file ${file}. Full line: ${words.join(' ')}`);
    }
  }

  /** fix_eos_table_rx.cpp compute_table with fix_eos_table.cpp spline_table: the spline and the lookup grid. */
  private buildTable(tb: Table): void {
    spline(tb.rfile, tb.efile, tb.ninput, 0.0, 0.0, tb.e2file);
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

  /** Sets the species data of the current state for the lookups. */
  private bind(s: SimState): void {
    this.data = this.rx ? this.rx.speciesData(s) : null;
  }

  /** The number of molecules of species k of atom id (1 without fix rx). */
  private molecules(k: number, id: number): number {
    return this.data ? this.data[k][id] : 1.0;
  }

  /** fix_eos_table_rx.cpp energy_lookup: the internal energy of atom id at internal temperature thetai. */
  private energyLookup(id: number, thetaIn: number): number {
    const tolerance = 1.0e-10;
    let nTotal = 0.0, nTotalPG = 0.0, nPG = 0;
    if (this.data) {
      for (let k = 0; k < this.nspecies; k++) {
        nTotal += this.data[k][id];
        if (Math.abs(this.moleculeCorrCoeff[k]) > tolerance) {
          nPG++;
          nTotalPG += this.data[k][id];
        }
      }
    } else {
      nTotal = 1.0;
    }
    let ui = 0.0;
    let thetai = thetaIn;
    for (let k = 0; k < this.nspecies; k++) {
      const tb = this.tables[k];
      thetai = Math.max(thetai, tb.lo);
      thetai = Math.min(thetai, tb.hi);
      const itable = Math.trunc((thetai - tb.lo) * tb.invdelta);
      const fraction = (thetai - tb.r[itable]) * tb.invdelta;
      let uTmp = tb.e[itable] + fraction * (tb.de[itable] ?? 0.0);
      uTmp += this.dHf[k];
      uTmp += this.tempCorrCoeff[k] * thetai;
      uTmp += this.energyCorr[k];
      if (nPG > 0) ui += this.moleculeCorrCoeff[k] * nTotalPG / nPG;
      ui += this.molecules(k, id) * uTmp;
    }
    return ui - (nTotal + 1.5) * this.sys.state.units.boltz * thetai;
  }

  /** fix_eos_table_rx.cpp temperature_lookup: the internal temperature at internal energy ui (secant method). */
  private temperatureLookup(id: number, ui: number, thetaIn: number): number {
    const tb = this.tables[0];
    const maxit = 100;
    let delta = 0.001;
    const tolerance = 1.0e-10;
    let t1 = Math.max(thetaIn, tb.lo);
    t1 = Math.min(t1, tb.hi);
    if (t1 === tb.hi) delta = -delta;
    let f1 = this.energyLookup(id, t1) - ui;
    let t2 = (1.0 + delta) * t1;
    let f2 = this.energyLookup(id, t2) - ui;
    let temp = 0.0;
    let it = 0;
    for (; it < maxit; it++) {
      if (Math.abs(f2 - f1) < MY_EPSILON) {
        if (Number.isNaN(f1) || Number.isNaN(f2)) throw new StyleError('NaN detected in secant solver.');
        temp = t1;
        temp = Math.max(temp, tb.lo);
        temp = Math.min(temp, tb.hi);
        this.sys.warn(`Secant solver did not converge because table bounds were exceeded: it=${it} id=${id} ui=${ui} thetai=${thetaIn} t1=${t1} t2=${t2} f1=${f1} f2=${f2} dpdTheta=${temp}`);
        break;
      }
      temp = t2 - f2 * (t2 - t1) / (f2 - f1);
      if (Math.abs(temp - t2) < tolerance) break;
      f1 = f2;
      t1 = t2;
      t2 = temp;
      f2 = this.energyLookup(id, t2) - ui;
    }
    if (it === maxit) throw new StyleError(`Maxit exceeded in secant solver: id=${id} ui=${ui} thetai=${thetaIn} t1=${t1} t2=${t2} f1=${f1} f2=${f2}`);
    if (Number.isNaN(f1) || Number.isNaN(f2) || Number.isNaN(ui) || Number.isNaN(thetaIn) || Number.isNaN(t1) || Number.isNaN(t2)) {
      throw new StyleError('NaN detected in secant solver');
    }
    return temp;
  }

  /** The internal temperature of an atom from its energies (uCond + uMech + uChem). */
  private temperatureOf(s: SimState, i: number): void {
    const uc = s.custom.get(DPD_UCOND)!.data, um = s.custom.get(DPD_UMECH)!.data, uh = s.custom.get(DPD_UCHEM)!.data;
    const th = s.custom.get(DPD_THETA)!.data;
    th[i] = this.temperatureLookup(i, uc[i] + um[i] + uh[i], th[i]);
  }

  /** fix_eos_table_rx.cpp end_of_step / setup: uChem gains the reaction energy change (uCG - uCGnew), which is reset. */
  private moveReactionEnergy(s: SimState): void {
    const uh = s.custom.get(DPD_UCHEM)!.data, ucg = s.custom.get(DPD_UCG)!.data, ucn = s.custom.get(DPD_UCGNEW)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const duChem = ucg[i] - ucn[i];
      uh[i] += duChem;
      ucg[i] = 0.0;
      ucn[i] = 0.0;
    }
  }

  /** fix_eos_table_rx.cpp init (no restart): uCond = 0, uMech = U(dpdTheta), uChem = 0. */
  init(): void {
    const s = this.sys.state;
    this.bind(s);
    const th = s.custom.get(DPD_THETA)!.data;
    const uc = s.custom.get(DPD_UCOND)!.data, um = s.custom.get(DPD_UMECH)!.data, uh = s.custom.get(DPD_UCHEM)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (!(th[i] > 0.0)) throw new StyleError('Internal temperature <= zero');
      uc[i] = 0.0;
      um[i] = this.energyLookup(i, th[i]);
      uh[i] = 0.0;
    }
  }

  /** fix_eos_table_rx.cpp setup: the energies of the setup force evaluation enter uChem; then the temperatures. */
  setup(): void {
    const s = this.sys.state;
    this.bind(s);
    this.moveReactionEnergy(s);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.temperatureOf(s, i);
  }

  postIntegrate(): void {
    this.updateTemperatures();
  }

  endOfStep(): void {
    const s = this.sys.state;
    this.bind(s);
    this.moveReactionEnergy(s);
    this.updateTemperatures();
  }

  private updateTemperatures(): void {
    const s = this.sys.state;
    this.bind(s);
    const th = s.custom.get(DPD_THETA)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.temperatureOf(s, i);
      if (!(th[i] > 0.0)) throw new StyleError('Internal temperature <= zero');
    }
  }
}

/** A keyword-argument integer (utils::inumeric). */
// As native LAMMPS does (src/DPD-REACT/fix_eos_table_rx.cpp at b332f28): utils::inumeric reads the integers.
const intArg = (w: string): number => inumeric(w);

/** A keyword-argument number (utils::numeric). */
const numArg = (w: string): number => {
  const v = Number(w);
  if (w === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  return v;
};
