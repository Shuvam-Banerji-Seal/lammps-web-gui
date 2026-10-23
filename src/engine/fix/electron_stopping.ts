import type { System } from '../system';
import { Fix } from './fix';
import { StyleError, potentialFile } from '../force/types';
import { inumeric, numeric } from '../force/util';
import { massOf } from '../atoms';
import type { Region } from '../region';

/*
 * fix electron/stopping — docs.lammps.org/fix_electron_stopping.html. Written
 * from the doc page and from black-box runs of the native binary.
 *
 * Doc: "The electronic stopping is implemented by applying a friction force to
 * each atom as" F_i = F0_i - (v_i / |v_i|) S_e. The stopping power S_e comes
 * from the table: "The stopping powers for intermediate energy values are
 * calculated with linear interpolation between 2 nearest points."
 *
 * Measured with native LAMMPS (black box, one atom, metal units):
 *   - the friction is not applied in setup: the first half kick of step 1 has
 *     no friction, so the first step loses half of the later per-step change;
 *   - the energy scalar adds S_e |v| dt per step, with v the velocity at the
 *     force evaluation (after the first half kick of the step);
 *   - the kinetic energy used for the Ecut test is 0.5 m v^2 mvv2e (eV for metal);
 *   - an atom with kinetic energy below the first table energy gets no friction,
 *     and one above the last table energy stops the run with an error;
 *   - minneigh counts neighbours from the neighbour list (pair cutoff plus skin):
 *     an atom with a neighbour at 4.0 A (pair cutoff 3.0 A, skin 2.0 A) counts,
 *     one at 6.0 A does not; minneigh 0 disables the check.
 */

interface StoppingTable {
  /** Energies (energy units), ascending. */
  e: Float64Array;
  /** Stopping power per atom type (1-based index into se[t - 1]), one value per energy. */
  se: Float64Array[];
}

/** Parses the table file: '#' lines and blank lines are skipped; each data line holds N+1 numbers. */
export const parseStoppingTable = (text: string, ntypes: number, name: string): StoppingTable => {
  const rows: number[][] = [];
  const lines = text.split('\n');
  for (let ln = 0; ln < lines.length; ln++) {
    const line = lines[ln].trim();
    if (line === '' || line.startsWith('#')) continue;
    const words = line.split(/\s+/);
    if (words.length !== ntypes + 1) {
      throw new StyleError(`fix electron/stopping: line ${ln + 1} of ${name} has ${words.length} columns, expected ${ntypes + 1} (energy and one stopping power per atom type)`);
    }
    const nums = words.map(Number);
    if (!nums.every((x) => Number.isFinite(x))) throw new StyleError(`fix electron/stopping: line ${ln + 1} of ${name} has a non-numeric entry`);
    rows.push(nums);
  }
  if (rows.length < 2) throw new StyleError(`fix electron/stopping: ${name} needs at least two energy rows to interpolate`);
  for (let r = 1; r < rows.length; r++) {
    if (!(rows[r][0] > rows[r - 1][0])) {
      throw new StyleError(`fix electron/stopping: energies in ${name} must be sorted from the smallest to the largest`);
    }
  }
  const e = Float64Array.from(rows.map((r) => r[0]));
  const se: Float64Array[] = [];
  for (let t = 1; t <= ntypes; t++) se.push(Float64Array.from(rows.map((r) => r[t])));
  return { e, se };
};

/** Linear interpolation of S_e for type t (1-based) at energy E within the table range. */
const interpolate = (tab: StoppingTable, t: number, E: number): number => {
  const e = tab.e, y = tab.se[t - 1];
  const last = e.length - 1;
  let lo = 0, hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (e[mid] <= E) lo = mid;
    else hi = mid;
  }
  const f = (E - e[lo]) / (e[hi] - e[lo]);
  return y[lo] + f * (y[hi] - y[lo]);
};

export class FixElectronStopping extends Fix {
  readonly style = 'electron/stopping';
  private readonly ecut: number;
  private readonly table: StoppingTable;
  private readonly minneigh: number;
  private readonly regionId: string | null;
  private regionObj: Region | null = null;
  /** Energy removed from the projectiles since the start of the run (f_ID). */
  private loss = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/EXTRA-FIX/fix_electron_stopping.cpp FixElectronStopping::FixElectronStopping at b332f28):
    // Ecut (utils::numeric, > 0) and the file are required; region and minneigh may each be given once.
    const illegal = 'Illegal fix electron/stopping command';
    if (args.length < 2) throw new StyleError(`${illegal}: too few arguments`);
    const ecut = numeric(args[0]);
    if (!(ecut > 0)) throw new StyleError(`${illegal}: Ecut <= 0`);
    this.ecut = ecut;
    let minneigh = 1;
    let minneighGiven = false;
    let regionId: string | null = null;
    for (let k = 2; k < args.length; k += 2) {
      const key = args[k];
      const value = args[k + 1];
      if (key === 'region') {
        if (regionId !== null) throw new StyleError(`${illegal}: region given twice`);
        if (value === undefined) throw new StyleError(`${illegal}: region name missing`);
        if (!sys.regions.has(value)) throw new StyleError(`Region ${value} for fix electron/stopping does not exist`);
        regionId = value;
      } else if (key === 'minneigh') {
        if (minneighGiven) throw new StyleError(`${illegal}: minneigh given twice`);
        minneighGiven = true;
        if (value === undefined) throw new StyleError(`${illegal}: minneigh number missing`);
        minneigh = inumeric(value);
        if (minneigh < 0) throw new StyleError(`${illegal}: minneigh < 0`);
      } else {
        throw new StyleError(`${illegal}: unknown argument`);
      }
    }
    this.minneigh = minneigh;
    this.regionId = regionId;
    this.table = parseStoppingTable(sys.readFile(args[1], potentialFile('electron stopping data table')), sys.state.ntypes, args[1]);
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  /** Measured: no friction is applied in the setup force evaluation (the friction starts with the first post-force of a step). */
  setup(): void { /* intentionally empty, see the file comment */ }

  init(): void { this.loss = 0; }

  private applies(i: number): boolean {
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return false;
    if (this.regionId === null) return true;
    this.regionObj ??= this.sys.region(this.regionId);
    return this.regionObj.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
  }

  postForce(): void {
    const s = this.sys.state;
    const nb = this.sys.nb;
    const counts = this.minneigh > 0 ? nb.occasionalList(s, nb.cutneighmax, true, s.ntypes).numneigh : null;
    const eLow = Math.max(this.ecut, this.table.e[0]);
    const eHigh = this.table.e[this.table.e.length - 1];
    const dt = s.dt;
    for (let i = 0; i < s.n; i++) {
      if (!this.applies(i)) continue;
      if (counts && counts[i] < this.minneigh) continue;
      const k = 3 * i;
      const vx = s.v[k], vy = s.v[k + 1], vz = s.v[k + 2];
      const v2 = vx * vx + vy * vy + vz * vz;
      const vmag = Math.sqrt(v2);
      if (!(vmag > 0)) continue;
      const ke = 0.5 * massOf(s, i) * v2 * s.units.mvv2e;
      if (ke < eLow) continue;
      if (ke > eHigh) {
        throw new StyleError(`fix ${this.id} (electron/stopping): kinetic energy ${ke} of atom ${s.id[i]} is above the largest table energy ${eHigh}`);
      }
      const se = interpolate(this.table, s.type[i], ke);
      s.f[k] -= (se * vx) / vmag;
      s.f[k + 1] -= (se * vy) / vmag;
      s.f[k + 2] -= (se * vz) / vmag;
      this.loss += se * vmag * dt;
    }
  }

  computeScalar(): number { return this.loss; }
}
