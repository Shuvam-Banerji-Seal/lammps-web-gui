import { StyleError } from '../force/types';
import type { System } from '../system';
import { FixTTM, type TtmKeywordHandler } from './ttm';

/*
 * fix ttm/cascade — the two-temperature model for radiation damage cascades (docs.lammps.org/fix_ttm.html,
 * section "Additional details for fix ttm/cascade"; plans/lammps-docs-live/fix_ttm.rst). It is fix ttm
 * (fix_ttm.ts, the base class) with three changes: fast atoms are cut off from the electron-ion friction
 * (keyword cutoff), electron-ion friction is switched off for a time (keyword offset), and the electron
 * specific heat and conductivity depend on the electron temperature through tables (keywords cetab and
 * ketab). There is no outfile keyword (the doc: "The fix ttm/grid and fix ttm/cascade commands do not
 * support the *outfile* keyword."); the grid is written with a restart file, as for ttm/grid.
 *
 * Syntax: fix ID group-ID ttm/cascade seed C_e rho_e kappa_e gamma_p gamma_s v_0 Nx Ny Nz keyword value ...
 * with keywords set, infile, cutoff, offset, cetab, ketab.
 *
 * Quoted doc sentences (docs.lammps.org/fix_ttm.html):
 *  "The *cutoff* keyword disables electron-ion interactions for fast-moving particles with velocities above the electronic stopping critical velocity, :math:`v_0`."
 *  "The *offset* keyword specifies an offset time (thermalization time) value during which electron-ion interactions are disabled for all particles."
 *  "The temperature ranges and properties are read from the table file."
 *  "The temperatures must be sorted from the lowest to the highest."
 *  "If a grid point would have an electronic temperature below or above the lowest or highest temperature given in the table file, the interpolation will employ the lowest or highest value, respectively."
 *  "When the cetab or ketab keywords are used, the specified fixed parameters are overwritten."
 *  "The electronic temperature at each grid point must be a non-zero positive value, both initially, and as the temperature evolves over time."
 *
 * Model (doc): the friction of an atom with speed v_i is gamma_p + gamma_s above v_0 and gamma_p below;
 * with cutoff, gamma_p is 0 above v_0; with offset, gamma_p is 0 for the time before the offset.
 *
 * Measured with native LAMMPS (black box, eight-atom 2x2x2 runs with a grid restart file written to read
 * the grid, and thermo f_2):
 *  - with no cetab/ketab, no cutoff and no offset the fix is identical to fix ttm (same forces, same
 *    energies, to the last bit, for gamma_s > 0 and speeds on both sides of v_0);
 *  - cutoff: an atom above v_0 gets only the stopping force -gamma_s v (no random part); an atom below v_0
 *    keeps its friction and random force; all atoms still draw their random numbers (the slow atoms'
 *    forces equal those of fix ttm in the same run);
 *  - at the speed v = v_0 exactly the atom is not "above" (the cutoff does not apply: the forces equal fix ttm);
 *  - offset: during the steps whose time is below the offset (time = step * dt) there is no friction and
 *    no random force, and the stopping force still acts on atoms above v_0; the offset 0.02 gives a
 *    first active step at time 0.02 (the comparison is strict); offset 0.015 and 0.0201 switch on
 *    between the same step pairs as the strict rule predicts;
 *  - cetab: the electron energy is sum over cells of V rho_e integral(C_e(T) dT) from 0 to T_e (the
 *    integral taken with C_e flat below the first table temperature and flat above the last): f_2[1]
 *    equals this to 1e-15 relative; the table is linear between nodes and flat outside the table;
 *  - the diffusion of ttm/cascade is div(kappa grad T) = kappa_c lap(T) + grad(kappa) . grad(T), with
 *    central differences of kappa(T) (not the pairwise harmonic flow of ttm/thermal): on the 4x4x4 grid of
 *    tests/oracle/w41ttm_cascade_ex the grid after one step matches to 1e-9 with this form, while the
 *    pairwise forms differ by 0.8 K in the neighbours of a hot cell; on a 2x2x2 grid the gradient term
 *    vanishes (both neighbours are one cell), where kappa_c lap(T) matches to 5e-13 and the harmonic mean
 *    differs by 1.3 K;
 *  - the diffusion step divides by C_e(T_c) rho_e of the cell at the start of each sub-step; the
 *    sub-step count uses the largest kappa_e(T_c) / C_e(T_c) over the cells, with the table values (dt 3
 *    gives 2 sub-steps and dt 6 gives 4, both matching);
 *  - the transfer to the atoms is removed as T_e -= transfer / (V rho_e C_e(T_e)) per sub-step (matches to 1e-12);
 *  - the positional C_e and kappa_e are overwritten by the tables when the tables are given; with only
 *    cetab, kappa_e is the positional value (and the reverse with only ketab);
 *  - errors: a table with fewer than two rows, with a temperature not strictly increasing, with a
 *    C_e <= 0 or (ketab) kappa_e <= 0 entry, a missing table file, a negative offset, gamma_p = 0,
 *    gamma_s < 0, the cutoff keyword with a value (it is then an unknown keyword), and the outfile keyword;
 *  - with no set and no infile the native run starts from an all-zero grid and gives no error, but the
 *    doc requires a non-zero positive temperature at every grid point; ttm/cascade refuses that input,
 *    and a zero temperature in infile, as the doc requires.
 */

interface Table {
  /** Temperatures, strictly increasing. */
  t: number[];
  /** Property values at the temperatures. */
  v: number[];
  /** Integral of the property from 0 to t[i] (used by the electron energy of C_e). */
  cum: number[];
}

/** Table file: "T value" per line, '#' comments and empty lines ignored. */
export const parseCascadeTable = (text: string, id: string, what: string, fname: string): Table => {
  const t: number[] = [], v: number[] = [];
  for (const raw of text.split('\n')) {
    const line = raw.replace(/#.*/, '').trim();
    if (!line) continue;
    const w = line.split(/\s+/);
    const a = Number(w[0]), b = Number(w[1]);
    if (w.length < 2 || !Number.isFinite(a) || !Number.isFinite(b)) {
      throw new StyleError(`fix ${id} ttm/cascade: ${what} table ${fname} needs a temperature and a value per line: '${raw.trim()}'`);
    }
    t.push(a); v.push(b);
  }
  if (t.length < 2) throw new StyleError(`fix ${id} ttm/cascade: the ${what} table ${fname} has less than 2 rows, interpolation is invalid`);
  for (let i = 1; i < t.length; i++) {
    if (!(t[i] > t[i - 1])) throw new StyleError(`fix ${id} ttm/cascade: the temperatures in ${what} table ${fname} must be strictly increasing`);
  }
  for (const x of v) {
    if (!(x > 0)) throw new StyleError(`fix ${id} ttm/cascade: the ${what} values in ${fname} must be > 0`);
  }
  const cum: number[] = [v[0] * t[0]];
  for (let i = 1; i < t.length; i++) cum.push(cum[i - 1] + ((v[i - 1] + v[i]) / 2) * (t[i] - t[i - 1]));
  return { t, v, cum };
};

/** Linear interpolation in the table, flat beyond its ends. */
export const interpTable = (tab: Table, T: number): number => {
  const { t, v } = tab;
  if (T <= t[0]) return v[0];
  if (T >= t[t.length - 1]) return v[v.length - 1];
  let i = 0;
  while (T > t[i + 1]) i++;
  return v[i] + ((v[i + 1] - v[i]) * (T - t[i])) / (t[i + 1] - t[i]);
};

/** Integral of the table property from 0 to T (flat below the first and above the last temperature). */
export const integralTable = (tab: Table, T: number): number => {
  const { t, v } = tab;
  let E = v[0] * Math.min(T, t[0]);
  if (T <= t[0]) return E;
  for (let i = 0; i < t.length - 1; i++) {
    const a = t[i], b = t[i + 1];
    if (T <= a) break;
    const x = Math.min(T, b);
    const cx = v[i] + ((v[i + 1] - v[i]) * (x - a)) / (b - a);
    E += ((v[i] + cx) / 2) * (x - a);
  }
  if (T > t[t.length - 1]) E += v[v.length - 1] * (T - t[t.length - 1]);
  return E;
};

/** fix ttm/cascade: cutoff and offset friction rules, and temperature-dependent C_e and kappa_e tables. */
export class FixTTMCascade extends FixTTM {
  private cutoff!: boolean;
  private offset!: number;
  private ceTab!: Table | null;
  private keTab!: Table | null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    const data = { cutoff: false, offset: 0, cetab: null as string | null, ketab: null as string | null };
    const handler: TtmKeywordHandler = (key, a, i) => {
      if (key === 'cutoff') {
        data.cutoff = true;
        return 1;
      }
      if (key === 'offset') {
        const v = Number(a[i + 1]);
        if (a[i + 1] === undefined || !Number.isFinite(v)) throw new StyleError(`fix ${id} ttm/cascade: keyword offset needs a time`);
        // Measured with native LAMMPS (black box): a negative offset stops the run with an error.
        if (v < 0) throw new StyleError(`fix ${id} ttm/cascade: time offset must be >= 0`);
        data.offset = v;
        return 2;
      }
      if (key === 'cetab' || key === 'ketab') {
        if (a[i + 1] === undefined) throw new StyleError(`fix ${id} ttm/cascade: keyword ${key} needs a table file`);
        data[key] = a[i + 1];
        return 2;
      }
      return 0;
    };
    super(sys, id, group, args, 'ttm/cascade', undefined, handler);
    this.cutoff = data.cutoff;
    this.offset = data.offset;
    this.ceTab = data.cetab !== null ? parseCascadeTable(sys.readFile(data.cetab), id, 'C_e', data.cetab) : null;
    this.keTab = data.ketab !== null ? parseCascadeTable(sys.readFile(data.ketab), id, 'kappa_e', data.ketab) : null;
    if (this.kw.setT === null && this.kw.infile === null) {
      throw new StyleError(`fix ${id} ttm/cascade needs the set or infile keyword: the electron temperature at each grid point must be a non-zero positive value (docs.lammps.org/fix_ttm.html)`);
    }
    for (let c = 0; c < this.Te.length; c++) {
      if (!(this.Te[c] > 0)) throw new StyleError(`fix ${id} ttm/cascade: the electron temperature of grid cell ${c + 1} must be a non-zero positive value`);
    }
  }

  protected cvolAt(_c: number, T: number): number {
    const ce = this.ceTab !== null ? interpTable(this.ceTab, T) : this.uniform().Ce;
    return ce * this.uniform().rhoE;
  }

  protected energyAt(c: number, T: number): number {
    if (this.ceTab === null) return super.energyAt(c, T);
    return this.cellVolume * this.uniform().rhoE * integralTable(this.ceTab, T);
  }

  protected kappaAt(_c: number, T: number): number {
    return this.keTab !== null ? interpTable(this.keTab, T) : this.uniform().kappa;
  }

  /**
   * The diffusion term of ttm/cascade is the form div(kappa grad T) = kappa lap(T) + grad(kappa) . grad(T),
   * with kappa at the cell (kappa_c) and central differences of kappa(T) and T over the neighbours.
   * Measured with native LAMMPS (black box, 4x4x4 grid with the table kappa_e, one hot cell): the grid after
   * one step matches to 1e-9 with this form; the pairwise harmonic or arithmetic means differ by 0.8 K in the
   * neighbours of the hot cell (the difference vanishes on a 2x2x2 grid, where both neighbours are one cell).
   */
  protected flowInto(ix: number, iy: number, iz: number, t: number, kc: number): number {
    const { nx, ny, nz } = this.p;
    const s = this.sys.state;
    const d = [(s.box.hi[0] - s.box.lo[0]) / nx, (s.box.hi[1] - s.box.lo[1]) / ny, (s.box.hi[2] - s.box.lo[2]) / nz];
    const at = [ix, iy, iz];
    let lap = 0;
    let grad = 0;
    for (let a = 0; a < 3; a++) {
      const plus = at.slice(); plus[a] += 1;
      const minus = at.slice(); minus[a] -= 1;
      const jp = this.wrapIndex(plus[0], plus[1], plus[2]);
      const jm = this.wrapIndex(minus[0], minus[1], minus[2]);
      const Tp = this.Te[jp], Tm = this.Te[jm];
      const Kp = this.kappaAt(jp, Tp), Km = this.kappaAt(jm, Tm);
      lap += (Tp + Tm - 2 * t) / (d[a] * d[a]);
      grad += ((Kp - Km) / (2 * d[a])) * ((Tp - Tm) / (2 * d[a]));
    }
    return kc * lap + grad;
  }

  /**
   * Electron-ion friction of one atom: zero above v_0 with cutoff, and zero for all atoms before the offset
   * time (time = step * dt at the force evaluation, the offset compared strictly).
   */
  protected gammaPFor(_c: number, speed: number): number {
    const s = this.sys.state;
    const u = this.uniform();
    if (this.cutoff && speed > u.v0) return 0;
    if (this.offset > 0 && s.step * s.dt < this.offset) return 0;
    return u.gammaP;
  }
}
