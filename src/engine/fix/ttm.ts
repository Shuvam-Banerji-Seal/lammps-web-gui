import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { RanMars } from '../rng';
import { nativeOrder } from '../atoms';

/*
 * fix ttm and fix ttm/grid — the two-temperature model. Written from the
 * documented behaviour (docs.lammps.org/fix_ttm.html, plans/lammps-docs/
 * fix_ttm.rst) and from black-box measurements with native LAMMPS (the
 * binary run on synthetic inputs). No LAMMPS source was consulted.
 *
 * Syntax: "fix ID group-ID ttm seed C_e rho_e kappa_e gamma_p gamma_s v_0 Nx
 * Ny Nz keyword value ..." ; ttm/grid takes the same arguments. Keywords:
 * set Tinit, infile file, outfile Nout file (ttm only).
 *
 * Model. The electron grid is an Nx x Ny x Nz cell array over the box. Each
 * end of step the grid is advanced by the heat diffusion equation of the doc's
 * Description (C_e rho_e dT_e/dt = div(kappa_e grad T_e) - g_p (T_e - T_a) +
 * g_s T_a', with the friction and stopping terms below), and the coupling to the atoms is an inhomogeneous Langevin force per atom
 * with the electron temperature of the atom's cell:
 *   F_i = -gamma_p v_i [mvv2e] + g2 (u - 1/2) + F_stop
 *   F_stop = -gamma_s v_i [mvv2e] when |v_i| > v_0 (electronic stopping), else 0
 *   g2 = sqrt(24 gamma_p kB T_e / (mvv2e dt)) / ftm2v, u uniform on [0,1)
 * with v_i the half-step velocity at the force evaluation.
 *
 * Measured with native LAMMPS (black box, one-atom and eight-atom runs with
 * thermo and per-step dumps, seed 4242):
 *  - the first RanMars draw is discarded at fix creation, then three draws per
 *    group atom in native storage order (x, y, z), exactly as fix langevin does;
 *  - the random and friction forces above reproduce the native forces to 1e-10
 *    relative (uniform -0.5 scaling, no mass dependence of the random amplitude);
 *  - there is no fix force at setup (step 0 forces are zero);
 *  - the stopping force switches on at |v| > v_0 and is -gamma_s v (measured
 *    for speeds 1.6 .. 5.2 with v_0 = 2.2);
 *  - the energy handed from electrons to atoms on a step is
 *    f_2[2] = sum_i F_i . v_i(end of step) dt, and the electron energy
 *    E = C_e rho_e V_cell T_e (summed over cells) falls by that amount;
 *    f_2[1] = total electron energy, f_2[2] = that step's transfer;
 *  - the diffusion step is explicit FTCS on the cell grid with periodic wrap;
 *    the outfile written at step n is the grid after the end-of-step update;
 *  - with a step too large for explicit FTCS native splits the step into k
 *    equal sub-steps; with C = dt kappa/(C_e rho_e) sum_d 1/(L_d/N_d)^2 the
 *    measured count is k = 1 for C <= 1/2 and k = floor(2C) + 1 above (checked
 *    at C = 0.25 .. 3.0 on two box shapes);
 *  - gamma_p = 0 is an error in native LAMMPS (so gamma_p must be > 0);
 *  - an infile with a UNITS: tag that differs from the run's units is an
 *    error; a file without the tag is read; a duplicate or missing grid index
 *    is an error; a negative temperature is an error; the set keyword with a
 *    value of 0 is an error; with both set and infile the infile values are kept;
 *  - outfile writes the file <name>.<step> every Nout steps (step 0 is not
 *    written): a header line # DATE: ... UNITS: <units> COMMENT: Electron
 *    temperature on NxxNyxNz grid at step <step> - created by fix ttm, then one
 *    line ix iy iz T per grid point.
 *
 * Quoted doc sentences (docs.lammps.org/fix_ttm.html):
 *  "The input file is a text file which may have comments starting with the '#' character."
 *  "The lines can appear in any order."
 *  "If all the grid point values are not specified, LAMMPS will generate an error."
 *  "The fix ttm/grid command does not support the *outfile* keyword."
 *  "These fixes require use of an orthogonal 3d simulation box with periodic boundary conditions in all dimensions."
 *  "They also require that the size and shape of the simulation box do not vary"
 *  "The vector values calculated are \"extensive\"."
 *  "The first quantity is the total energy of the electronic subsystem."
 *  "The second quantity is the energy transferred from the electronic to the atomic subsystem on that timestep."
 */

/** %.16g, as the outfile values are written (no exponent for the temperature range used here). */
const fmt16g = (x: number): string => {
  if (x === 0) return '0';
  const s = x.toPrecision(16);
  if (s.includes('e')) return Number(s).toPrecision(16).replace(/\.?0+e/, 'e');
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
};

/** Stability: sub-steps needed for explicit FTCS, from the measured rule (see header). */
export const substepCount = (C: number): number => (C <= 0.5 + 1e-12 ? 1 : Math.floor(2 * C + 1e-9) + 1);

interface TtmParams {
  seed: number;
  Ce: number;
  rhoE: number;
  kappa: number;
  gammaP: number;
  gammaS: number;
  v0: number;
  nx: number;
  ny: number;
  nz: number;
}

/** Parses the common argument list of fix ttm and fix ttm/grid. */
export const parseTtmArgs = (id: string, style: string, args: string[]): TtmParams => {
  if (args.length < 10) throw new StyleError(`usage: fix ID group ${style} seed C_e rho_e kappa_e gamma_p gamma_s v_0 Nx Ny Nz [keywords]`);
  const num = (w: string, what: string): number => {
    const v = Number(w);
    if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`fix ${id} ${style}: ${what} must be a number, got '${w}'`);
    return v;
  };
  const posInt = (w: string, what: string): number => {
    const v = num(w, what);
    if (!Number.isInteger(v) || v < 1) throw new StyleError(`fix ${id} ${style}: ${what} must be a positive integer, got '${w}'`);
    return v;
  };
  const p: TtmParams = {
    seed: posInt(args[0], 'seed'),
    Ce: num(args[1], 'C_e'),
    rhoE: num(args[2], 'rho_e'),
    kappa: num(args[3], 'kappa_e'),
    gammaP: num(args[4], 'gamma_p'),
    gammaS: num(args[5], 'gamma_s'),
    v0: num(args[6], 'v_0'),
    nx: posInt(args[7], 'Nx'),
    ny: posInt(args[8], 'Ny'),
    nz: posInt(args[9], 'Nz'),
  };
  if (!(p.Ce > 0)) throw new StyleError(`fix ${id} ${style}: C_e must be > 0`);
  if (!(p.rhoE > 0)) throw new StyleError(`fix ${id} ${style}: rho_e must be > 0`);
  if (p.kappa < 0) throw new StyleError(`fix ${id} ${style}: kappa_e must be >= 0`);
  // Measured with native LAMMPS (black box): gamma_p = 0 stops the run with an error that requires gamma_p > 0.
  if (!(p.gammaP > 0)) throw new StyleError(`fix ${id} ${style}: gamma_p must be > 0`);
  if (p.gammaS < 0) throw new StyleError(`fix ${id} ${style}: gamma_s must be >= 0`);
  if (p.v0 < 0) throw new StyleError(`fix ${id} ${style}: v_0 must be >= 0`);
  return p;
};

/**
 * Keywords after the required arguments. Returns the initial grid spec.
 * Shared with fix ttm/mod. `outfileAllowed` is false for ttm/grid.
 */
export interface TtmKeywords {
  setT: number | null;
  infile: string | null;
  outNevery: number;
  outfile: string | null;
}

export const parseTtmKeywords = (id: string, style: string, args: string[], outfileAllowed: boolean): TtmKeywords => {
  const k: TtmKeywords = { setT: null, infile: null, outNevery: 0, outfile: null };
  for (let i = 0; i < args.length;) {
    const key = args[i];
    if (key === 'set') {
      const t = Number(args[i + 1]);
      if (args[i + 1] === undefined || !Number.isFinite(t)) throw new StyleError(`fix ${id} ${style}: keyword set needs a temperature`);
      // Measured with native LAMMPS (black box): set Tinit = 0 stops the run with an error that requires Tinit > 0.
      if (!(t > 0)) throw new StyleError(`fix ${id} ${style}: set Tinit must be > 0`);
      k.setT = t;
      i += 2;
    } else if (key === 'infile') {
      if (args[i + 1] === undefined) throw new StyleError(`fix ${id} ${style}: keyword infile needs a file name`);
      k.infile = args[i + 1];
      i += 2;
    } else if (key === 'outfile') {
      if (!outfileAllowed) throw new StyleError(`fix ${id} ${style} does not support the outfile keyword (use dump grid or restart instead)`);
      const n = Number(args[i + 1]);
      if (!Number.isInteger(n) || n < 1 || args[i + 2] === undefined) throw new StyleError(`fix ${id} ${style}: outfile needs Nout (positive integer) and a file name`);
      k.outNevery = n;
      k.outfile = args[i + 2];
      i += 3;
    } else {
      throw new StyleError(`fix ${id} ${style}: unknown keyword '${key}'`);
    }
  }
  return k;
};

/**
 * Reads the infile grid: "Each line contains four numeric columns: ix,iy,iz,Temperature."
 * Returns one temperature per grid point (x fastest).
 */
export const parseTtmInfile = (text: string, id: string, style: string, nx: number, ny: number, nz: number, units: string, fname: string): Float64Array => {
  const lines = text.split('\n');
  const first = lines[0] ?? '';
  const tag = /UNITS:\s*(\S+)/.exec(first);
  if (tag && tag[1] !== units) {
    throw new StyleError(`fix ${id} ${style}: infile ${fname} was written for units ${tag[1]} but ${units} units are in use`);
  }
  const n = nx * ny * nz;
  const out = new Float64Array(n);
  const seen = new Uint8Array(n);
  for (const raw of lines) {
    const line = raw.replace(/#.*/, '').trim();
    if (!line) continue;
    const w = line.split(/\s+/);
    if (w.length < 4) throw new StyleError(`fix ${id} ${style}: infile ${fname} line needs ix iy iz T: '${raw.trim()}'`);
    const ix = Number(w[0]), iy = Number(w[1]), iz = Number(w[2]), t = Number(w[3]);
    if (![ix, iy, iz, t].every(Number.isFinite)) throw new StyleError(`fix ${id} ${style}: infile ${fname} has a non-numeric entry: '${raw.trim()}'`);
    if (!Number.isInteger(ix) || !Number.isInteger(iy) || !Number.isInteger(iz) || ix < 1 || ix > nx || iy < 1 || iy > ny || iz < 1 || iz > nz) {
      throw new StyleError(`fix ${id} ${style}: infile ${fname} grid index out of range: '${raw.trim()}'`);
    }
    if (t < 0) throw new StyleError(`fix ${id} ${style}: infile ${fname} electron temperatures must not be negative`);
    const c = (ix - 1) + nx * ((iy - 1) + ny * (iz - 1));
    if (seen[c]) throw new StyleError(`fix ${id} ${style}: infile ${fname} sets grid point ${ix} ${iy} ${iz} twice`);
    seen[c] = 1;
    out[c] = t;
  }
  for (let c = 0; c < n; c++) {
    if (!seen[c]) throw new StyleError(`fix ${id} ${style}: infile ${fname} did not set all ${nx}x${ny}x${nz} grid temperatures`);
  }
  return out;
};

/**
 * fix ttm (and ttm/grid, identical in a serial run: the grid is simply
 * stored once, see the header). Uses the cell-averaged electron temperature
 * of each atom's cell for the coupling.
 */
export class FixTTM extends Fix {
  readonly style: string;
  vectorFlag = true;
  sizeVector = 2;
  extvector = 1;

  protected readonly p: TtmParams;
  protected readonly kw: TtmKeywords;
  protected Te: Float64Array;
  protected readonly cellVolume: number;
  protected readonly rng: RanMars;
  /** The fix force of the current step per atom (3N), kept for the energy tally. */
  protected fl: Float64Array;
  /** Native storage order (physical atom index per slot) when fl was computed. */
  protected flOrder = new Int32Array(0);
  protected lastTransfer = 0;

  constructor(sys: System, id: string, group: string, args: string[], style: 'ttm' | 'ttm/grid' = 'ttm') {
    super(sys, id, group, args);
    this.style = style;
    const s = sys.state;
    if (s.dimension !== 3) throw new StyleError(`fix ${id} ${style} requires a 3d simulation`);
    if (s.box.triclinic) throw new StyleError(`fix ${id} ${style} requires an orthogonal box (triclinic boxes are not supported)`);
    if (!s.box.periodic.every(Boolean)) throw new StyleError(`fix ${id} ${style} requires periodic boundaries in all dimensions`);
    this.p = parseTtmArgs(id, style, args);
    this.kw = parseTtmKeywords(id, style, args.slice(10), style === 'ttm');
    const nGrid = this.p.nx * this.p.ny * this.p.nz;
    const lx = s.box.hi[0] - s.box.lo[0], ly = s.box.hi[1] - s.box.lo[1], lz = s.box.hi[2] - s.box.lo[2];
    this.cellVolume = (lx / this.p.nx) * (ly / this.p.ny) * (lz / this.p.nz);
    this.Te = new Float64Array(nGrid);
    if (this.kw.infile !== null) {
      this.Te = parseTtmInfile(sys.readFile(this.kw.infile), id, style, this.p.nx, this.p.ny, this.p.nz, s.units.style, this.kw.infile);
    } else if (this.kw.setT !== null) {
      this.Te.fill(this.kw.setT);
    }
    // Measured with native LAMMPS (black box): the first draw of the generator is discarded at creation.
    this.rng = new RanMars(this.p.seed);
    this.rng.uniform();
    this.fl = new Float64Array(0);
  }

  /** Grid index of an atom (periodic wrap, x fastest). */
  protected cellOf(i: number): number {
    const s = this.sys.state;
    const { nx, ny, nz } = this.p;
    const cell = (x: number, lo: number, len: number, n: number): number => {
      let f = (x - lo) / len;
      f -= Math.floor(f);
      return Math.min(n - 1, Math.max(0, Math.floor(f * n)));
    };
    const ix = cell(s.x[3 * i], s.box.lo[0], s.box.hi[0] - s.box.lo[0], nx);
    const iy = cell(s.x[3 * i + 1], s.box.lo[1], s.box.hi[1] - s.box.lo[1], ny);
    const iz = cell(s.x[3 * i + 2], s.box.lo[2], s.box.hi[2] - s.box.lo[2], nz);
    return ix + nx * (iy + ny * iz);
  }

  /** Heat capacity times volume of one cell: energy per kelvin. */
  protected get heatPerCell(): number { return this.p.Ce * this.p.rhoE * this.cellVolume; }

  /**
   * Native LAMMPS adds no new fix force on the setup step (measured: step 0 forces are zero). A later run
   * keeps the fix force of the previous step, but by storage slot: the first half-kick of the second run
   * applies the force of slot k to whatever atom now occupies slot k, because the setup sort reorders
   * the atoms (measured with native LAMMPS, 8 atoms, dt 0.2: the implied kick of atom i equals the step-3
   * force of the atom that was in its slot before the sort; the same test on one atom gives the force
   * of that atom). So setup adds the stored per-slot force, not a fresh draw and not the force of the
   * atom itself.
   */
  setup(): void {
    if (this.fl.length === 0) return;
    const s = this.sys.state;
    if (this.flOrder.length !== s.n) {
      throw new StyleError(`fix ${this.id} ${this.style}: atoms were added or deleted between runs (carry-over of the last force is not supported)`);
    }
    const ord = nativeOrder(s);
    for (let k = 0; k < s.n; k++) {
      const i = ord[k], j = this.flOrder[k];
      if (!(s.mask[i] & this.groupBit)) continue;
      s.f[3 * i] += this.fl[3 * j]; s.f[3 * i + 1] += this.fl[3 * j + 1]; s.f[3 * i + 2] += this.fl[3 * j + 2];
    }
  }

  postForce(): void {
    const s = this.sys.state;
    const u = s.units;
    const { f, v, mask } = s;
    const { gammaP, gammaS, v0 } = this.p;
    if (this.fl.length !== 3 * s.n) this.fl = new Float64Array(3 * s.n);
    this.flOrder = Int32Array.from(nativeOrder(s));
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      const T = this.Te[this.cellOf(i)];
      // g1 multiplies v (friction), g2 scales the uniform deviate (see header)
      const g1 = -gammaP / u.ftm2v;
      const g2 = Math.sqrt((24 * gammaP * u.boltz * T) / (u.mvv2e * s.dt)) / u.ftm2v;
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
      const stop = speed > v0 ? -gammaS / u.ftm2v : 0;
      const rx = g2 * (this.rng.uniform() - 0.5);
      const ry = g2 * (this.rng.uniform() - 0.5);
      const rz = g2 * (this.rng.uniform() - 0.5);
      const fx = g1 * vx + stop * vx + rx;
      const fy = g1 * vy + stop * vy + ry;
      const fz = g1 * vz + stop * vz + rz;
      f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
      this.fl[3 * i] = fx; this.fl[3 * i + 1] = fy; this.fl[3 * i + 2] = fz;
    }
  }

  endOfStep(): void {
    const s = this.sys.state;
    const { nx, ny, nz } = this.p;
    const cells = nx * ny * nz;
    // energy handed to the atoms this step: sum F . v dt over the group, per cell
    const transfer = new Float64Array(cells);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const dE = (this.fl[3 * i] * s.v[3 * i] + this.fl[3 * i + 1] * s.v[3 * i + 1] + this.fl[3 * i + 2] * s.v[3 * i + 2]) * s.dt;
      transfer[this.cellOf(i)] += dE;
    }
    let total = 0;
    for (let c = 0; c < cells; c++) total += transfer[c];
    this.lastTransfer = total;
    this.advance(transfer);
    if (this.kw.outfile !== null && s.step % this.kw.outNevery === 0) this.writeGrid(s.step);
  }

  /**
   * One end-of-step update of the electron grid: the heat diffusion equation solved explicitly (FTCS,
   * periodic), split into k sub-steps when needed (see header). Each sub-step also removes its share
   * transfer/k of the energy handed to the atoms. Measured with native LAMMPS (black box, a k = 2 step
   * with coupling on, outfile compared cell by cell): the share goes in every sub-step; putting the whole
   * transfer before or after the diffusion does not reproduce the native grid.
   */
  protected advance(transfer: Float64Array): void {
    const { nx, ny, nz, kappa } = this.p;
    const s = this.sys.state;
    const lx = (s.box.hi[0] - s.box.lo[0]) / nx, ly = (s.box.hi[1] - s.box.lo[1]) / ny, lz = (s.box.hi[2] - s.box.lo[2]) / nz;
    const D = kappa / (this.p.Ce * this.p.rhoE);
    const C = s.dt * D * (1 / (lx * lx) + 1 / (ly * ly) + 1 / (lz * lz));
    const k = substepCount(C);
    const h = s.dt / k;
    const cells = nx * ny * nz;
    const next = new Float64Array(cells);
    const heat = this.heatPerCell * k;
    const idx = (ix: number, iy: number, iz: number) => ((ix + nx) % nx) + nx * (((iy + ny) % ny) + ny * ((iz + nz) % nz));
    for (let sub = 0; sub < k; sub++) {
      for (let iz = 0; iz < nz; iz++) {
        for (let iy = 0; iy < ny; iy++) {
          for (let ix = 0; ix < nx; ix++) {
            const c = idx(ix, iy, iz);
            const t = this.Te[c];
            const lap = (this.Te[idx(ix + 1, iy, iz)] - 2 * t + this.Te[idx(ix - 1, iy, iz)]) / (lx * lx)
              + (this.Te[idx(ix, iy + 1, iz)] - 2 * t + this.Te[idx(ix, iy - 1, iz)]) / (ly * ly)
              + (this.Te[idx(ix, iy, iz + 1)] - 2 * t + this.Te[idx(ix, iy, iz - 1)]) / (lz * lz);
            next[c] = t + h * D * lap - transfer[c] / heat;
            // the coupling needs sqrt(T_e): a cell that cools below zero has no defined amplitude (native not measured)
            if (!(next[c] >= 0)) throw new StyleError(`fix ${this.id} ${this.style}: electron temperature of grid cell ${c + 1} became negative`);
          }
        }
      }
      this.Te.set(next);
    }
  }

  /** Writes the grid in the infile layout (outfile "T.out.<step>"). */
  protected writeGrid(step: number): void {
    const { nx, ny, nz } = this.p;
    const s = this.sys.state;
    const date = new Date().toISOString().slice(0, 10);
    const head = `# DATE: ${date} UNITS: ${s.units.style} COMMENT: Electron temperature on ${nx}x${ny}x${nz} grid at step ${step} - created by fix ${this.style}\n`;
    const lines: string[] = [];
    for (let iz = 0; iz < nz; iz++) for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) {
      lines.push(`${ix + 1} ${iy + 1} ${iz + 1} ${fmt16g(this.Te[ix + nx * (iy + ny * iz)])}`);
    }
    this.sys.writeFile(`${this.kw.outfile}.${step}`, head + lines.join('\n') + '\n', false);
  }

  computeVector(i: number): number {
    if (i === 0) {
      let e = 0;
      for (let c = 0; c < this.Te.length; c++) e += this.heatPerCell * this.Te[c];
      return e;
    }
    if (i === 1) return this.lastTransfer;
    throw new StyleError(`fix ${this.id} ${this.style}: vector index ${i + 1} out of range`);
  }

  boxChanged(): void {
    throw new StyleError(`fix ${this.id} ${this.style}: the box size and shape must not change (fix npt, fix deform and the like are not supported)`);
  }
}
