import { StyleError, potentialFile } from '../force/types';
import { inumeric, missingArgs } from '../force/util';
import type { System } from '../system';
import { FixTTM, type TtmPositional } from './ttm';
import { libm } from '../libm';

/*
 * fix ttm/mod — the two-temperature model with a temperature-dependent electronic heat capacity and
 * conductivity, a laser source and a region of coupled cells (docs.lammps.org/fix_ttm.html, section
 * "Additional details for fix ttm/mod", plans/lammps-docs-live/fix_ttm.rst). It is fix ttm (fix/ttm.ts,
 * the base class) with the hooks and the advance() below.
 *
 * Syntax: fix ID group-ID ttm/mod seed init_file Nx Ny Nz keyword value ...; keywords set, infile, outfile.
 *
 * Quoted doc sentences (docs.lammps.org/fix_ttm.html):
 *  "Fix *ttm/mod* adds options to account for external heat sources (e.g. at"
 *  "heat sources (e.g. laser heating in ablation simulations):"
 *  "duration of the pulse is set by the parameter *tau* in the *init_file*."
 *  "The current fix *ttm/mod* implementation allows TTM simulations with a"
 *  "The vacuum region is defined as the grid cells with zero"
 *  "The numerical scheme does not allow energy"
 *  "The *lsurface* and *rsurface* parameters always define the range of"
 *  "cells whose x grid index *ix* satisfies *lsurface* <= *ix* <"
 *  "l_skin, electron grid units (positive integer)"
 *  "Every line with an odd number is considered as a comment and"
 *
 * Model: C_e = C_0 + (a_0 + a_1 X + a_2 X^2 + a_3 X^3 + a_4 X^4) exp(-(A X)^2), X = T_e / 1000 (doc);
 * kappa_e = D_e rho_e C_e (doc); the coupling and the stopping are the base class ones; the laser source
 * is I_0 exp(-x / l_skin) (doc), with the amplitude and the time window measured below.
 *
 * Measured with native LAMMPS (black box; 8-atom crystals with 2x2x2 or 4x4x4 grids, per-step dumps of
 * positions, velocities and forces, per-step outfiles, thermo f_2[1], f_2[2]):
 *  - energy: a cell holds V rho_e Int_0^T C_e(t) dt (a closed integral, not C_e(T) T; f_2[1] = 429689.70099 for
 *    a_0 = 1, A = 0.18, C_0 = 1, rho_e = 1, eight cells at 1000 K);
 *  - diffusion: dT_c/dt = 1/(rho C(T_c)) sum_nb D_e rho C((T_c + T_nb)/2) (T_nb - T_c)/dx_d^2 (explicit; the
 *    conductance at the mean temperature of the pair, the heat capacity of the cell itself; the own-cell and the
 *    harmonic and arithmetic means do not match); a neighbour with T_e = 0 is skipped; the flow is not limited
 *    to the coupled cells;
 *  - sub-steps: k = max(2, floor(4 C_s) + 1) with C_s = dt D_e sum_d 1/dx_d^2 (2 up to C_s = 0.49, 3 at 0.51,
 *    5 at 1.0, 13 at 3.0 on a cube; 4 at C_s = 0.8 on a 6 x 6 x 12 box with a 4^3 grid: the sum over the
 *    directions, not the largest one);
 *  - a sub-step is three updates in this order, each divided by the heat capacity the previous one left:
 *    diffusion with C(T), laser source with C after the diffusion, coupling transfer / k with C after the source;
 *    the other orders differ by 1e-3 K (transfer before or after the diffusion, on the 4^3 diffusion case) and by
 *    2e-8 K (source with C before the diffusion, on the laser case);
 *  - the coupling (forces, random stream, transfer f_2[2]) is the base one with gamma_p, gamma_s, v_0 and the
 *    lambda and n_ion of the file (no effect while B = 0);
 *  - cells: floor(x / (L / N)) (the cell width rounded first; x / L * N puts the faces of the example's 54.309 A box
 *    one cell low), with the periodic wrap for a position below lo by more than rounding (an atom below lo by 9e-6 A
 *    is in the top cell from the second step on), a rounding error below lo is in the first cell, and x = hi is in
 *    the last cell; this decides the cell of the atoms of the diamond lattice, which sit on the cell faces;
 *  - an empty cell (T_e = 0) exchanges no energy and its atoms keep the friction and the electronic stopping
 *    (gamma_p + gamma_s above v_0) with no force from the random part, and they draw their random numbers;
 *  - atoms outside lsurface <= ix < rsurface take no coupling force and draw nothing; lsurface < 0 and
 *    lsurface >= rsurface are errors; rsurface above Nx is accepted; n_ion = 0 is an error;
 *  - laser (I_0 != 0): per sub-step T_c += h S_c / (rho C), S_c = I_0 / (dx_x l_skin) exp(-(ix - lsurface) / l_skin)
 *    for ix >= lsurface (no upper limit; the amplitude scales with 1/dx_x, 1/l_skin); no source in an empty cell;
 *    on while the sub-step ends before tau, the clock being the sum of the timesteps since the fix was defined
 *    (a sub-step ending exactly at tau is off);
 *  - the outfile lists the cells with z fastest (1 1 1, 1 1 2, 1 2 1, ...); its header line reads, after the date,
 *    UNITS: metal COMMENT: Electron temperature 4x4x4 grid at step 1. Created by fix ttm/mod (native text, no quotes);
 *  - the repeated setup row of a later run reports f_2[2] = 0.
 *
 * Electron heat capacity <= 0 (the Si oracle case, C_e(1000 K) < 0): the sub-step count is 2 (the rule above
 * gives 4 for the same case with C_0 raised so that C_e > 0). Measured on that one case only.
 *
 * Not reproduced (refused with a StyleError): B != 0 (electronic pressure force), surface_movement 1 (moving
 * vacuum surface), and I_0 != 0 with l_skin <= 0.
 */

/** The positional part of fix ttm/mod: seed, init_file, Nx, Ny, Nz. */
export interface TtmModPositional extends TtmPositional {
  initFile: string;
}

export const parseTtmModPositional = (id: string, args: string[]): TtmModPositional => {
  // As native LAMMPS does (src/EXTRA-FIX/fix_ttm_mod.cpp FixTTMMod::FixTTMMod at b332f28): the integers are read with
  // utils::inumeric, and the seed and grid sizes are checked after the keywords
  if (args.length < 5) throw missingArgs('fix ttm/mod');
  const seed = inumeric(args[0]);
  const nx = inumeric(args[2]), ny = inumeric(args[3]), nz = inumeric(args[4]);
  const check = (): void => {
    if (seed <= 0) throw new StyleError(`Invalid random number seed ${seed} in fix ttm/mod command`);
    if (nx <= 0 || ny <= 0 || nz <= 0) throw new StyleError('Fix ttm/mod grid sizes must be > 0');
  };
  return { seed, initFile: args[1], nx, ny, nz, consumed: 5, scalar: null, check };
};

export interface TtmModParams {
  a: number[];
  Cz: number;
  A: number;
  rhoE: number;
  De: number;
  gammaP: number;
  gammaS: number;
  v0: number;
  I0: number;
  lsurface: number;
  rsurface: number;
  lSkin: number;
  tau: number;
  B: number;
  lambda: number;
  nIon: number;
  surfaceMovement: number;
  teMin: number;
}

const INTEGER_KEYS = new Set(['lsurface', 'rsurface', 'l_skin', 'surface_movement']);
const KEY_ORDER = ['a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'C_0', 'A', 'rho_e', 'D_e', 'gamma_p', 'gamma_s', 'v_0', 'I_0',
  'lsurface', 'rsurface', 'l_skin', 'tau', 'B', 'lambda', 'n_ion', 'surface_movement', 'T_e_min'];

/** Reads the parameter file: even-numbered lines (1-based) carry the values, odd lines are comments. */
export const parseTtmModParams = (text: string, id: string, fname: string): TtmModParams => {
  const lines = text.split('\n');
  const values: Record<string, number> = {};
  KEY_ORDER.forEach((key, k) => {
    const raw = lines[2 * k + 1];
    if (raw === undefined) throw new StyleError(`fix ${id} ttm/mod: parameter file ${fname} ends before '${key}'`);
    const w = raw.trim().split(/\s+/)[0] ?? '';
    if (INTEGER_KEYS.has(key)) {
      // Measured with native LAMMPS (black box): 1.0 and 0.5 in these slots stop the run with an integer-parse error.
      if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`fix ${id} ttm/mod: ${key} in ${fname} must be an integer, got '${w}'`);
    }
    const v = Number(w);
    if (w === '' || !Number.isFinite(v)) throw new StyleError(`fix ${id} ttm/mod: ${key} in ${fname} must be a number, got '${w}'`);
    values[key] = v;
  });
  return {
    a: [values.a_0, values.a_1, values.a_2, values.a_3, values.a_4],
    Cz: values.C_0,
    A: values.A,
    rhoE: values.rho_e,
    De: values.D_e,
    gammaP: values.gamma_p,
    gammaS: values.gamma_s,
    v0: values.v_0,
    I0: values.I_0,
    lsurface: values.lsurface,
    rsurface: values.rsurface,
    lSkin: values.l_skin,
    tau: values.tau,
    B: values.B,
    lambda: values.lambda,
    nIon: values.n_ion,
    surfaceMovement: values.surface_movement,
    teMin: values.T_e_min,
  };
};

/**
 * Sub-step count of one end-of-step update: max(2, floor(4 C_s) + 1) with C_s = dt D_e sum_d 1/dx_d^2 (see the header);
 * two when a heat capacity is not positive. Exported for the unit tests.
 */
export const ttmModSubsteps = (Cs: number, negativeCv: boolean): number => (negativeCv ? 2 : Math.max(2, Math.floor(4 * Cs) + 1));

/**
 * Grid index of a position along one direction (see FixTTMMod.cellOf): g = (x - lo) / (hi - lo) * n; a rounding error below lo
 * is index 0, a lower position wraps to the top, x = hi is n - 1. Exported for the unit tests.
 */
export const ttmModGridIndex = (x: number, lo: number, hi: number, n: number): number => {
  // the cell width L / n is rounded first (x / (L / n)): this is the native index on the example's 54.309 A box, which
  // a x / L * n index gets wrong on its cell faces by one cell
  const g = (x - lo) / ((hi - lo) / n);
  if (g < 0 && g > -1e-12) return 0;
  const f = Math.floor(g);
  if (f < 0) return ((f % n) + n) % n;
  return Math.min(n - 1, f);
};

/** Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration on the Legendre polynomial). */
const gaussLegendre = (n: number): { x: number[]; w: number[] } => {
  const x: number[] = [], w: number[] = [];
  for (let i = 1; i <= n; i++) {
    let z = libm.cos((Math.PI * (i - 0.25)) / (n + 0.5));
    let dp = 1;
    for (let iter = 0; iter < 100; iter++) {
      let p0 = 1, p1 = z;
      for (let k = 2; k <= n; k++) {
        const p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1; p1 = p2;
      }
      dp = (n * (z * p1 - p0)) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    x.push(z);
    w.push(2 / ((1 - z * z) * dp * dp));
  }
  return { x, w };
};

const GL = gaussLegendre(16);
const PANELS = 4;

/**
 * fix ttm/mod: temperature-dependent C_e and kappa_e (see the header), the laser source and the vacuum
 * region. Coupling, stopping and the random stream are the base class ones.
 */
export class FixTTMMod extends FixTTM {
  readonly par: TtmModParams;
  /** Fix clock: the sum of the timesteps since the fix was defined (measured: see the header). */
  private elapsed = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    const pos = parseTtmModPositional(id, args);
    super(sys, id, group, args, 'ttm/mod', pos);
    const fname = pos.initFile;
    this.par = parseTtmModParams(sys.readFile(fname, potentialFile('ttm/mod parameter')), id, fname);
    const p = this.par;
    // Measured with native LAMMPS (black box): n_ion = 0 stops the run with a message that says ionic_density must be > 0.0.
    if (!(p.nIon > 0)) throw new StyleError(`fix ${id} ttm/mod: n_ion (ionic density) must be > 0`);
    // Measured with native LAMMPS (black box): a negative surface index stops the run; lsurface >= rsurface stops it too.
    if (p.lsurface < 0 || p.rsurface < 0) throw new StyleError(`fix ${id} ttm/mod: Surface coordinates must be >= 0`);
    if (p.lsurface >= p.rsurface) throw new StyleError(`fix ${id} ttm/mod: Left surface coordinate must be less than right surface coordinate`);
    if (!(p.rhoE > 0)) throw new StyleError(`fix ${id} ttm/mod: rho_e must be > 0`);
    if (p.De < 0) throw new StyleError(`fix ${id} ttm/mod: D_e must be >= 0`);
    if (p.gammaS < 0) throw new StyleError(`fix ${id} ttm/mod: gamma_s must be >= 0`);
    if (p.v0 < 0) throw new StyleError(`fix ${id} ttm/mod: v_0 must be >= 0`);
    if (p.gammaP < 0) throw new StyleError(`fix ${id} ttm/mod: gamma_p must be >= 0`);
    const unsupported = ttmModBlockers(p);
    if (unsupported.length > 0) throw new StyleError(`fix ${id} ttm/mod: not reproduced in this engine: ${unsupported.join('; ')}`);
    if (p.I0 !== 0 && !(p.lSkin > 0)) throw new StyleError(`fix ${id} ttm/mod: l_skin must be > 0 when I_0 is set`);
  }

  /** Electron heat capacity C_e(T) of the doc: C_0 + (a_0 + a_1 X + ... + a_4 X^4) exp(-(A X)^2), X = T/1000. */
  cof(T: number): number {
    const p = this.par;
    const X = T / 1000;
    const poly = p.a[0] + X * (p.a[1] + X * (p.a[2] + X * (p.a[3] + X * p.a[4])));
    return p.Cz + poly * libm.exp(-(p.A * X) * (p.A * X));
  }

  /** Int_0^T C_e(t) dt (composite Gauss-Legendre; the integrand is smooth on the scale of the panels). */
  private heatIntegral(T: number): number {
    if (!(T > 0)) return 0;
    const h = T / PANELS;
    let sum = 0;
    for (let q = 0; q < PANELS; q++) {
      const mid = (q + 0.5) * h, half = h / 2;
      for (let i = 0; i < GL.x.length; i++) sum += GL.w[i] * half * this.cof(mid + half * GL.x[i]);
    }
    return sum;
  }

  protected cvolAt(_c: number, T: number): number {
    return this.par.rhoE * this.cof(T);
  }

  protected energyAt(_c: number, T: number): number {
    return this.cellVolume * this.par.rhoE * this.heatIntegral(T);
  }

  protected gammaPFor(_c: number, _speed: number): number {
    return this.par.gammaP;
  }

  protected gammaSFor(speed: number): number {
    return speed > this.par.v0 ? this.par.gammaS : 0;
  }

  /** Empty cells (T_e = 0) are vacuum. */
  protected vacuum(c: number): boolean {
    return this.Te[c] === 0;
  }

  /**
   * Atoms outside lsurface <= ix < rsurface take no electron-ion coupling. An empty cell (T_e = 0) is not uncoupled: its atoms
   * keep the friction and the electronic stopping and draw the random numbers with a zero amplitude (measured with native
   * LAMMPS: the friction coefficient of an atom in an empty cell is gamma_p + gamma_s above v_0, and the next atom's draws
   * are unchanged; the energy of an empty cell is not changed).
   */
  protected uncoupled(c: number): boolean {
    const ix = c % this.p.nx;
    return ix < this.par.lsurface || ix >= this.par.rsurface;
  }

  /**
   * Grid cell of an atom along one direction: g = (x - lo) / (L / N); an atom a rounding error below lo (|g| < 1e-12) is in the first
   * cell, an atom further below lo wraps to the top cell (periodic image), and x = hi is in the last cell. Measured with native
   * LAMMPS (black box): an atom on a cell face goes to the upper cell (x = 1.5 of a 6 A box with 4 cells: cell 2); an atom at
   * x = hi goes to the last cell (not to cell 1); an atom at x = -5.6e-21 (the lattice rounding of the Si oracle) is in cell 1
   * at its first step, while an atom at x = -9e-6 that has moved below lo is in the top cell by the second step (matches the
   * native grid to 1e-12 on the 8-atom diamond case; the other rules differ by up to 3.5 K).
   */
  protected cellOf(i: number): number {
    const s = this.sys.state;
    const { nx, ny, nz } = this.p;
    const ix = ttmModGridIndex(s.x[3 * i], s.box.lo[0], s.box.hi[0], nx);
    const iy = ttmModGridIndex(s.x[3 * i + 1], s.box.lo[1], s.box.hi[1], ny);
    const iz = ttmModGridIndex(s.x[3 * i + 2], s.box.lo[2], s.box.hi[2], nz);
    return ix + nx * (iy + ny * iz);
  }

  /**
   * On the setup of a later run the thermo output shows the electron-atom transfer of that setup as zero (measured with
   * native LAMMPS: f_2[2] of the repeated setup row is 0, where fix ttm repeats the last value).
   */
  setup(): void {
    super.setup();
    this.lastTransfer = 0;
  }

  endOfStep(): void {
    this.elapsed += this.sys.state.dt;
    super.endOfStep();
  }

  protected outfileZFastest = true;

  protected outfileComment(step: number): string {
    const { nx, ny, nz } = this.p;
    return `Electron temperature ${nx}x${ny}x${nz} grid at step ${step}. Created by fix ttm/mod`;
  }

  /**
   * One end-of-step update: k explicit sub-steps (k = max(2, floor(4 C_s) + 1), C_s = dt D_e sum 1/dx_d^2),
   * each adding the diffusion with the kappa of the mean temperature of each pair, the laser source while it is
   * on, and the coupling transfer / k, all divided by the volumetric heat capacity rho_e C_e(T_c) of the cell.
   */
  protected advance(transfer: Float64Array): void {
    const { nx, ny, nz } = this.p;
    const s = this.sys.state;
    const p = this.par;
    const dx = [0, 1, 2].map((d) => (s.box.hi[d] - s.box.lo[d]) / [nx, ny, nz][d]);
    const cells = nx * ny * nz;
    const Cs = s.dt * p.De * (1 / (dx[0] * dx[0]) + 1 / (dx[1] * dx[1]) + 1 / (dx[2] * dx[2]));
    // Sub-step count: max(2, floor(4 C_s) + 1) for positive heat capacities (measured on cubic and non-cubic grids,
    // see the header). With a non-positive heat capacity in some cell the native count is 2 whatever C_s is (measured
    // on the Si case: C_e(1000 K) < 0; the same case with C_0 raised to make C_e positive takes four sub-steps).
    let negativeCv = false;
    for (let c = 0; c < cells && !negativeCv; c++) {
      if (this.Te[c] !== 0 && this.cvolAt(c, this.Te[c]) <= 0) negativeCv = true;
    }
    const k = ttmModSubsteps(Cs, negativeCv);
    if (!(k <= 1e6)) throw new StyleError(`fix ${this.id} ${this.style}: the electron diffusion needs ${k} sub-steps per step`);
    const h = s.dt / k;
    const t0 = this.elapsed - s.dt;
    const amplitude = new Float64Array(cells);
    for (let c = 0; c < cells; c++) {
      const ix = c % nx;
      if (p.I0 === 0 || ix < p.lsurface) continue;
      amplitude[c] = (p.I0 / (dx[0] * p.lSkin)) * libm.exp(-(ix - p.lsurface) / p.lSkin);
    }
    for (let sub = 0; sub < k; sub++) {
      // the laser is on while the sub-step's end time is before tau (measured: a sub-step ending exactly at tau is off)
      const laserOn = p.I0 !== 0 && t0 + (sub + 1) * h < p.tau;
      const next = new Float64Array(cells);
      for (let iz = 0; iz < nz; iz++) {
        for (let iy = 0; iy < ny; iy++) {
          for (let ix = 0; ix < nx; ix++) {
            const c = this.wrapIndex(ix, iy, iz);
            const t = this.Te[c];
            if (t === 0) { next[c] = 0; continue; }
            const cv = this.cvolAt(c, t);
            // a negative C_e is allowed (measured: the example's parameters give C_e < 0 near 1000 K and the run goes on)
            if (cv === 0) throw new StyleError(`fix ${this.id} ${this.style}: the electron heat capacity is zero at ${t} K`);
            const nb: [number, number][] = [
              [this.wrapIndex(ix + 1, iy, iz), 0], [this.wrapIndex(ix - 1, iy, iz), 0],
              [this.wrapIndex(ix, iy + 1, iz), 1], [this.wrapIndex(ix, iy - 1, iz), 1],
              [this.wrapIndex(ix, iy, iz + 1), 2], [this.wrapIndex(ix, iy, iz - 1), 2],
            ];
            let flux = 0;
            for (const [j, d] of nb) {
              const tj = this.Te[j];
              if (tj === 0) continue;
              const kappa = p.De * p.rhoE * this.cof(0.5 * (t + tj));
              flux += kappa * (tj - t) / (dx[d] * dx[d]);
            }
            // Measured with native LAMMPS (black box, 4x4x4 grid, three sub-steps, coupling on, and the laser on): the
            // sub-step is three updates in this order, each divided by the heat capacity at the temperature the
            // previous one left: diffusion (C at T), laser source (C after the diffusion), electron-atom transfer / k
            // (C after the source). The split matches the native grid to 1e-11; the other orders do not (3e-4 and 2e-8).
            const source = laserOn ? amplitude[c] : 0;
            const tD = t + (h * flux) / cv;
            const tS = source !== 0 ? tD + (h * source) / this.cvolAt(c, tD) : tD;
            const cvS = this.cvolAt(c, tS);
            if (cvS === 0) throw new StyleError(`fix ${this.id} ${this.style}: the electron heat capacity is zero at ${tS} K`);
            next[c] = tS - transfer[c] / (this.cellVolume * cvS * k);
            // a negative electron temperature has no coupling amplitude; the engine stops (not measured with native LAMMPS)
            if (!(next[c] >= 0)) throw new StyleError(`fix ${this.id} ${this.style}: electron temperature of grid cell ${c + 1} became negative`);
          }
        }
      }
      this.Te.set(next);
    }
  }
}

/**
 * The parts of a fix ttm/mod input this engine does not reproduce (empty = none). The electronic pressure
 * (B) and the moving vacuum surface (surface_movement) are not measured; they are refused by name.
 */
export const ttmModBlockers = (p: TtmModParams): string[] => {
  const out: string[] = [];
  if (p.B !== 0) out.push('B != 0 (electronic pressure force)');
  if (p.surfaceMovement !== 0) out.push('surface_movement 1 (moving vacuum surface)');
  return out;
};
