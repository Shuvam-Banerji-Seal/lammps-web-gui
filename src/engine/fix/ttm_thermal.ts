import { StyleError } from '../force/types';
import type { System } from '../system';
import { missingArgs } from '../force/util';
import { FixTTM, type TtmKeywordHandler, type TtmPositional } from './ttm';

/*
 * fix ttm/thermal — the two-temperature model with electronic properties set per grid cell
 * (docs.lammps.org/fix_ttm.html, section "Additional details for fix ttm/thermal"; plans/lammps-docs-live/
 * fix_ttm.rst). It is fix ttm (fix_ttm.ts, the base class) with four changes: the properties come from a
 * file, there is no electronic stopping, a *source* keyword adds heating to the electrons, and cells
 * without electrons are allowed.
 *
 * Syntax: fix ID group-ID ttm/thermal seed properties_file Nx Ny Nz keyword value ...; keywords set,
 * infile, outfile (as fix ttm) and source.
 *
 * Quoted doc sentences (docs.lammps.org/fix_ttm.html):
 *  "Fix *ttm/thermal* uses the heat diffusion equation with possible external heat sources (e.g. inductive heating)."
 *  "The effects of electron stopping have been removed:"
 *  "Also note that compared to the original *fix ttm*, it uses use a volumetric specific heat, :math:`C_\mathrm{vol}` , which represents the product of :math:`C_e \rho_e`."
 *  "The current fix *ttm/thermal* implementation allows TTM simulations with TTM cells that do not contain electrons (vacuum or insulators)."
 *  "The numerical scheme does not allow energy exchange with such cells."
 *  "The fix *ttm/thermal* parameter file *properties_file* uses a similar syntax as the keyword *infile*."
 *  "Comment lines are allowed and each line should have properties listed in the order:"
 *  "the grid must match the one declared in the fix and all grid points must have all properties set or *ttm/thermal* will exit with an error."
 *
 * Model (doc): C_vol dT_e/dt = div(kappa_eff grad T_e) - g_p (T_e - T_a) + eta s, with kappa_eff the
 * harmonic mean of the two cells' conductivities, C_vol, kappa_e, gamma_p and eta per cell (the
 * properties file columns "ix iy iz C_vol kappa_e gamma_p eta"), s the source.
 *
 * Measured with native LAMMPS (black box, eight-atom 2x2x2 runs, thermo f_2 and the outfile):
 *  - the flow between two cells uses the harmonic mean 2 k_a k_b / (k_a + k_b): with four different
 *    conductivities on a 2x2x2 grid the native grid matches to 5e-13 absolute (the arithmetic mean and
 *    the cell's own conductivity differ by 5 and 11 kelvin); a 4x4x4 grid with random per-cell
 *    properties, four sub-steps per step, matches in tests/oracle/w41ttm_thermal4.in;
 *  - the update divides the flow by the cell's own C_vol: C_vol 0.5 .. 3 per cell matches to 5e-13;
 *  - the energy the cells hold is C_vol V_cell T per cell (f_2[1] = 216000 = 8 cells * 27 * 1000 for C_vol 1);
 *  - the source adds dT = eta s dt / C_vol per cell (eta scales it: 0.5 and 0.25 match to 5e-13); it
 *    enters the electron energy (f_2[1] rises by source V dt in each cell) and not the atom transfer;
 *  - the sub-step count of the explicit diffusion uses the largest kappa/C_vol over all cells (vacuum
 *    cells included): with kappa/C_vol = 8 in one cell and 0.1 / C_vol 1 elsewhere, dt 0.5 is split into 3
 *    (the native grid matches; 2 and 4 sub-steps do not); with kappa/C_vol = 0.4 and 4 in two cells, the
 *    count is 2 (the largest kappa over the smallest C_vol would give 3, and does not match);
 *  - a cell with T_e = 0 is a cell without electrons: its T stays 0 under diffusion, the source and
 *    the coupling, and no flow passes between it and a neighbour (the grid matches when the flow terms
 *    through the empty cell are dropped, to 1e-12); an empty cell contributes nothing to f_2[1];
 *  - an atom in a cell without electrons gets no force (friction or random) and draws no random
 *    number: the next atom takes the draws the empty-cell atom would have used (the stream shifts by
 *    exactly one atom's three draws);
 *  - gamma_p = 0 is accepted for a cell (no error);
 *  - kappa_e = 0 is accepted; kappa_e < 0, C_vol < 0, gamma_p < 0 and eta < 0 stop the run with an error;
 *  - C_vol = 0 is accepted by native LAMMPS but freezes the whole grid (the diffusion step is never
 *    applied, measured over three steps); it is refused here, as is C_vol 1e-300 (the same freeze);
 *  - a properties line with fewer than seven numbers stops the run; extra words after the seventh are ignored;
 *  - a UNITS: tag on the first line of the file must match the run's units (a tag on any other line is not read);
 *  - the default grid (no set, no infile) is all zero electrons (no error), so no coupling and no energy;
 *  - the outfile of ttm/thermal is the ttm outfile (header: created by fix ttm/thermal).
 * Duplicate grid entries in the properties file are refused: the native result for them is not
 * reproduced (the value kept depends on the order of the lines).
 */

export interface ThermalPositional extends TtmPositional {
  propsFile: string;
}

/** The positional part: seed properties_file Nx Ny Nz. */
export const parseThermalPositional = (id: string, args: string[]): ThermalPositional => {
  const style = 'ttm/thermal';
  // As native LAMMPS does (src/EXTRA-FIX/fix_ttm_thermal.cpp at b332f28)
  if (args.length < 5) throw missingArgs('fix ttm/thermal');
  const seed = Number(args[0]);
  if (!Number.isInteger(seed) || seed < 1) throw new StyleError(`fix ${id} ${style}: seed must be a positive integer, got '${args[0]}'`);
  const dim = (w: string, what: string): number => {
    const v = Number(w);
    if (!Number.isInteger(v) || v < 1) throw new StyleError(`fix ${id} ${style}: ${what} must be a positive integer, got '${w}'`);
    return v;
  };
  return {
    seed,
    propsFile: args[1],
    nx: dim(args[2], 'Nx'),
    ny: dim(args[3], 'Ny'),
    nz: dim(args[4], 'Nz'),
    consumed: 5,
    scalar: null,
  };
};

export interface ThermalCells {
  cvol: Float64Array;
  kappa: Float64Array;
  gammaP: Float64Array;
  eta: Float64Array;
}

/**
 * Reads the properties file: "ix iy iz C_vol kappa_e gamma_p eta" per grid point, x fastest in the result.
 * Comments ('#') and empty lines are skipped; every grid point must be set exactly once.
 */
export const parseThermalProps = (text: string, id: string, nx: number, ny: number, nz: number, units: string, fname: string): ThermalCells => {
  const lines = text.split('\n');
  const tag = /UNITS:\s*(\S+)/.exec(lines[0] ?? '');
  if (tag && tag[1] !== units) {
    throw new StyleError(`fix ${id} ttm/thermal: properties file ${fname} was written for units ${tag[1]} but ${units} units are in use`);
  }
  const n = nx * ny * nz;
  const out: ThermalCells = { cvol: new Float64Array(n), kappa: new Float64Array(n), gammaP: new Float64Array(n), eta: new Float64Array(n) };
  const seen = new Uint8Array(n);
  for (const raw of lines) {
    const line = raw.replace(/#.*/, '').trim();
    if (!line) continue;
    const w = line.split(/\s+/);
    if (w.length < 7) throw new StyleError(`fix ${id} ttm/thermal: properties file ${fname} line needs ix iy iz C_vol kappa_e gamma_p eta: '${raw.trim()}'`);
    const [ix, iy, iz, cvol, kappa, gp, eta] = w.slice(0, 7).map(Number);
    if (![ix, iy, iz, cvol, kappa, gp, eta].every(Number.isFinite)) throw new StyleError(`fix ${id} ttm/thermal: properties file ${fname} has a non-numeric entry: '${raw.trim()}'`);
    if (!Number.isInteger(ix) || !Number.isInteger(iy) || !Number.isInteger(iz) || ix < 1 || ix > nx || iy < 1 || iy > ny || iz < 1 || iz > nz) {
      throw new StyleError(`fix ${id} ttm/thermal: properties file ${fname} grid index out of range: '${raw.trim()}'`);
    }
    // Measured with native LAMMPS (black box): C_vol < 0, kappa_e < 0, gamma_p < 0 and eta < 0 each stop the run with an error.
    if (cvol === 0) throw new StyleError(`fix ${id} ttm/thermal: C_vol = 0 at ${ix} ${iy} ${iz} is not reproduced (native LAMMPS freezes the whole grid); use T_e = 0 for a cell without electrons`);
    if (!(cvol > 0)) throw new StyleError(`fix ${id} ttm/thermal: electron specific heat in ${fname} must be > 0`);
    if (!(kappa >= 0)) throw new StyleError(`fix ${id} ttm/thermal: electron conductivity in ${fname} must be >= 0`);
    if (!(gp >= 0)) throw new StyleError(`fix ${id} ttm/thermal: electron coupling in ${fname} must be >= 0`);
    if (!(eta >= 0)) throw new StyleError(`fix ${id} ttm/thermal: electron inductive response in ${fname} must be >= 0`);
    const c = (ix - 1) + nx * ((iy - 1) + ny * (iz - 1));
    if (seen[c]) throw new StyleError(`fix ${id} ttm/thermal: properties file ${fname} sets grid point ${ix} ${iy} ${iz} twice`);
    seen[c] = 1;
    out.cvol[c] = cvol; out.kappa[c] = kappa; out.gammaP[c] = gp; out.eta[c] = eta;
  }
  for (let c = 0; c < n; c++) {
    if (!seen[c]) throw new StyleError(`fix ${id} ttm/thermal: properties file ${fname} did not set all ${nx}x${ny}x${nz} grid points`);
  }
  return out;
};

/** fix ttm/thermal: per-cell properties from the properties file; no stopping; source keyword. */
export class FixTTMThermal extends FixTTM {
  private cells!: ThermalCells;
  private source!: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    const pos = parseThermalPositional(id, args);
    const kw = { source: null as number | null };
    // Measured with native LAMMPS (black box): the source keyword takes one value (negative values are accepted).
    const handler: TtmKeywordHandler = (key, a, i) => {
      if (key !== 'source') return 0;
      const v = Number(a[i + 1]);
      if (a[i + 1] === undefined || !Number.isFinite(v)) throw new StyleError(`fix ${id} ttm/thermal: keyword source needs a value (energy/(time*volume))`);
      kw.source = v;
      return 2;
    };
    super(sys, id, group, args, 'ttm/thermal', pos, handler);
    this.source = kw.source ?? 0;
    this.cells = parseThermalProps(sys.readFile(pos.propsFile), id, this.p.nx, this.p.ny, this.p.nz, sys.state.units.style, pos.propsFile);
  }

  protected cvolAt(c: number, _T: number): number {
    return this.cells.cvol[c];
  }

  protected kappaAt(c: number, _T: number): number {
    return this.cells.kappa[c];
  }

  /** Harmonic mean of the two conductivities (doc: kappa_eff = 2 kappa_a kappa_b / (kappa_a + kappa_b)). */
  protected pairKappa(kc: number, kj: number): number {
    return kc + kj > 0 ? (2 * kc * kj) / (kc + kj) : 0;
  }

  protected gammaPFor(c: number, _speed: number): number {
    return this.cells.gammaP[c];
  }

  /** No electronic stopping in ttm/thermal (the doc: "The effects of electron stopping have been removed"). */
  protected gammaSFor(_speed: number): number {
    return 0;
  }

  protected vacuum(c: number): boolean {
    return this.Te[c] === 0;
  }

  protected sourceAt(c: number): number {
    return this.cells.eta[c] * this.source;
  }
}
