import { StyleError } from '../force/types';

/*
 * fix ttm/mod — the parameter file is parsed here (docs.lammps.org/fix_ttm.html,
 * plans/lammps-docs/fix_ttm.rst). The style itself is NOT registered: its
 * physics is not reproduced (see the report of wave 14), so an input that uses
 * fix ttm/mod gets the engine's unsupported-style StyleError.
 *
 * Syntax: "fix ID group-ID ttm/mod seed init_file Nx Ny Nz keyword value ...".
 * Parameter file (doc): "Every line with an odd number is considered as a
 * comment and ignored. The lines with the even numbers are treated as follows:"
 * — the 22 values a_0 .. a_4, C_0, A, rho_e, D_e, gamma_p, gamma_s, v_0, I_0,
 * lsurface, rsurface, l_skin, tau, B, lambda, n_ion, surface_movement, T_e_min.
 *
 * Measured with native LAMMPS (black box, synthetic files):
 *  - lsurface, rsurface, l_skin and surface_movement are read as integers:
 *    a value such as 1.0 or 0.5 in those slots stops the run with an integer-parse error;
 *  - text after the value on an even line is accepted;
 *  - D_e = 0 is accepted; the native diffusion with D_e > 0 on a 4-cell grid
 *    does not follow the explicit FTCS update of fix ttm (cell-dependent
 *    deviation 0.98 .. 0.979 of the ttm step even for constant C_e), and a
 *    single atom in a ttm/mod run gets no coupling force, while an eight-atom
 *    run has a non-zero electron-atom energy transfer with zero per-atom force
 *    on the dump: the coupling of ttm/mod is not the ttm one.
 */

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
 * The parts of a fix ttm/mod input this engine cannot reproduce (empty = none). The style itself is not
 * registered, so an input that uses it gets the engine's unsupported fix style StyleError.
 */
export const ttmModBlockers = (p: TtmModParams, nx: number): string[] => {
  const out: string[] = [];
  if (p.De !== 0) out.push('D_e > 0 (electron heat diffusion)');
  if (p.I0 !== 0) out.push('I_0 != 0 (laser source)');
  if (p.B !== 0) out.push('B != 0 (electronic pressure force)');
  if (p.surfaceMovement !== 0) out.push('surface_movement 1 (moving vacuum surface)');
  if (p.lsurface !== 1 || p.rsurface !== nx) out.push('lsurface/rsurface other than the whole grid (vacuum cells)');
  return out;
};
