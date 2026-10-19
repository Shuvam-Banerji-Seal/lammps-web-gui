/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the parameter tables of
 * src/EXTRA-PAIR/d3_parameters.h (contributing authors Sonia Salomoni and Arthur France-Lanord, IMPMC, CNRS, Sorbonne
 * Universite; copyright Sandia Corporation): r2r4_ref, rcov_ref, r0ab_table and c6ab_table. This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The C6 table (32385 rows of 5 numbers) is too large for one lazily loaded chunk, so its rows are split over the modules
 * c6ab_0.ts ... c6ab_6.ts, imported on first use of pair_style dispersion/d3 (loadD3Tables). The numbers are kept as
 * decimal text with the native literals' exact values, so Number() gives the doubles the C++ compiler gives.
 */

/** dispersion/d3: NUM_ELEMENTS, N_PARS_ROWS, N_PARS_COLS (pair_dispersion_d3.cpp). */
export const NUM_ELEMENTS = 94;
export const N_PARS_ROWS = 32385;
export const N_PARS_COLS = 5;

export interface D3Tables {
  /** r2r4_ref[NUM_ELEMENTS + 1] */
  r2r4: Float64Array;
  /** rcov_ref[NUM_ELEMENTS + 1] */
  rcov: Float64Array;
  /** r0ab_table[NUM_ELEMENTS][NUM_ELEMENTS], row-major */
  r0ab: Float64Array;
  /** c6ab_table[N_PARS_ROWS][N_PARS_COLS], row-major */
  c6ab: Float64Array;
}

let tables: D3Tables | null = null;

const parse = (text: string, out: Float64Array, offset = 0): number => {
  let n = offset;
  for (const w of text.split(/\s+/)) if (w) out[n++] = Number(w);
  return n;
};

/** The tables, once loadD3Tables has run (the registry's loader awaits it before the style can be made). */
export const d3Tables = (): D3Tables => {
  if (!tables) throw new Error('pair dispersion/d3: the parameter tables are not loaded');
  return tables;
};

/** Imports the data modules and parses them (once). */
export const loadD3Tables = async (): Promise<void> => {
  if (tables) return;
  const [ref, ...parts] = await Promise.all([
    import('./d3_reference'),
    import('./c6ab_0'), import('./c6ab_1'), import('./c6ab_2'), import('./c6ab_3'),
    import('./c6ab_4'), import('./c6ab_5'), import('./c6ab_6'),
  ]);
  const r2r4 = new Float64Array(NUM_ELEMENTS + 1);
  const rcov = new Float64Array(NUM_ELEMENTS + 1);
  const r0ab = new Float64Array(NUM_ELEMENTS * NUM_ELEMENTS);
  const c6ab = new Float64Array(N_PARS_ROWS * N_PARS_COLS);
  if (parse(ref.R2R4, r2r4) !== r2r4.length || parse(ref.RCOV, rcov) !== rcov.length || parse(ref.R0AB, r0ab) !== r0ab.length) {
    throw new Error('pair dispersion/d3: reference table has the wrong length');
  }
  let n = 0;
  for (const p of parts) n = parse(p.ROWS, c6ab, n);
  if (n !== c6ab.length) throw new Error(`pair dispersion/d3: C6 table has ${n} numbers, expected ${c6ab.length}`);
  tables = { r2r4, rcov, r0ab, c6ab };
};
