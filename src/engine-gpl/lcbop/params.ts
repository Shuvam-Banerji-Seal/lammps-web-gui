/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_lcbop.cpp (PairLCBOP::read_file and PairLCBOP::spline_init) and
 * src/MANYBODY/pair_lcbop.h (the parameter and table layout), at commit b332f28.
 * Authors credited in the C++ file header: Dominik Wojt (Wroclaw University of Technology),
 * based on pair_airebo by Ase Henry (MIT). Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { StyleError, type StyleContext } from '../../engine/force/types';
import { PotentialEntryReader, nextDouble, potentialUnitFactor } from '../manybody2/common';

/** The 28 scalar parameters of read_file, in the order of the potential file. */
export interface LcbopParams {
  r_1: number; r_2: number; gamma_1: number; A: number; B_1: number; B_2: number; alpha: number;
  beta_1: number; beta_2: number; d: number; C_1: number; C_4: number; C_6: number; L: number;
  kappa: number; R_0: number; R_1: number; r_0: number; r_1_LR: number; r_2_LR: number;
  v_1: number; v_2: number; eps_1: number; eps_2: number; lambda_1: number; lambda_2: number;
  eps: number; delta: number;
}

/** The F_conj bicubic field of one (N_ij, N_ji, N_conj) cell (pair_lcbop.h TF_conj_field). */
export interface TfConj {
  f_00: number; f_01: number; f_10: number; f_11: number;
  f_x_00: number; f_x_01: number; f_x_10: number; f_x_11: number;
  f_y_00: number; f_y_01: number; f_y_10: number; f_y_11: number;
}

/** Spline tables: gX[6] mesh points, gC[power][interval] (flat, gC[power * 5 + interval]), F_conj_field[N_ij][N_ji][N_conj]. */
export interface LcbopTables {
  gX: Float64Array;
  gC: Float64Array;
  F_conj_field: TfConj[][][];
}

const EMPTY_TF = (): TfConj => ({
  f_00: 0, f_01: 0, f_10: 0, f_11: 0, f_x_00: 0, f_x_01: 0, f_x_10: 0, f_x_11: 0,
  f_y_00: 0, f_y_01: 0, f_y_10: 0, f_y_11: 0,
});

export const newParams = (): LcbopParams => ({
  r_1: 0, r_2: 0, gamma_1: 0, A: 0, B_1: 0, B_2: 0, alpha: 0, beta_1: 0, beta_2: 0, d: 0, C_1: 0, C_4: 0, C_6: 0,
  L: 0, kappa: 0, R_0: 0, R_1: 0, r_0: 0, r_1_LR: 0, r_2_LR: 0, v_1: 0, v_2: 0, eps_1: 0, eps_2: 0,
  lambda_1: 0, lambda_2: 0, eps: 0, delta: 0,
});

/**
 * PairLCBOP::read_file. Values are read in the file order: 28 scalars (one per line), the F_conj
 * data (2 values of N_conj x 3 kinds of data x 4 rows of 4 values), the 6 G mesh points and the 6 x 5
 * G coefficients. Any read error is reported as native: "Error reading LCBOP potential file <name>: <what>".
 */
export const readLcbopFile = (
  text: string, fileName: string, ctx: StyleContext,
): { params: LcbopParams; gX: Float64Array; gC: Float64Array; F_data: Float64Array } => {
  const p = newParams();
  const gX = new Float64Array(6);
  const gC = new Float64Array(6 * 5);
  // F_conj_data[i][j][k][l] stored at ((i * 4 + j) * 2 + k) * 3 + l
  const F_data = new Float64Array(4 * 4 * 2 * 3);
  try {
    // the units check of PotentialFileReader (auto_convert 0): a mismatch is an error
    potentialUnitFactor(text, 'LCBOP', fileName, ctx, false);
    const rd = new PotentialEntryReader(text.split(/\r?\n/), 'LCBOP');
    const values = (n: number): string[] => {
      const w = rd.next(n);
      if (w === null) throw new StyleError('Missing line in LCBOP file!');
      return w;
    };
    const scalars: (keyof LcbopParams)[] = [
      'r_1', 'r_2', 'gamma_1', 'A', 'B_1', 'B_2', 'alpha', 'beta_1', 'beta_2', 'd', 'C_1', 'C_4', 'C_6', 'L',
      'kappa', 'R_0', 'R_1', 'r_0', 'r_1_LR', 'r_2_LR', 'v_1', 'v_2', 'eps_1', 'eps_2', 'lambda_1', 'lambda_2',
      'eps', 'delta',
    ];
    for (const key of scalars) p[key] = nextDouble(values(1)[0]);
    for (let k = 0; k < 2; k++) {
      for (let l = 0; l < 3; l++) {
        for (let i = 0; i < 4; i++) {
          const w = values(4);
          for (let j = 0; j < 4; j++) F_data[((i * 4 + j) * 2 + k) * 3 + l] = nextDouble(w[j]);
        }
      }
    }
    const gx = values(6);
    for (let i = 0; i < 6; i++) gX[i] = nextDouble(gx[i]);
    for (let i = 0; i < 6; i++) {
      const w = values(5);
      for (let j = 0; j < 5; j++) gC[i * 5 + j] = nextDouble(w[j]);
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new StyleError(`Error reading LCBOP potential file ${fileName}: ${msg}`);
  }
  return { params: p, gX, gC, F_data };
};

/** PairLCBOP::spline_init: the 2 x 3 x 3 bicubic F_conj fields from the 4 x 4 data. */
export const splineInit = (F_data: Float64Array): TfConj[][][] => {
  const D = (i: number, j: number, k: number, l: number): number => F_data[((i * 4 + j) * 2 + k) * 3 + l];
  const F: TfConj[][][] = [];
  for (let a = 0; a < 3; a++) {
    F.push([]);
    for (let b = 0; b < 3; b++) {
      F[a].push([EMPTY_TF(), EMPTY_TF()]);
    }
  }
  for (let N_conj_ij = 0; N_conj_ij < 2; N_conj_ij++) {
    for (let N_ij = 0; N_ij < 4 - 1; N_ij++) {
      for (let N_ji = 0; N_ji < 4 - 1; N_ji++) {
        const field = F[N_ij][N_ji][N_conj_ij];
        field.f_00 = D(N_ij, N_ji, N_conj_ij, 0);
        field.f_01 = D(N_ij, N_ji + 1, N_conj_ij, 0);
        field.f_10 = D(N_ij + 1, N_ji, N_conj_ij, 0);
        field.f_11 = D(N_ij + 1, N_ji + 1, N_conj_ij, 0);
        field.f_x_00 = D(N_ij, N_ji, N_conj_ij, 1) - field.f_10 + field.f_00;
        field.f_x_01 = D(N_ij, N_ji + 1, N_conj_ij, 1) - field.f_11 + field.f_01;
        field.f_x_10 = -(D(N_ij + 1, N_ji, N_conj_ij, 1) - field.f_10 + field.f_00);
        field.f_x_11 = -(D(N_ij + 1, N_ji + 1, N_conj_ij, 1) - field.f_11 + field.f_01);
        field.f_y_00 = D(N_ij, N_ji, N_conj_ij, 2) - field.f_01 + field.f_00;
        field.f_y_01 = -(D(N_ij, N_ji + 1, N_conj_ij, 2) - field.f_01 + field.f_00);
        field.f_y_10 = D(N_ij + 1, N_ji, N_conj_ij, 2) - field.f_11 + field.f_10;
        field.f_y_11 = -(D(N_ij + 1, N_ji + 1, N_conj_ij, 2) - field.f_11 + field.f_10);
      }
    }
  }
  return F;
};
