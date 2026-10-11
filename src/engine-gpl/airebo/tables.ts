/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_airebo.cpp (PairAIREBO::read_file and the spline
 * array layout of pair_airebo.h), src/MANYBODY/pair_airebo_morse.cpp and
 * src/MANYBODY/pair_rebo.cpp (variant headers), at commit b332f28.
 * The reader is token based: LAMMPS reads one value per line for the scalars and
 * the spline counts; every spline value in the potential files sits on its own line.
 * Authors credited in the C++ file headers: Ase Henry (MIT); bugfixes and optimizations:
 * Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer (Temple U), Markus Hoehnerbach
 * (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M: Thomas C. O'Connor (JHU) 2014.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';

export type AireboVariant = 'airebo' | 'rebo' | 'airebo/morse';

/** Spline coefficient tables of read_file (flat, row-major as the C++ arrays). */
export class SplineTables {
  gCdom = new Float64Array(5);
  /** gC1[4][6], gC2[4][6] */
  gC1 = new Float64Array(24);
  gC2 = new Float64Array(24);
  gHdom = new Float64Array(4);
  /** gH[3][6] */
  gH = new Float64Array(18);
  /** pCCdom[2][2], pCHdom[2][2] */
  pCCdom = new Float64Array(4);
  pCHdom = new Float64Array(4);
  /** pCC[4][4][16], pCH[4][4][16] */
  pCC = new Float64Array(256);
  pCH = new Float64Array(256);
  /** piCCdom[3][2], piCHdom[3][2], piHHdom[3][2] */
  piCCdom = new Float64Array(6);
  piCHdom = new Float64Array(6);
  piHHdom = new Float64Array(6);
  /** piCC[4][4][9][64], piCH, piHH, Tijc */
  piCC = new Float64Array(9216);
  piCH = new Float64Array(9216);
  piHH = new Float64Array(9216);
  /** Tijdom[3][2] */
  Tijdom = new Float64Array(6);
  Tijc = new Float64Array(9216);
}

/** Scalar parameters of read_file; element pairs are indexed [ii*2+jj] with 0 = C, 1 = H. */
export interface AireboParams {
  smin: number; Nmin: number; Nmax: number; NCmin: number; NCmax: number;
  thmin: number; thmax: number;
  rcmin: Float64Array; rcmax: Float64Array; rcmaxsq: Float64Array; rcmaxp: Float64Array;
  Q: Float64Array; alpha: Float64Array; A: Float64Array; rho: Float64Array;
  /** BIJc[ii][jj][m] and Beta[ii][jj][m] at [(ii*2+jj)*3+m] */
  BIJc: Float64Array; Beta: Float64Array;
  rcLJmin: Float64Array; rcLJmax: Float64Array; rcLJmaxsq: Float64Array;
  bLJmin: Float64Array; bLJmax: Float64Array;
  epsilon: Float64Array; sigma: Float64Array; epsilonT: Float64Array;
  /** Morse variant only (zeros otherwise). */
  epsilonM: Float64Array; alphaM: Float64Array; reqM: Float64Array;
}

export const newParams = (): AireboParams => {
  const a4 = () => new Float64Array(4);
  return {
    smin: 0, Nmin: 0, Nmax: 0, NCmin: 0, NCmax: 0, thmin: 0, thmax: 0,
    rcmin: a4(), rcmax: a4(), rcmaxsq: a4(), rcmaxp: a4(),
    Q: a4(), alpha: a4(), A: a4(), rho: a4(),
    BIJc: new Float64Array(12), Beta: new Float64Array(12),
    rcLJmin: a4(), rcLJmax: a4(), rcLJmaxsq: a4(), bLJmin: a4(), bLJmax: a4(),
    epsilon: a4(), sigma: a4(), epsilonT: a4(),
    epsilonM: a4(), alphaM: a4(), reqM: a4(),
  };
};

/**
 * The values of a potential file after the first two header lines: one value per line, the first
 * word of each non-empty line (the parameter lines carry the symbol name after the value; the
 * LAMMPS reader takes one line per value, PotentialFileReader::next_double).
 */
const valueWords = (text: string): string[] => {
  const lines = text.split(/\r?\n/);
  const out: string[] = [];
  for (let n = 2; n < lines.length; n++) {
    const hash = lines[n].indexOf('#');
    const body = (hash >= 0 ? lines[n].slice(0, hash) : lines[n]).trim();
    if (body === '') continue;
    out.push(body.split(/\s+/)[0]);
  }
  return out;
};

/**
 * PairAIREBO::read_file. The second line must carry the variant header (the C++ checks
 * "# AIREBO ", "# REBO2 " or "# AIREBO-M "). Returns the parameters and the spline tables.
 */
export const readAireboFile = (
  text: string, fileName: string, variant: AireboVariant, morseflag: boolean,
): { params: AireboParams; tables: SplineTables } => {
  const header = variant === 'rebo' ? '# REBO2 ' : variant === 'airebo/morse' ? '# AIREBO-M ' : '# AIREBO ';
  const line2 = text.split(/\r?\n/)[1] ?? '';
  if (!line2.includes(header)) {
    throw new StyleError(`Potential file ${fileName} does not match AIREBO/REBO style variant: ${header}: ${line2}`);
  }
  const tok = valueWords(text);
  let p = 0;
  const bad = (what: string): never => {
    throw new StyleError(`reading ${what} section in ${fileName} file: unexpected end of file or bad number`);
  };
  const num = (what: string): number => {
    if (p >= tok.length) bad(what);
    const v = Number(tok[p++]);
    if (!Number.isFinite(v)) bad(what);
    return v;
  };
  const int = (what: string): number => {
    const v = num(what);
    if (!Number.isInteger(v)) bad(what);
    return v;
  };
  const vec = (dest: Float64Array, off: number, n: number, what: string): void => {
    for (let k = 0; k < n; k++) dest[off + k] = num(what);
  };
  const P = newParams();
  const T = new SplineTables();
  // global parameters, one per line in the file order of the C++ params vector
  const cch = (): [number, number, number] => [num('global parameters'), num('global parameters'), num('global parameters')];
  const rcmin3 = cch(), rcmax3 = cch(), rcmaxp3 = cch();
  P.smin = num('global parameters');
  P.Nmin = num('global parameters');
  P.Nmax = num('global parameters');
  P.NCmin = num('global parameters');
  P.NCmax = num('global parameters');
  // CC, CH, HH triples: stored into the symmetric [2][2] arrays below
  const Q3 = cch(), alpha3 = cch(), A3 = cch();
  const BIJ9 = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(() => num('global parameters'));
  const Beta9 = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(() => num('global parameters'));
  const rho3 = cch();
  const rcLJmin3 = cch(), rcLJmax3 = cch(), bLJmin3 = cch(), bLJmax3 = cch();
  const eps3 = cch(), sig3 = cch(), epsT3 = cch();
  let morse: { eM: [number, number, number]; aM: [number, number, number]; rM: [number, number, number] } | null = null;
  if (morseflag) morse = { eM: cch(), aM: cch(), rM: cch() };
  // symmetric [2][2] fill: index (ii*2+jj) with C=0, H=1: CC = [0,0], CH = [0,1]=[1,0], HH = [1,1]
  const sym = (dest: Float64Array, v: [number, number, number]): void => {
    dest[0] = v[0]; dest[1] = v[1]; dest[2] = v[1]; dest[3] = v[2];
  };
  sym(P.rcmin, rcmin3);
  sym(P.rcmax, rcmax3);
  sym(P.rcmaxp, rcmaxp3);
  // thmin and thmax are fixed in the C++ read_file
  P.thmin = -1.0;
  P.thmax = -0.995;
  for (let ii = 0; ii < 2; ii++) {
    for (let jj = 0; jj < 2; jj++) P.rcmaxsq[ii * 2 + jj] = P.rcmax[ii * 2 + jj] * P.rcmax[ii * 2 + jj];
  }
  sym(P.Q, Q3);
  sym(P.alpha, alpha3);
  sym(P.A, A3);
  sym(P.rho, rho3);
  sym(P.rcLJmin, rcLJmin3);
  sym(P.rcLJmax, rcLJmax3);
  for (let k = 0; k < 4; k++) P.rcLJmaxsq[k] = P.rcLJmax[k] * P.rcLJmax[k];
  sym(P.bLJmin, bLJmin3);
  sym(P.bLJmax, bLJmax3);
  sym(P.epsilon, eps3);
  sym(P.sigma, sig3);
  sym(P.epsilonT, epsT3);
  // BIJc and Beta: [CC][m], [CH][m] (= [HC][m]), [HH][m]
  for (let m = 0; m < 3; m++) {
    P.BIJc[(0 * 2 + 0) * 3 + m] = BIJ9[m];
    P.BIJc[(0 * 2 + 1) * 3 + m] = BIJ9[3 + m];
    P.BIJc[(1 * 2 + 0) * 3 + m] = BIJ9[3 + m];
    P.BIJc[(1 * 2 + 1) * 3 + m] = BIJ9[6 + m];
    P.Beta[(0 * 2 + 0) * 3 + m] = Beta9[m];
    P.Beta[(0 * 2 + 1) * 3 + m] = Beta9[3 + m];
    P.Beta[(1 * 2 + 0) * 3 + m] = Beta9[3 + m];
    P.Beta[(1 * 2 + 1) * 3 + m] = Beta9[6 + m];
  }
  if (morse) {
    sym(P.epsilonM, morse.eM);
    sym(P.alphaM, morse.aM);
    sym(P.reqM, morse.rM);
  }
  // gC spline: number of domain points, then the domain, then gC1 and gC2 per interval
  let limit = int('gC spline');
  vec(T.gCdom, 0, limit, 'gC spline');
  for (let i = 0; i < limit - 1; i++) vec(T.gC1, i * 6, 6, 'gC spline');
  for (let i = 0; i < limit - 1; i++) vec(T.gC2, i * 6, 6, 'gC spline');
  // gH spline
  limit = int('gH spline');
  vec(T.gHdom, 0, limit, 'gH spline');
  for (let i = 0; i < limit - 1; i++) vec(T.gH, i * 6, 6, 'gH spline');
  // pCC spline: domains (2 rows of limit/2), then the 16-value patches
  limit = int('pCC spline');
  for (let i = 0; i < Math.trunc(limit / 2); i++) vec(T.pCCdom, i * 2, Math.trunc(limit / 2), 'pCC spline');
  for (let i = 0; i < Math.trunc(T.pCCdom[1]); i++) {
    for (let j = 0; j < Math.trunc(T.pCCdom[3]); j++) vec(T.pCC, (i * 4 + j) * 16, 16, 'pCC spline');
  }
  // pCH spline
  limit = int('pCH spline');
  for (let i = 0; i < Math.trunc(limit / 2); i++) {
    for (let j = 0; j < Math.trunc(limit / 2); j++) T.pCHdom[i * 2 + j] = num('pCH spline');
  }
  for (let i = 0; i < Math.trunc(T.pCHdom[1]); i++) {
    for (let j = 0; j < Math.trunc(T.pCHdom[3]); j++) vec(T.pCH, (i * 4 + j) * 16, 16, 'pCH spline');
  }
  // piCC, piCH, piHH and Tij: domains (3 rows of 2) and 64-value patches
  const readConj = (dom: Float64Array, dest: Float64Array, what: string): void => {
    const lim = int(what);
    for (let i = 0; i < Math.trunc(lim / 2); i++) {
      for (let j = 0; j < Math.trunc(lim / 3); j++) dom[i * 2 + j] = num(what);
    }
    for (let i = 0; i < Math.trunc(dom[1]); i++) {
      for (let j = 0; j < Math.trunc(dom[3]); j++) {
        for (let k = 0; k < Math.trunc(dom[5]); k++) vec(dest, ((i * 4 + j) * 9 + k) * 64, 64, what);
      }
    }
  };
  readConj(T.piCCdom, T.piCC, 'piCC spline');
  readConj(T.piCHdom, T.piCH, 'piCH spline');
  readConj(T.piHHdom, T.piHH, 'piHH spline');
  readConj(T.Tijdom, T.Tijc, 'Tij spline');
  return { params: P, tables: T };
};
