/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_comb.cpp and src/MANYBODY/pair_comb.h (PairComb::read_file,
 * setup_params, the Param layout and the cutoff functions comb_fc, comb_fc2, comb_fc3 and their
 * derivatives), at commit b332f28. Authors credited in the C++ file headers: Tzu-Ray Shan (U Florida,
 * then Sandia), based on the HELL MD program of Simon Phillpot (UF) and Aidan Thompson's Tersoff code
 * in LAMMPS. Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';
import { joinPotentialEntries } from '../../engine/force/util';
import { log as libmLog } from '../libm/log';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** Number of words in one COMB potential entry (PairComb::NPARAMS_PER_LINE). */
export const NPARAMS_PER_LINE = 49;

/** Per (i,j,k) parameter set, C++ field names (pair_comb.h struct Param). */
export interface CombParam {
  ielement: number; jelement: number; kelement: number;
  lam11: number; lam12: number; lam21: number; lam22: number;
  c: number; d: number; h: number;
  gamma: number; powerm: number;
  powern: number; beta: number;
  biga1: number; biga2: number; bigb1: number; bigb2: number;
  bigd: number; bigr: number;
  cut: number; cutsq: number;
  c1: number; c2: number; c3: number; c4: number;
  plp1: number; plp3: number; plp6: number; a123: number; aconf: number;
  rlm1: number; rlm2: number;
  romiga: number; romigb: number; romigc: number; romigd: number; addrep: number;
  QU1: number; QL1: number; DU1: number; DL1: number; Qo1: number; dQ1: number; aB1: number; bB1: number; nD1: number; bD1: number;
  QU2: number; QL2: number; DU2: number; DL2: number; Qo2: number; dQ2: number; aB2: number; bB2: number; nD2: number; bD2: number;
  chi: number; dj: number; dk: number; dl: number; dm: number; esm1: number;
  cmn1: number; cmn2: number; cml1: number; cml2: number;
  coulcut: number; lcut: number; lcutsq: number; hfocor: number;
  powermint: number;
}

/** pair_comb.cpp powint: integer power by binary exponentiation (MathSpecial::powint). */
export const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

export const cube = (x: number): number => x * x * x;
export const square = (x: number): number => x * x;

const MY_PI = Math.PI;
const MY_PI2 = Math.PI / 2;
const MY_PI4 = Math.PI / 4;

/** Entry field order of one potential line after the three element names (pair_comb.cpp read_file). */
const FIELDS = [
  'powerm', 'c', 'd', 'h', 'powern', 'beta', 'lam21', 'lam22', 'bigb1', 'bigb2', 'bigr', 'bigd',
  'lam11', 'lam12', 'biga1', 'biga2', 'plp1', 'plp3', 'plp6', 'a123', 'aconf', 'addrep',
  'romigb', 'romigc', 'romigd', 'romiga', 'QL1', 'QU1', 'DL1', 'DU1', 'QL2', 'QU2', 'DL2', 'DU2',
  'chi', 'dj', 'dk', 'dl', 'dm', 'esm1', 'cmn1', 'cml1', 'cmn2', 'cml2', 'coulcut', 'hfocor',
] as const;

/**
 * Reads the entries whose three element names are all in `elements` (other lines are skipped, as in
 * read_file). Every field is checked with the same sanity rules and messages as the LAMMPS reader.
 */
export const readCombEntries = (text: string, elements: readonly string[]): CombParam[] => {
  const lines = joinPotentialEntries(text.split(/\r?\n/), NPARAMS_PER_LINE);
  const out: CombParam[] = [];
  for (const line of lines) {
    const w = line.trim() ? line.trim().split(/\s+/) : [];
    if (!w.length) continue;
    const ie = elements.indexOf(w[0]);
    if (ie < 0) continue;
    const je = elements.indexOf(w[1]);
    if (je < 0) continue;
    const ke = elements.indexOf(w[2]);
    if (ke < 0) continue;
    if (w.length < NPARAMS_PER_LINE) throw new StyleError(`COMB potential entry ${w[0]} ${w[1]} ${w[2]} has ${w.length} values, expected ${NPARAMS_PER_LINE}`);
    // one object literal per entry: a fixed object shape keeps the pair loops on fast property access
    const nums: number[] = [];
    for (let k = 0; k < FIELDS.length; k++) {
      const v = Number(w[3 + k]);
      if (!Number.isFinite(v)) throw new StyleError(`COMB potential: bad number '${w[3 + k]}' for ${FIELDS[k]}`);
      nums.push(v);
    }
    const [powerm, c, d, h, powern, beta, lam21, lam22, bigb1, bigb2, bigr, bigd, lam11, lam12, biga1, biga2,
      plp1, plp3, plp6, a123, aconf, addrep, romigb, romigc, romigd, romiga, QL1, QU1, DL1, DU1, QL2, QU2, DL2, DU2,
      chi, dj, dk, dl, dm, esm1, cmn1, cml1, cmn2, cml2, coulcut, hfocor] = nums;
    const q: CombParam = {
      ielement: ie, jelement: je, kelement: ke,
      lam11, lam12, lam21, lam22, c, d, h, gamma: 1, powerm, powern, beta,
      biga1, biga2, bigb1, bigb2, bigd, bigr, cut: 0, cutsq: 0,
      c1: 0, c2: 0, c3: 0, c4: 0,
      plp1, plp3, plp6, a123, aconf, rlm1: 0, rlm2: 0,
      romiga, romigb, romigc, romigd, addrep,
      QU1, QL1, DU1, DL1, Qo1: 0, dQ1: 0, aB1: 0, bB1: 0, nD1: 0, bD1: 0,
      QU2, QL2, DU2, DL2, Qo2: 0, dQ2: 0, aB2: 0, bB2: 0, nD2: 0, bD2: 0,
      chi, dj, dk, dl, dm, esm1, cmn1, cmn2, cml1, cml2,
      coulcut, lcut: 0, lcutsq: 0, hfocor, powermint: 0,
    };
    q.powermint = Math.trunc(q.powerm);
    if (q.lam11 < 0 || q.lam12 < 0 || q.c < 0 || q.d < 0 || q.powern < 0 || q.beta < 0 ||
      q.lam21 < 0 || q.lam22 < 0 || q.bigb1 < 0 || q.bigb2 < 0 || q.biga1 < 0 || q.biga2 < 0 ||
      q.bigr < 0 || q.bigd < 0 || q.bigd > q.bigr ||
      q.powerm - q.powermint !== 0 || (q.powermint !== 3 && q.powermint !== 1) ||
      q.plp1 < 0 || q.plp3 < 0 || q.plp6 < 0 || q.a123 > 360 || q.aconf < 0 ||
      q.addrep < 0 || q.romigb < 0 || q.romigc < 0 || q.romigd < 0 || q.romiga < 0 ||
      q.QL1 > 0 || q.QU1 < 0 || q.DL1 < 0 || q.DU1 > 0 || q.QL2 > 0 || q.QU2 < 0 ||
      q.DL2 < 0 || q.DU2 > 0 || q.chi < 0 || q.esm1 < 0) {
      throw new StyleError('Illegal COMB parameter');
    }
    if (q.lam11 < q.lam21 || q.lam12 < q.lam22 || q.biga1 < q.bigb1 || q.biga2 < q.bigb2) {
      throw new StyleError('Illegal COMB parameter');
    }
    out.push(q);
  }
  return out;
};

/**
 * PairComb::setup_params: elem3param lookup (one exact entry per element triplet) and the parameters
 * derived from the inputs. Returns the table and cutmax / cutmin / cor_flag.
 */
export const setupCombParams = (params: CombParam[], elements: readonly string[]) => {
  const nel = elements.length;
  const elem3param = new Int32Array(nel * nel * nel);
  for (let i = 0; i < nel; i++) for (let j = 0; j < nel; j++) for (let k = 0; k < nel; k++) {
    let n = -1;
    for (let m = 0; m < params.length; m++) {
      const p = params[m];
      if (i === p.ielement && j === p.jelement && k === p.kelement) {
        if (n >= 0) throw new StyleError(`Potential file has a duplicate entry for: ${elements[i]} ${elements[j]} ${elements[k]}`);
        n = m;
      }
    }
    if (n < 0) throw new StyleError(`Potential file is missing an entry for: ${elements[i]} ${elements[j]} ${elements[k]}`);
    elem3param[(i * nel + j) * nel + k] = n;
  }
  for (const p of params) {
    p.cut = p.bigr + p.bigd;
    p.cutsq = p.cut * p.cut;
    p.c1 = Math.pow(2.0 * p.powern * 1.0e-16, -1.0 / p.powern);
    p.c2 = Math.pow(2.0 * p.powern * 1.0e-8, -1.0 / p.powern);
    p.c3 = 1.0 / p.c2;
    p.c4 = 1.0 / p.c1;
    p.rlm1 = 0.5 * (p.lam11 + p.lam12) * p.romigc;
    p.rlm2 = 0.5 * (p.lam21 + p.lam22) * p.romigd;
    p.Qo1 = (p.QU1 + p.QL1) / 2.0;
    p.dQ1 = (p.QU1 - p.QL1) / 2.0;
    p.aB1 = 1.0 / (1.0 - powint(Math.abs(p.Qo1 / p.dQ1), 10));
    p.bB1 = Math.pow(Math.abs(p.aB1), 0.1) / p.dQ1;
    p.nD1 = libmLog(p.DU1 / (p.DU1 - p.DL1)) / libmLog(p.QU1 / (p.QU1 - p.QL1));
    p.bD1 = Math.pow(p.DL1 - p.DU1, 1.0 / p.nD1) / (p.QU1 - p.QL1);
    p.Qo2 = (p.QU2 + p.QL2) / 2.0;
    p.dQ2 = (p.QU2 - p.QL2) / 2.0;
    p.aB2 = 1.0 / (1.0 - powint(Math.abs(p.Qo2 / p.dQ2), 10));
    p.bB2 = Math.pow(Math.abs(p.aB2), 0.1) / p.dQ2;
    p.nD2 = libmLog(p.DU2 / (p.DU2 - p.DL2)) / libmLog(p.QU2 / (p.QU2 - p.QL2));
    p.bD2 = Math.pow(p.DL2 - p.DU2, 1.0 / p.nD2) / (p.QU2 - p.QL2);
    p.lcut = p.coulcut;
    p.lcutsq = p.lcut * p.lcut;
    p.gamma = 1.0;
  }
  let cutmax = 0, cutmin = 0, corFlag = 0;
  for (const p of params) {
    if (p.cut > cutmax) cutmax = p.cut;
    if (p.lcut > cutmax) cutmax = p.lcut;
    if (p.cutsq > cutmin) cutmin = p.cutsq + 0.2;
    if (p.hfocor > 0.0001) corFlag = 1;
  }
  return { elem3param, cutmax, cutmin, corFlag };
};

/** comb_fc: cutoff function with R = bigr, D = bigd. */
export const combFc = (r: number, p: CombParam): number => {
  const R = p.bigr, D = p.bigd;
  if (r < R - D) return 1.0;
  if (r > R + D) return 0.0;
  return 0.5 * (1.0 - libmSin(MY_PI2 * (r - R) / D));
};

export const combFcD = (r: number, p: CombParam): number => {
  const R = p.bigr, D = p.bigd;
  if (r < R - D) return 0.0;
  if (r > R + D) return 0.0;
  return -(MY_PI4 / D) * libmCos(MY_PI2 * (r - R) / D);
};

export const combFc2 = (r: number): number => {
  const R = 1.89350, D = R + 0.050;
  if (r < R) return 0.0;
  if (r > D) return 1.0;
  return 0.5 * (1.0 + libmCos(MY_PI * (r - R) / (D - R)));
};

export const combFc2D = (r: number): number => {
  const R = 1.89350, D = R + 0.050;
  if (r < R) return 0.0;
  if (r > D) return 0.0;
  return -(MY_PI2 / (D - R)) * libmSin(MY_PI * (r - R) / (D - R));
};

export const combFc3 = (r: number): number => {
  const R = 2.51350, D = R + 0.050;
  if (r < R) return 1.0;
  if (r > D) return 0.0;
  return 0.5 * (1.0 + libmCos(MY_PI * (r - R) / (D - R)));
};

export const combFc3D = (r: number): number => {
  const R = 2.51350, D = R + 0.050;
  if (r < R) return 0.0;
  if (r > D) return 0.0;
  return -(MY_PI2 / (D - R)) * libmSin(MY_PI * (r - R) / (D - R));
};

/** comb_gijk: angular function g(cos theta). */
export const combGijk = (costheta: number, p: CombParam): number => {
  const cc = p.c * p.c;
  const dd = p.d * p.d;
  const hcth = p.h - costheta;
  return p.gamma * (1.0 + cc / dd - cc / (dd + hcth * hcth));
};

export const combGijkD = (costheta: number, p: CombParam): number => {
  const cc = p.c * p.c;
  const dd = p.d * p.d;
  const hcth = p.h - costheta;
  const numerator = -2.0 * cc * hcth;
  const denominator = 1.0 / (dd + hcth * hcth);
  return p.gamma * numerator * denominator * denominator;
};

/** comb_bij: bond order function (zeta = effective coordination). */
export const combBij = (zeta: number, p: CombParam): number => {
  const tmp = p.beta * zeta;
  if (tmp > p.c1) return 1.0 / Math.sqrt(tmp);
  if (tmp > p.c2) return (1.0 - Math.pow(tmp, -1.0 * p.powern) / (2.0 * p.powern)) / Math.sqrt(tmp);
  if (tmp < p.c4) return 1.0;
  if (tmp < p.c3) return 1.0 - Math.pow(tmp, p.powern) / (2.0 * p.powern);
  return Math.pow(1.0 + Math.pow(tmp, p.powern), -1.0 / (2.0 * p.powern));
};

export const combBijD = (zeta: number, p: CombParam): number => {
  const tmp = p.beta * zeta;
  if (tmp > p.c1) return p.beta * -0.5 * Math.pow(tmp, -1.5);
  if (tmp > p.c2) {
    return p.beta * (-0.5 * Math.pow(tmp, -1.5) *
      (1.0 - (1.0 + 1.0 / (2.0 * p.powern)) * Math.pow(tmp, -p.powern)));
  }
  if (tmp < p.c4) return 0.0;
  if (tmp < p.c3) return -0.5 * p.beta * Math.pow(tmp, p.powern - 1.0);
  const tmpN = Math.pow(tmp, p.powern);
  return -0.5 * Math.pow(1.0 + tmpN, -1.0 - (1.0 / (2.0 * p.powern))) * tmpN / zeta;
};

/** Charge-dependent D, B and A (PairComb::comb_fa, comb_fa_d, repulsive and qfo_short share these). */
export const combBsi = (p: CombParam, qi: number, qj: number, Di: number, Dj: number): { Bsi: number; Bsj: number } => ({
  Bsi: p.bigb1 * Math.exp(p.lam21 * Di) * (p.aB1 - Math.abs(powint(p.bB1 * (qi - p.Qo1), 10))),
  Bsj: p.bigb2 * Math.exp(p.lam22 * Dj) * (p.aB2 - Math.abs(powint(p.bB2 * (qj - p.Qo2), 10))),
});

export const combDi = (p: CombParam, qi: number): number => p.DU1 + Math.pow(Math.abs(p.bD1 * (p.QU1 - qi)), p.nD1);
export const combDj = (p: CombParam, qj: number): number => p.DU2 + Math.pow(Math.abs(p.bD2 * (p.QU2 - qj)), p.nD2);
