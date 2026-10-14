/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_comb3.cpp and src/MANYBODY/pair_comb3.h (PairComb3::read_lib, read_file,
 * setup_params), at commit b332f28. Authors credited in the C++ file headers: Tzu-Ray Shan (Sandia), with
 * the COMB3 development of Ting-Wei Liang and Tzu-Ray Shan. Copyright (2003) Sandia Corporation (GNU
 * General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';
import { joinPotentialEntries } from '../../engine/force/util';
import { powint } from '../comb/params';

/** Words per potential entry (PairComb3::NPARAMS_PER_LINE). */
export const NPARAMS_PER_LINE = 74;

/** Per (i,j,k) parameter set; field names follow pair_comb3.h struct Param. */
export interface Comb3Param {
  ielement: number; jelement: number; kelement: number; powermint: number;
  ielementgp: number; jelementgp: number; kelementgp: number;
  ang_flag: number; pcn_flag: number; rad_flag: number; tor_flag: number;
  lami: number; lambda: number; alfi: number; alpha1: number; alpha2: number; alpha3: number; beta: number;
  pcos6: number; pcos5: number; pcos4: number; pcos3: number; pcos2: number; pcos1: number; pcos0: number;
  gamma: number; powerm: number; powern: number; bigA: number; bigB1: number; bigB2: number; bigB3: number;
  bigd: number; bigr: number; cut: number; cutsq: number; c1: number; c2: number; c3: number; c4: number;
  p6p0: number; p6p1: number; p6p2: number; p6p3: number; p6p4: number; p6p5: number; p6p6: number;
  ptork1: number; ptork2: number;
  addrepr: number; addrep: number; vdwflag: number;
  QU: number; QL: number; DU: number; DL: number; Qo: number; dQ: number; aB: number; bB: number; nD: number; bD: number;
  qmin: number; qmax: number;
  chi: number; dj: number; dk: number; dl: number; dm: number; esm: number;
  cmn1: number; cmn2: number; pcmn1: number; pcmn2: number;
  coulcut: number; lcut: number; lcutsq: number;
  veps: number; vsig: number; pcna: number; pcnb: number; pcnc: number; pcnd: number; polz: number; curl: number; pcross: number;
  paaa: number; pbbb: number;
  curlcut1: number; curlcut2: number; curl0: number;
}

/** Numeric words after the three element names, in file order: 7 integers, then these 64 reals (read_file). */
const INT_FIELDS = ['ielementgp', 'jelementgp', 'kelementgp', 'ang_flag', 'pcn_flag', 'rad_flag', 'tor_flag'] as const;
const REAL_FIELDS = [
  'vdwflag', 'powerm', 'veps', 'vsig', 'paaa', 'pbbb', 'lami', 'alfi', 'powern', 'QL', 'QU', 'DL', 'DU',
  'qmin', 'qmax', 'chi', 'dj', 'dk', 'dl', 'esm', 'cmn1', 'cmn2', 'pcmn1', 'pcmn2', 'coulcut', 'polz', 'curl',
  'curlcut1', 'curlcut2', 'curl0', 'alpha1', 'bigB1', 'alpha2', 'bigB2', 'alpha3', 'bigB3', 'lambda', 'bigA',
  'beta', 'bigr', 'bigd', 'pcos6', 'pcos5', 'pcos4', 'pcos3', 'pcos2', 'pcos1', 'pcos0', 'pcna', 'pcnb', 'pcnc',
  'pcnd', 'p6p0', 'p6p1', 'p6p2', 'p6p3', 'p6p4', 'p6p5', 'p6p6', 'ptork1', 'ptork2', 'addrepr', 'addrep', 'pcross',
] as const;

/**
 * PairComb3::read_file: the entries whose three element names are all in `elements`, with the sanity checks of
 * the C++ reader (the same error for every violated rule, as in the native code).
 */
export const readComb3Entries = (text: string, elements: readonly string[]): Comb3Param[] => {
  const lines = joinPotentialEntries(text.split(/\r?\n/), NPARAMS_PER_LINE);
  const out: Comb3Param[] = [];
  for (const line of lines) {
    const w = line.trim() ? line.trim().split(/\s+/) : [];
    if (!w.length) continue;
    const ie = elements.indexOf(w[0]);
    const je = ie >= 0 ? elements.indexOf(w[1] ?? '') : -1;
    const ke = je >= 0 ? elements.indexOf(w[2] ?? '') : -1;
    if (ke < 0) continue;
    if (w.length < NPARAMS_PER_LINE) throw new StyleError(`COMB3 potential entry ${w[0]} ${w[1]} ${w[2]} has ${w.length} values, expected ${NPARAMS_PER_LINE}`);
    const num = (k: number): number => {
      const v = Number(w[k]);
      if (!Number.isFinite(v)) throw new StyleError(`COMB3 potential: bad number '${w[k]}' in entry ${w[0]} ${w[1]} ${w[2]}`);
      return v;
    };
    const p: Record<string, number> = {};
    INT_FIELDS.forEach((name, k) => { p[name] = num(3 + k); });
    REAL_FIELDS.forEach((name, k) => { p[name] = num(3 + INT_FIELDS.length + k); });
    const q = {
      ielement: ie, jelement: je, kelement: ke, powermint: Math.trunc(p.powerm),
      ielementgp: p.ielementgp, jelementgp: p.jelementgp, kelementgp: p.kelementgp,
      ang_flag: p.ang_flag, pcn_flag: p.pcn_flag, rad_flag: p.rad_flag, tor_flag: p.tor_flag,
      lami: p.lami, lambda: p.lambda, alfi: p.alfi, alpha1: p.alpha1, alpha2: p.alpha2, alpha3: p.alpha3, beta: p.beta,
      pcos6: p.pcos6, pcos5: p.pcos5, pcos4: p.pcos4, pcos3: p.pcos3, pcos2: p.pcos2, pcos1: p.pcos1, pcos0: p.pcos0,
      gamma: 1, powerm: p.powerm, powern: p.powern, bigA: p.bigA, bigB1: p.bigB1, bigB2: p.bigB2, bigB3: p.bigB3,
      bigd: p.bigd, bigr: p.bigr, cut: 0, cutsq: 0, c1: 0, c2: 0, c3: 0, c4: 0,
      p6p0: p.p6p0, p6p1: p.p6p1, p6p2: p.p6p2, p6p3: p.p6p3, p6p4: p.p6p4, p6p5: p.p6p5, p6p6: p.p6p6,
      ptork1: p.ptork1, ptork2: p.ptork2, addrepr: p.addrepr, addrep: p.addrep, vdwflag: p.vdwflag,
      QU: p.QU, QL: p.QL, DU: p.DU, DL: p.DL, Qo: 0, dQ: 0, aB: 0, bB: 0, nD: 0, bD: 0, qmin: p.qmin, qmax: p.qmax,
      chi: p.chi, dj: p.dj, dk: p.dk, dl: p.dl, dm: 0, esm: p.esm, cmn1: p.cmn1, cmn2: p.cmn2, pcmn1: p.pcmn1, pcmn2: p.pcmn2,
      coulcut: p.coulcut, lcut: 0, lcutsq: 0,
      veps: p.veps, vsig: p.vsig, pcna: p.pcna, pcnb: p.pcnb, pcnc: p.pcnc, pcnd: p.pcnd, polz: p.polz, curl: p.curl,
      pcross: p.pcross, paaa: p.paaa, pbbb: p.pbbb, curlcut1: p.curlcut1, curlcut2: p.curlcut2, curl0: p.curl0,
    } as Comb3Param;
    // parameter sanity checks (pair_comb3.cpp read_file)
    if (q.lambda < 0.0 || q.powern < 0.0 || q.beta < 0.0 || q.alpha1 < 0.0 ||
      q.bigB1 < 0.0 || q.bigA < 0.0 || q.bigB2 < 0.0 || q.alpha2 < 0.0 ||
      q.bigB3 < 0.0 || q.alpha3 < 0.0 || q.bigr < 0.0 || q.bigd < 0.0 ||
      q.bigd > q.bigr || q.powerm - q.powermint !== 0.0 || q.addrepr < 0.0 || q.powermint < 1.0 ||
      q.QL > 0.0 || q.QU < 0.0 || q.DL < 0.0 || q.DU > 0.0 || q.pcross < 0.0 ||
      q.esm < 0.0 || q.veps < 0.0 || q.vsig < 0.0 || q.vdwflag < 0.0) {
      throw new StyleError('Illegal COMB3 parameter');
    }
    if (q.ielementgp < 1 || q.ielementgp > 3 || q.jelementgp < 1 || q.jelementgp > 3 ||
      q.kelementgp < 1 || q.kelementgp > 3) {
      throw new StyleError('Invalid element group index in COMB3 potential file');
    }
    out.push(q);
  }
  return out;
};

/** PairComb3::setup_params: elem3param (one exact entry per element triplet) and the derived parameters. */
export const setupComb3Params = (params: Comb3Param[], nel: number) => {
  const elem3param = new Int32Array(nel * nel * nel);
  for (let i = 0; i < nel; i++) for (let j = 0; j < nel; j++) for (let k = 0; k < nel; k++) {
    let n = -1;
    for (let m = 0; m < params.length; m++) {
      const p = params[m];
      if (i === p.ielement && j === p.jelement && k === p.kelement) {
        if (n >= 0) throw new StyleError(`Potential file has a duplicate entry for: element ${i} ${j} ${k}`);
        n = m;
      }
    }
    if (n < 0) throw new StyleError(`Potential file is missing an entry for: element ${i} ${j} ${k}`);
    elem3param[(i * nel + j) * nel + k] = n;
  }
  for (const p of params) {
    p.cut = p.bigr + p.bigd;
    p.cutsq = p.cut * p.cut;
    p.c1 = Math.pow(2.0 * p.powern * 1.0e-16, -1.0 / p.powern);
    p.c2 = Math.pow(2.0 * p.powern * 1.0e-8, -1.0 / p.powern);
    p.c3 = 1.0 / p.c2;
    p.c4 = 1.0 / p.c1;
    p.Qo = (p.QU + p.QL) / 2.0;
    p.dQ = (p.QU - p.QL) / 2.0;
    p.aB = 1.0 / (1.0 - powint(Math.abs(p.Qo / p.dQ), 10));
    p.bB = Math.pow(Math.abs(p.aB), 0.1) / p.dQ;
    p.nD = Math.log(p.DU / (p.DU - p.DL)) / Math.log(p.QU / (p.QU - p.QL));
    p.bD = Math.pow(p.DL - p.DU, 1.0 / p.nD) / (p.QU - p.QL);
    p.lcut = p.coulcut;
    p.lcutsq = p.lcut * p.lcut;
  }
  // cutmin is compared with squared distances and set to cutsq + 2.0 (pair_comb3.cpp setup_params, as written)
  let cutmin = 0.0, cutmax = 0.0;
  for (const p of params) {
    if (p.cutsq > cutmin) cutmin = p.cutsq + 2.0;
    if (p.lcut > cutmax) cutmax = p.lcut;
  }
  return { elem3param, cutmin, cutmax, chicut1: 7.0, chicut2: cutmax };
};

/** The COMB3 library file (read_lib): cutoff constants, angle and spline tables. */
export interface Comb3Lib {
  ccutoff: Float64Array; // 6
  chA: Float64Array; // 7 (ch_a)
  nsplpcn: number; nsplrad: number; nspltor: number;
  maxx: number; maxy: number; maxz: number;
  maxxc: number; maxyc: number; maxconj: number;
  maxxcn: Int32Array; // 4
  vmaxxcn: Float64Array; // 4
  dvmaxxcn: Float64Array; // 4
  ntab: number;
  pang: Float64Array; dpang: Float64Array; ddpang: Float64Array; // ntab+1
  // spline grids, flat: pcnGrid[(((l*5+i)*5+j)*5+k)], pcnCubs[((((l*4+i)*4+j)*4+k)*64+m)] etc.
  pcnGrid: Float64Array; pcnGridx: Float64Array; pcnGridy: Float64Array; pcnGridz: Float64Array;
  pcnCubs: Float64Array;
  radGrid: Float64Array; radGridx: Float64Array; radGridy: Float64Array; radGridz: Float64Array;
  radSpl: Float64Array;
  torGrid: Float64Array; torGridx: Float64Array; torGridy: Float64Array; torGridz: Float64Array;
  torSpl: Float64Array;
}

/** Token reader over the library file: a "line" read takes one whole line, a "dvector" read takes tokens across lines. */
class LibReader {
  private lines: string[][] = [];
  private line = 0;
  private tok = 0;
  constructor(text: string) {
    for (const raw of text.split(/\r?\n/)) {
      const h = raw.indexOf('#');
      const t = (h >= 0 ? raw.slice(0, h) : raw).trim();
      if (t) this.lines.push(t.split(/\s+/));
    }
  }
  private take(): string {
    while (this.line < this.lines.length && this.tok >= this.lines[this.line].length) {
      this.line++;
      this.tok = 0;
    }
    if (this.line >= this.lines.length) throw new StyleError('COMB3 library file ended early');
    return this.lines[this.line][this.tok++];
  }
  /** One whole line (next_values): returns its tokens as numbers. */
  values(): number[] {
    if (this.tok > 0) { this.line++; this.tok = 0; }
    if (this.line >= this.lines.length) throw new StyleError('COMB3 library file ended early');
    return this.lines[this.line++].map(Number);
  }
  /** n numbers, possibly across lines (next_dvector). */
  dvector(out: Float64Array, offset: number, n: number): void {
    for (let k = 0; k < n; k++) out[offset + k] = Number(this.take());
  }
}

/** PairComb3::read_lib (lib.comb3). The spline grids are stored flat, indexed as in the C++ declarations. */
export const readComb3Lib = (text: string): Comb3Lib => {
  const r = new LibReader(text);
  const ccutoff = new Float64Array(6);
  r.dvector(ccutoff, 0, 6);
  const chA = new Float64Array(7);
  r.dvector(chA, 0, 7);
  let v = r.values();
  const nsplpcn = v[0], nsplrad = v[1], nspltor = v[2];
  v = r.values();
  const maxx = v[0], maxy = v[1], maxz = v[2];
  v = r.values();
  const maxxc = v[0], maxyc = v[1], maxconj = v[2];
  const maxxcn = new Int32Array(4), vmaxxcn = new Float64Array(4), dvmaxxcn = new Float64Array(4);
  for (let l = 0; l < nsplpcn; l++) {
    const w = r.values();
    maxxcn[l] = w[1];
    vmaxxcn[l] = w[2];
    dvmaxxcn[l] = w[3];
  }
  const ntab = r.values()[0];
  const pang = new Float64Array(ntab + 1), dpang = new Float64Array(ntab + 1), ddpang = new Float64Array(ntab + 1);
  for (let i = 0; i <= ntab; i++) {
    const w = r.values();
    pang[i] = w[1];
    dpang[i] = w[2];
    ddpang[i] = w[3];
  }
  const pcnGrid = new Float64Array(4 * 5 * 5 * 5), pcnGridx = new Float64Array(4 * 5 * 5 * 5);
  const pcnGridy = new Float64Array(4 * 5 * 5 * 5), pcnGridz = new Float64Array(4 * 5 * 5 * 5);
  const g5 = (l: number, i: number, j: number, k: number): number => ((l * 5 + i) * 5 + j) * 5 + k;
  for (let l = 0; l < nsplpcn; l++) for (let i = 0; i <= maxx; i++) for (let j = 0; j <= maxy; j++) for (let k = 0; k <= maxz; k++) {
    const w = r.values();
    const ll = w[0] - 1, ii = w[1], jj = w[2], kk = w[3];
    const at = g5(ll, ii, jj, kk);
    pcnGrid[at] = w[4]; pcnGridx[at] = w[5]; pcnGridy[at] = w[6]; pcnGridz[at] = w[7];
  }
  const pcnCubs = new Float64Array(4 * 4 * 4 * 4 * 64);
  const c4 = (l: number, i: number, j: number, k: number): number => (((l * 4 + i) * 4 + j) * 4 + k) * 64;
  for (let l = 0; l < nsplpcn; l++) for (let i = 0; i < maxx; i++) for (let j = 0; j < maxy; j++) for (let k = 0; k < maxz; k++) {
    const w = r.values();
    const base = c4(w[0] - 1, w[1], w[2], w[3]);
    r.dvector(pcnCubs, base, 32);
    r.dvector(pcnCubs, base + 32, 32);
  }
  // rad_grid[3][5][5][11], rad_spl[3][4][4][10][64]
  const g511 = (l: number, i: number, j: number, k: number): number => ((l * 5 + i) * 5 + j) * 11 + k;
  const radGrid = new Float64Array(3 * 5 * 5 * 11), radGridx = new Float64Array(3 * 5 * 5 * 11);
  const radGridy = new Float64Array(3 * 5 * 5 * 11), radGridz = new Float64Array(3 * 5 * 5 * 11);
  for (let l = 0; l < nsplrad; l++) for (let i = 0; i <= maxxc; i++) for (let j = 0; j <= maxyc; j++) for (let k = 0; k < maxconj; k++) {
    const w = r.values();
    const at = g511(w[0] - 1, w[1], w[2], w[3] - 1);
    radGrid[at] = w[4]; radGridx[at] = w[5]; radGridy[at] = w[6]; radGridz[at] = w[7];
  }
  const radSpl = new Float64Array(3 * 4 * 4 * 10 * 64);
  const s4 = (l: number, i: number, j: number, k: number): number => (((l * 4 + i) * 4 + j) * 10 + k) * 64;
  for (let l = 0; l < nsplrad; l++) for (let i = 0; i < maxxc; i++) for (let j = 0; j < maxyc; j++) for (let k = 0; k < maxconj - 1; k++) {
    const w = r.values();
    const base = s4(w[0] - 1, w[1], w[2], w[3] - 1);
    r.dvector(radSpl, base, 32);
    r.dvector(radSpl, base + 32, 32);
  }
  const torGrid = new Float64Array(1 * 5 * 5 * 11), torGridx = new Float64Array(1 * 5 * 5 * 11);
  const torGridy = new Float64Array(1 * 5 * 5 * 11), torGridz = new Float64Array(1 * 5 * 5 * 11);
  for (let l = 0; l < nspltor; l++) for (let i = 0; i <= maxxc; i++) for (let j = 0; j <= maxyc; j++) for (let k = 0; k < maxconj; k++) {
    const w = r.values();
    const at = g511(w[0] - 1, w[1], w[2], w[3] - 1);
    torGrid[at] = w[4]; torGridx[at] = w[5]; torGridy[at] = w[6]; torGridz[at] = w[7];
  }
  const torSpl = new Float64Array(1 * 4 * 4 * 10 * 64);
  for (let l = 0; l < nspltor; l++) for (let i = 0; i < maxxc; i++) for (let j = 0; j < maxyc; j++) for (let k = 0; k < maxconj - 1; k++) {
    const w = r.values();
    const base = s4(w[0] - 1, w[1], w[2], w[3] - 1);
    r.dvector(torSpl, base, 32);
    r.dvector(torSpl, base + 32, 32);
  }
  return {
    ccutoff, chA, nsplpcn, nsplrad, nspltor, maxx, maxy, maxz, maxxc, maxyc, maxconj,
    maxxcn, vmaxxcn, dvmaxxcn, ntab, pang, dpang, ddpang,
    pcnGrid, pcnGridx, pcnGridy, pcnGridz, pcnCubs,
    radGrid, radGridx, radGridy, radGridz, radSpl,
    torGrid, torGridx, torGridy, torGridz, torSpl,
  };
};
