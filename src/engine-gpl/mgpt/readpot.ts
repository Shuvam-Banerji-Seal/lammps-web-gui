/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MGPT/mgpt_readpot.cpp and src/MGPT/mgpt_readpot.h (struct potdata) at commit b332f28 (the MGPT package).
 * Fast MGPT algorithm: Tomas Oppelstrup (LLNL, 2015), from the matrix MGPT v4.4 FORTRAN routine of John Moriarty
 * (LLNL, 2006). Copyright (2015) Lawrence Livermore National Security, LLC, LLNL-CODE-674031, distributed under the
 * GNU General Public License version 2; the LLNL notice is kept in NOTICE.llnl next to this file.
 * This file is GPL-2.0 too (see src/engine-gpl/LICENSE).
 *
 * Reads the parmin and potin files of an MGPT potential and interpolates its volume-dependent parameters (the
 * multi-ion coefficients va..ve, the bond-length function parameters p1, al, rp, r00, the volume term evol0 and the
 * pair potential table, each with its derivative with respect to the atomic volume) at the atomic volume given in
 * pair_coeff.
 */
import { StyleError } from '../../engine/force/types';
import { exp } from '../libm/exp';
import { pow } from '../libm/pow';
import { evalspline, makespline, type SplineValue } from './splinetab';

/** mgpt_readpot.h struct potdata (the fields the pair style reads). */
export interface PotData {
  va: number; vb: number; vc: number; vd: number; ve: number; p1: number; al: number; rp: number; r00: number; pn: number;
  dva: number; dvb: number; dvc: number; dvd: number; dve: number; dp1: number; dal: number; drp: number; dr00: number;
  evol0: number; devol0: number;
  /** nr-1 cubics (4 coefficients each) of the pair energy and of its volume derivative, over [r0, r1]. */
  vpairSpline: Float64Array;
  dvpairSpline: Float64Array;
  r0: number; r1: number; nr: number;
  mass: number; rcrit: number; rmax: number;
  lang: number; lmax: number;
  anorm3: number; anorm4: number;
  ddl: Float64Array;
  ipot: number; mode: number;
  metal: string;
  inputVol: number;
}

/** potdata::eval_pot: the pair energy and its r derivative at r. */
export const evalPot = (P: PotData, r: number, out: SplineValue): void => {
  evalspline(P.nr - 1, P.r0, P.r1, P.vpairSpline, r, out);
};

/** potdata::eval_vir: the volume derivative of the pair energy at r (in out.y). */
export const evalVir = (P: PotData, r: number, out: SplineValue): void => {
  evalspline(P.nr - 1, P.r0, P.r1, P.dvpairSpline, r, out);
};

/**
 * Line reader with the semantics of the C stdio calls of mgpt_readpot.cpp: fgets returns the next line (with a 1024-byte
 * buffer, a longer line comes in pieces of at most 1023 characters) and leaves the previous line in place at end of file.
 */
class LineReader {
  private pos = 0;
  line = '';
  constructor(private readonly text: string) {}
  /** fgets(line, 1024, in): false at end of file (line keeps its previous contents). */
  fgets(): boolean {
    const t = this.text;
    if (this.pos >= t.length) return false;
    let end = t.indexOf('\n', this.pos);
    end = end < 0 ? t.length : end + 1;
    if (end - this.pos > 1023) end = this.pos + 1023;
    this.line = t.slice(this.pos, end);
    this.pos = end;
    return true;
  }
}

/** line[strspn(line, " \t")] == '#': a comment line. */
const isComment = (line: string): boolean => {
  let k = 0;
  while (k < line.length && (line[k] === ' ' || line[k] === '\t')) k++;
  return line[k] === '#';
};

type Conv = 'f' | 'd' | 's';

/**
 * sscanf with conversions %lf (f), %d (d) and %s (s) separated by blanks: the values converted before the first failure
 * (strtod / strtol syntax). The caller keeps its old value for every conversion past the returned list, as sscanf does.
 */
const sscanf = (line: string, convs: readonly Conv[]): (number | string)[] => {
  const out: (number | string)[] = [];
  let k = 0;
  const ws = (c: string): boolean => c === ' ' || c === '\t' || c === '\n' || c === '\r' || c === '\f' || c === '\v';
  for (const c of convs) {
    while (k < line.length && ws(line[k])) k++;
    if (k >= line.length) break;
    const rest = line.slice(k);
    if (c === 's') {
      let e = k;
      while (e < line.length && !ws(line[e])) e++;
      out.push(line.slice(k, e));
      k = e;
    } else if (c === 'd') {
      const m = /^[+-]?\d+/.exec(rest);
      if (!m) break;
      out.push(parseInt(m[0], 10));
      k += m[0].length;
    } else {
      const m = /^[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)/i.exec(rest);
      if (!m) break;
      const w = m[0].toLowerCase();
      const sign = w.startsWith('-') ? -1 : 1;
      const body = w.replace(/^[+-]/, '');
      out.push(body.startsWith('inf') ? sign * Infinity : body === 'nan' ? NaN : Number(w));
      k += m[0].length;
    }
  }
  return out;
};

const num = (v: number | string | undefined, old: number): number => (typeof v === 'number' ? v : old);

/** mgpt_readpot.cpp fgauss and hgauss (pow(x - 1.0, 2) is the product the compiler makes of it). */
const fgauss = (x: number, al: number): number => {
  const t = x - 1.0;
  return exp(-al * (t * t));
};
const hgauss = (x: number, al: number): number => {
  const t = x - 1.0;
  return (1.0 + al * (t * t)) * exp(-al * (t * t));
};

/** mgpt_readpot.cpp fl: the bond-length function. */
const fl = (r: number, mode: number, rp: number, p1: number, al: number, r0: number, pn: number): number => {
  let term: number;
  if (mode <= 4) term = pow(rp / r, p1);
  else term = exp(-p1 * (pow(r / rp, pn) - 1.0) / pn);

  if (r <= r0) return term;
  const quan = al * (r / r0 - 1.0) * (r / r0 - 1.0);
  if (mode <= 2) return term * exp(-quan);
  return term * (1.0 + quan) * exp(-quan);
};

/** mgpt_readpot.cpp getparmindata: the volume grid of a potin file (one-line parmin files). */
const getparmindata = (potin: string, potinName: string): { nvol: number; vol0: number; x0: number; x1: number } => {
  const rd = new LineReader(potin);
  const volarr: number[] = [];
  let metal = '', ipot = 0, mode = 0;
  while (rd.fgets()) {
    if (isComment(rd.line)) continue;
    const h = sscanf(rd.line, ['s', 'd', 'd']);
    if (h.length !== 3) throw new StyleError(`Error on potin file ${potinName}. line = ${rd.line.trimEnd()}`);
    if (volarr.length === 0) {
      [metal, ipot, mode] = h as [string, number, number];
    } else if (h[0] !== metal || h[1] !== ipot || h[2] !== mode) {
      throw new StyleError(`Error on potin file ${potinName}, parameter mismatch: metal = '${metal}' ipot = ${ipot} mode = ${mode}, `
        + `metalx = '${h[0]}' ipotx = ${h[1]} modex = ${h[2]}`);
    }
    rd.fgets();
    const v = sscanf(rd.line, ['f', 'f', 'f', 'f']);
    volarr.push(num(v[1], 0));
    for (let i = 0; i < 5; i++) rd.fgets();
    const g = sscanf(rd.line, ['f', 'f', 'f']);
    const r0x = num(g[0], 0), r1x = num(g[1], 0), drx = num(g[2], 0);
    const nrx = Math.trunc((r1x - r0x) / drx + 1.1);
    for (let i = 0; i < nrx; i++) rd.fgets();
  }
  const n = volarr.length;
  if (n === 0) throw new StyleError(`Invalid potin file '${potinName}', no volume records.`);
  volarr.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  const vol0 = volarr[n >> 1];
  return { nvol: n, vol0, x0: pow(volarr[0] / vol0, 1.0 / 3.0), x1: pow(volarr[n - 1] / vol0, 1.0 / 3.0) };
};

/** atan(1.0), which the compiler folds: pi/4 rounded to double (Math.PI / 4 is that value exactly). */
const ATAN1 = Math.PI / 4;

/** potdata::readpot: the potential at the atomic volume vol (a.u.). */
export const readpot = (parmin: string, parminName: string, potin: string, potinName: string, vol: number): PotData => {
  const ddl = new Float64Array(5);
  let x0 = 0, x1 = 0, dx = 0, vol0 = 0;
  let L = 0;

  // Read potential data
  const pr = new LineReader(parmin);
  do {
    pr.fgets();
  } while (isComment(pr.line));

  // Test to see whether this is a one-line or two-line version of parmin
  const first = sscanf(pr.line, ['f', 'f', 'f', 'f', 'd']);
  for (let k = 0; k < 4 && k < first.length; k++) ddl[k + 1] = first[k] as number;
  if (first.length === 5) {
    L = first[4] as number;
    // One-line version, call getparmindata to figure out volume table spacing.
    const g = getparmindata(potin, potinName);
    vol0 = g.vol0;
    x0 = g.x0;
    x1 = g.x1;
    dx = (x1 - x0) / (g.nvol - 1);
  } else {
    // Two-line version, reparse this line, and read second line
    const a = sscanf(pr.line, ['f', 'f', 'f', 'f']);
    x0 = num(a[0], x0); x1 = num(a[1], x1); dx = num(a[2], dx); vol0 = num(a[3], vol0);
    pr.fgets();
    const b = sscanf(pr.line, ['f', 'f', 'f', 'f', 'd']);
    for (let k = 0; k < 4 && k < b.length; k++) ddl[k + 1] = b[k] as number;
    if (b.length === 5) L = b[4] as number;
  }
  const rwsScale = pow(3.0 * vol0 / (16.0 * ATAN1), 1.0 / 3.0);

  const lang = L + 1;
  const lmax = 2 * L + 1;
  const s = ddl[1], p = ddl[2], d = ddl[3], f = ddl[4];
  const ss = s * s, pp = p * p, dd = d * d, ff = f * f;
  const anorm3 = s * ss + 2.0 * (p * pp + d * dd + f * ff);
  const anorm4 = ss * ss + 2.0 * (pp * pp + dd * dd + ff * ff);

  const nx = Math.trunc((x1 - x0) / dx + 1.1); // Really: 1+round((x1-x0)/dx)
  if (!(nx >= 4)) throw new StyleError(`MGPT parmin file ${parminName}: the volume table needs at least 4 points (it has ${nx})`);
  const vatab = new Float64Array(nx), vbtab = new Float64Array(nx), vctab = new Float64Array(nx);
  const vdtab = new Float64Array(nx), vetab = new Float64Array(nx);
  const p1tab = new Float64Array(nx), altab = new Float64Array(nx);
  const r0rwstab = new Float64Array(nx), evol0tab = new Float64Array(nx);
  const tag = new Uint8Array(nx);

  let metal = '', ipot = 0, mode = 0, pn = 0;
  let r0 = 0, r1 = 0, nr = 0;
  let vpairtab = new Float64Array(0);
  let mass = 0, rcrws = 0, rmrws = 0;

  const rd = new LineReader(potin);
  for (let ii = 0; ii < nx; ii++) {
    do {
      if (!rd.fgets()) throw new StyleError(`MGPT potin file ${potinName} ends before volume record ${ii} (the parmin file asks for ${nx})`);
    } while (isComment(rd.line));

    // Read element type, mode, and pn parameter
    const h = sscanf(rd.line, ['s', 'd', 'd', 'f']);
    const nf = h.length;
    if (nf < 3) {
      throw new StyleError(`Inconsistency in potential input file (${potinName}) at record ${ii}: expected at least three fields. Number of fields = ${nf}`);
    }
    const metalx = h[0] as string, ipotx = h[1] as number, modex = h[2] as number;
    let pnx = nf === 4 ? (h[3] as number) : 0;
    if (modex <= 4) {
      pnx = 1.0;
    } else if (modex <= 6) {
      if (nf !== 4) {
        throw new StyleError(`Inconsistency in potential input file (${potinName}) at record ${ii}: mode = ${modex}, number of fields = ${nf}`);
      }
    }
    // a mode above 6 is reported natively on stdout without stopping (fl() then uses its mode > 4 branch)

    if (ii === 0) {
      metal = metalx; ipot = ipotx; mode = modex;
      if (nf === 4) pn = h[3] as number;
      if (modex <= 4) pn = pnx;
    } else if (metal !== metalx || ipotx !== ipot || modex !== mode || pnx !== pn) {
      // Check that {metal,ipot,mode}x == {metal,ipot,mode}
      throw new StyleError(`Inconsistency in potential input file (${potinName}) at record ${ii}: `
        + `metalx != metal (${metalx} != ${metal}) or ipotx != ipot (${ipotx} != ${ipot}) or modex != mode (${modex} != ${mode}) `
        + `or pnx != pn (${pnx.toFixed(3)} != ${pn.toFixed(3)}).`);
    }
    rd.fgets();
    const zl = sscanf(rd.line, ['f', 'f', 'f', 'f']);
    const ivol = num(zl[1], 0);
    mass = num(zl[3], mass);

    const ifrac = (pow(ivol / vol0, 1.0 / 3.0) - x0) / ((x1 - x0) / (nx - 1));
    const i = Math.trunc(ifrac + 0.1);
    if (Math.abs(i - ifrac) > 0.01 || i < 0 || i >= nx) {
      throw new StyleError(`Volume point not in table... ii=${ii} i=${i} ifrac=${ifrac.toExponential(5)} vol=${ivol.toExponential(5)} (MGPT potin file ${potinName})`);
    } else if (tag[i] === 1) {
      throw new StyleError(`Duplicate volume point in table.... ii=${ii} i=${i} ifrac=${ifrac.toExponential(5)} vol=${ivol.toExponential(5)} (MGPT potin file ${potinName})`);
    } else tag[i] = 1;

    rd.fgets();
    const l3 = sscanf(rd.line, ['f', 'f', 'f', 'f']);
    r0rwstab[i] = num(l3[0], r0rwstab[i]); altab[i] = num(l3[1], altab[i]);
    rcrws = num(l3[2], rcrws); rmrws = num(l3[3], rmrws);
    rd.fgets();
    const l4 = sscanf(rd.line, ['f', 'f', 'f']);
    p1tab[i] = num(l4[0], p1tab[i]); evol0tab[i] = num(l4[2], evol0tab[i]);

    rd.fgets();
    const l5 = sscanf(rd.line, ['f', 'f', 'f', 'f', 'f']);
    vatab[i] = num(l5[0], vatab[i]); vbtab[i] = num(l5[1], vbtab[i]); vctab[i] = num(l5[2], vctab[i]);
    vdtab[i] = num(l5[3], vdtab[i]); vetab[i] = num(l5[4], vetab[i]);
    if (ipot === 1) {
      vatab[i] *= vdtab[i];
      vctab[i] *= vctab[i];
      vetab[i] *= vetab[i];
    }

    rd.fgets();

    rd.fgets();
    const l7 = sscanf(rd.line, ['f', 'f', 'f']);
    const r0x = num(l7[0], 0), r1x = num(l7[1], 0), drx = num(l7[2], 0);
    const nrx = Math.trunc((r1x - r0x) / drx + 1.1); // Really: 1+round((r1-r0)/dr)

    if (ii === 0) {
      r0 = r0x; r1 = r1x; nr = nrx;
      if (!(nr >= 4)) throw new StyleError(`MGPT potin file ${potinName}: the pair table needs at least 4 points (it has ${nr})`);
      vpairtab = new Float64Array(nx * nr);
    }
    // (later records: native does not check that {r0,r1,dr,nr}x == {r0,r1,dr,nr}; it reads nr lines)

    for (let j = 0; j < nr; j++) {
      rd.fgets();
      const lp = sscanf(rd.line, ['f', 'f', 'f', 'f']);
      vpairtab[i * nr + j] = num(lp[1], vpairtab[i * nr + j]);

      // Add screening and fl() part to pair energy table
      const al = altab[i];
      const p1 = p1tab[i];

      const bscreen = al > 0.0;

      const xi = x0 + i / (nx - 1) * (x1 - x0);
      const rws = rwsScale * xi;

      let r0rws = r0rwstab[i];
      const r00 = r0rws * rws, rp = 1.8 * rws;
      if (!bscreen) r0rws = 10.0;

      const r = r0 + j * (r1 - r0) / (nr - 1);

      const rrws = r / rws;
      const flr = fl(r, mode, rp, p1, al, r00, pn);
      const fl2 = flr * flr;
      const v2a = vatab[i] * fl2 * fl2;
      const v2b = vbtab[i] * fl2;
      let fscr = 1.0;

      if (bscreen && rrws >= r0rws) {
        const arg = rrws / r0rwstab[i];
        const fs = mode <= 2 ? fgauss(arg, al) : hgauss(arg, al);
        fscr = fs * fs;
      }

      vpairtab[i * nr + j] = vpairtab[i * nr + j] * fscr + v2a - v2b;
    }
  }

  for (let i = 0; i < nx; i++) {
    if (tag[i] === 0) throw new StyleError(`Volume point missing in table. i = ${i} (MGPT potin file ${potinName})`);
  }

  // Make table
  const x = pow(vol / vol0, 1.0 / 3.0);
  const dxdv = 1.0 / (3.0 * vol0 * x * x);

  const C = new Float64Array(4 * Math.max(nr, nx));
  const sv: SplineValue = { y: 0, dy: 0, d2y: 0 };
  const at = (tab: Float64Array): [number, number] => {
    makespline(nx, 1, tab, 0, C);
    evalspline(nx - 1, x0, x1, C, x, sv);
    return [sv.y, sv.dy * dxdv];
  };
  const [va, dva] = at(vatab);
  const [vb, dvb] = at(vbtab);
  const [vc, dvc] = at(vctab);
  const [vd, dvd] = at(vdtab);
  const [ve, dve] = at(vetab);
  const [p1, dp1] = at(p1tab);
  let [al, dal] = at(altab);
  if (mode === 2 || mode === 4 || mode === 6) {
    al = 125.0;
    dal = 0.0;
  }

  const [r0rws, dr0rws] = at(r0rwstab);
  const rws = rwsScale * x;
  const r00 = r0rws * rws;
  const dr00 = dr0rws * rws + r0rws * rwsScale * dxdv;
  const rp = 1.8 * rws;
  const drp = 1.8 * rwsScale * dxdv;

  const [evol0, devol0] = at(evol0tab);

  const y = new Float64Array(nr);
  const dy = new Float64Array(nr);
  for (let j = 0; j < nr; j++) {
    makespline(nx, nr, vpairtab, j, C);
    evalspline(nx - 1, x0, x1, C, x, sv);
    y[j] = sv.y;
    dy[j] = sv.dy * dxdv;
  }
  const vpairSpline = new Float64Array(4 * (nr - 1));
  const dvpairSpline = new Float64Array(4 * (nr - 1));
  makespline(nr, 1, y, 0, vpairSpline);
  makespline(nr, 1, dy, 0, dvpairSpline);

  return {
    va, vb, vc, vd, ve, p1, al, rp, r00, pn,
    dva, dvb, dvc, dvd, dve, dp1, dal, drp, dr00,
    evol0, devol0,
    vpairSpline, dvpairSpline, r0, r1, nr,
    mass, rcrit: rcrws * rws, rmax: rmrws * rws,
    lang, lmax, anorm3, anorm4, ddl, ipot, mode, metal, inputVol: vol,
  };
};
