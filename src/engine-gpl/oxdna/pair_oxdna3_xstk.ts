/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna3_xstk.cpp and
 * pair_oxdna3_xstk.h (class PairOxdna3Xstk, pair_style oxdna3/xstk): the oxDNA3 cross-stacking between the base
 * sites of two nucleotides on opposite strands. Unlike oxDNA (PairOxdnaXstk) the C++ is a separate class: the base
 * sites are type-dependent (NucleotideOxdna3::base_site<N>, N = type % 4), the cross-stacking strength and the
 * angle parameters of the 3' and 5' neighbour pairs are tetramer-dependent tables (the "33" set uses the 3'
 * neighbours of the two atoms, the "55" set the 5' neighbours; tetramerTable in common.ts), and the angles 7 and 8
 * have one theta0 for each set. The energy is f1 f2 f3 (f2_33 f4_4,33 f7,33 f8,33 + f2_55 f4_4,55 f7,55 f8,55) and
 * the C++ skips a term as soon as one of its factors vanishes (the early rejection criteria); the gating below
 * follows the C++ so that the zero-factor cases match. Original author (from the headers): Oliver Henrich
 * (University of Strathclyde, Glasgow); described by Ouldridge, Louis and Doye (J Chem Phys 2011). LAMMPS is
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext, potentialFile } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { OXDNA_3P, OXDNA_5P, isEllipsoid } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { DF2, DF4, F2, F4, baseSiteOxdna3, lrfOf, oxdnaConstants, tetIndex, tetramerTable, potentialLines } from './common';
import { sinFma as libmSin } from '../libm/sincos_fma';
import { pow as libmPow } from '../libm/pow';
import { acos as libmAcos } from '../libm/acos';

/** The per-tetramer tables of one neighbour set ("33" or "55"). */
const SET_KEYS = ['cut0', 'cutc', 'cutlo', 'cuthi', 'cutlc', 'cuthc', 'a4', 'th4', 'dth4ast', 'b4', 'dth4c'] as const;
/** The per-pair names of the single-valued tables. */
const PAIR_KEYS = ['k', 'blo', 'bhi', 'a1', 'th1', 'dth1ast', 'b1', 'dth1c', 'a2', 'th2', 'dth2ast', 'b2', 'dth2c',
  'a3', 'th3', 'dth3ast', 'b3', 'dth3c', 'a7', 'th7_33', 'th7_55', 'dth7ast', 'b7', 'dth7c',
  'a8', 'th8_33', 'th8_55', 'dth8ast', 'b8', 'dth8c'] as const;

const bang = (aa: number, ast: number): number => aa * aa * ast * ast / (1 - aa * ast * ast);
const cang = (aa: number, ast: number): number => 1 / (aa * ast);

export class PairOxdna3Xstk extends Pair {
  readonly name: string = 'oxdna3/xstk';
  // the global virial is the dot product of positions and forces (pair_oxdna3_xstk.cpp: virial_fdotr_compute)
  virialFdotr = true;
  private nt = 0;
  private set!: Uint8Array;
  private pair!: Record<string, Float64Array>;
  /** tetramer tables of the 33 and 55 sets, keys "<name>_33" and "<name>_55", indexed [i][j][k][l] with nt entries per axis */
  private tet!: Record<string, Float64Array>;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nt = ntypes + 1;
    const n2 = this.nt * this.nt, n4 = libmPow(this.nt, 4);
    this.pair = {};
    for (const k of PAIR_KEYS) this.pair[k] = new Float64Array(n2);
    this.tet = {};
    for (const set of ['33', '55']) for (const k of SET_KEYS) this.tet[`${k}_${set}`] = new Float64Array(n4);
    this.set = new Uint8Array(n2);
  }

  /**
   * pair_oxdna3_xstk.cpp coeff(): "pair_coeff I J file" only (narg 3). The file line "I J xstk" holds k_xst, the
   * n^4 tables cut_0, cut_c, cut_lo, cut_hi of the 33 set and of the 55 set, a1 theta1_0 dtheta1_ast a2 theta2_0
   * dtheta2_ast a3 theta3_0 dtheta3_ast, the n^4 tables a4, theta4_0, dtheta4_ast of the 33 set and of the 55 set,
   * then a7 theta7_0_33 theta7_0_55 dtheta7_ast a8 theta8_0_33 theta8_0_55 dtheta8_ast (the order of the C++ reader).
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Incorrect args for pair coefficients in oxdna3/xstk, use potential file');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    if (ilo !== jlo || ihi !== jhi) throw new StyleError('Incorrect args for pair coefficients in oxdna3/xstk');
    if (ihi > 4) throw new StyleError('pair oxdna3/xstk does not support more than 4 atom types for A, C, G and T');
    const n = ihi;
    const m4 = libmPow(n, 4);
    const need = 1 + 14 * m4 + 17;
    const text = ctx?.readFile(args[2], potentialFile('oxdna3 potential')) ?? '';
    let vals: number[] | null = null;
    for (const w of potentialLines(text, [args[0], args[1]], 'xstk')) {
      if (w.length < 3) continue;
      if (w[2] === 'xstk') {
        if (w.length < 3 + need) throw new StyleError(`Problem parsing oxdna3 potential file: xstk needs ${need} values in ${args[2]}`);
        vals = w.slice(3, 3 + need).map((x) => parseNum(x, 'xstk'));
        break;
      }
    }
    if (!vals) throw new StyleError(`No corresponding xstk potential found in file ${args[2]} for pair type ${args[0]} ${args[1]}`);
    let p = 0;
    const take = (count: number): number[] => { const r = vals!.slice(p, p + count); p += count; return r; };
    const [k] = take(1);
    const tabs: Record<string, Float64Array> = {};
    for (const set of ['33', '55']) {
      tabs[`cut0_${set}`] = tetramerTable(take(m4), n);
      tabs[`cutc_${set}`] = tetramerTable(take(m4), n);
      tabs[`cutlo_${set}`] = tetramerTable(take(m4), n);
      tabs[`cuthi_${set}`] = tetramerTable(take(m4), n);
    }
    const [a1, th1, dth1, a2, th2, dth2, a3, th3, dth3] = take(9);
    for (const set of ['33', '55']) {
      tabs[`a4_${set}`] = tetramerTable(take(m4), n);
      tabs[`th4_${set}`] = tetramerTable(take(m4), n);
      tabs[`dth4ast_${set}`] = tetramerTable(take(m4), n);
    }
    const [a7, th7_33, th7_55, dth7, a8, th8_33, th8_55, dth8] = take(8);

    // smoothing of the lo and hi tails: the strength and the cut tables of the AAAA tetramer (1, 1, 1, 1), 33 set
    const m = n + 1;
    const aaaa = tetIndex(m, 1, 1, 1, 1);
    const c0 = tabs.cut0_33[aaaa], cc = tabs.cutc_33[aaaa], clo = tabs.cutlo_33[aaaa], chi = tabs.cuthi_33[aaaa];
    const dlo = clo - c0, dhi = chi - c0, dc = c0 - cc;
    const bLo = 0.25 * dlo * dlo / (0.5 * dlo * dlo - k * 0.5 * dc * dc / k);
    const bHi = 0.25 * dhi * dhi / (0.5 * dhi * dhi - k * 0.5 * dc * dc / k);
    const b1 = bang(a1, dth1), c1 = cang(a1, dth1);
    const b2 = bang(a2, dth2), c2 = cang(a2, dth2);
    const b3 = bang(a3, dth3), c3 = cang(a3, dth3);
    const b7 = bang(a7, dth7), c7 = cang(a7, dth7);
    const b8 = bang(a8, dth8), c8 = cang(a8, dth8);

    const nt = this.nt;
    const P = this.pair;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = jlo; j <= jhi; j++) {
        const q = i * nt + j;
        const vals2: Record<string, number> = {
          k, blo: bLo, bhi: bHi, a1, th1, dth1ast: dth1, b1, dth1c: c1, a2, th2, dth2ast: dth2, b2, dth2c: c2,
          a3, th3, dth3ast: dth3, b3, dth3c: c3, a7, th7_33, th7_55, dth7ast: dth7, b7, dth7c: c7,
          a8, th8_33, th8_55, dth8ast: dth8, b8, dth8c: c8,
        };
        for (const key of PAIR_KEYS) P[key][q] = vals2[key];
        this.set[q] = 1;
      }
    }
    // tetramer parameters: i and l from the terminal type 0, j and k in the range (the loops of the C++ coeff)
    for (const set of ['33', '55']) {
      const T = this.tet;
      for (let i = 0; i <= n; i++) {
        for (let j = 1; j <= n; j++) {
          for (let kk = 1; kk <= n; kk++) {
            for (let l = 0; l <= n; l++) {
              const src = tetIndex(m, i, j, kk, l);
              const q = ((i * nt + j) * nt + kk) * nt + l;
              const cut0 = tabs[`cut0_${set}`][src], cutc = tabs[`cutc_${set}`][src];
              const cutlo = tabs[`cutlo_${set}`][src], cuthi = tabs[`cuthi_${set}`][src];
              T[`cut0_${set}`][q] = cut0;
              T[`cutc_${set}`][q] = cutc;
              T[`cutlo_${set}`][q] = cutlo;
              T[`cuthi_${set}`][q] = cuthi;
              T[`cutlc_${set}`][q] = cutlo - 0.5 * (cutlo - cut0) / bLo;
              T[`cuthc_${set}`][q] = cuthi - 0.5 * (cuthi - cut0) / bHi;
              const a4 = tabs[`a4_${set}`][src], dth4 = tabs[`dth4ast_${set}`][src];
              T[`a4_${set}`][q] = a4;
              T[`th4_${set}`][q] = tabs[`th4_${set}`][src];
              T[`dth4ast_${set}`][q] = dth4;
              T[`b4_${set}`][q] = a4 * a4 * dth4 * dth4 / (1 - a4 * dth4 * dth4);
              T[`dth4c_${set}`][q] = 1 / (a4 * dth4);
            }
          }
        }
      }
    }
  }

  /** pair_oxdna3_xstk.cpp init_one(): cut_max over cut_xst_hc_33 and _55 [a][i][j][b] for a, b below ntypes. */
  initOne(i: number, j: number): number {
    const nt = this.nt;
    if (!this.set[i * nt + j]) throw new StyleError('Coefficient mixing not defined in oxDNA');
    let cutMax = 0.0;
    for (let a = 0; a < this.ntypes; a++) {
      for (let b = 0; b < this.ntypes; b++) {
        const q = ((a * nt + i) * nt + j) * nt + b;
        cutMax = Math.max(this.tet.cuthc_33[q], cutMax);
        cutMax = Math.max(this.tet.cuthc_55[q], cutMax);
      }
    }
    return cutMax;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.custom.has(OXDNA_5P) || !s.quat || !s.torque) throw new StyleError('pair_style oxdna3/xstk requires atom_style hybrid bond ellipsoid oxdna');
    const list = pc.half!;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nall = pc.nall;
    const nlocal = pc.nlocal;
    const nt = this.nt;
    const P = this.pair;
    const T = this.tet;
    const constants = oxdnaConstants(s.units.style);
    const special = pc.specialLJ;
    const tq = new Float64Array(3 * nall);
    const eatom = pc.eatom, vatom = pc.vatom;
    const id5p = s.custom.get(OXDNA_5P)!.data, id3p = s.custom.get(OXDNA_3P)!.data;
    // local reference frames (fix OXDNA/LRF; ghosts take the frame of their owner) and the base sites by type
    const L = new Float64Array(9 * nall);
    for (let k = 0; k < nall; k++) {
      const o = owner[k];
      if (isEllipsoid(s, o)) lrfOf(s.quat, 4 * o, L, 9 * k);
    }
    const rbs = new Float64Array(3 * nall);
    const tmp = [0, 0, 0];
    for (let k = 0; k < nall; k++) {
      baseSiteOxdna3(constants, type[k], L.subarray(9 * k, 9 * k + 3), tmp);
      rbs[3 * k] = tmp[0]; rbs[3 * k + 1] = tmp[1]; rbs[3 * k + 2] = tmp[2];
    }
    // the type of the atom with a tag (0 = no neighbour)
    const typeOfTag = new Map<number, number>();
    for (let k = 0; k < nlocal; k++) typeOfTag.set(s.id[k], type[k]);
    const partner = (id: number): number => (id ? (typeOfTag.get(id) ?? 0) : 0);

    let evdwl = 0;
    const delr = [0, 0, 0], drn = [0, 0, 0], delf = [0, 0, 0], delta = [0, 0, 0], deltb = [0, 0, 0], dir = [0, 0, 0];
    const ax = [0, 0, 0], az = [0, 0, 0], bx = [0, 0, 0], bz = [0, 0, 0];
    const cross = (u: readonly number[], v: readonly number[], out: number[]) => {
      out[0] = u[1] * v[2] - u[2] * v[1]; out[1] = u[2] * v[0] - u[0] * v[2]; out[2] = u[0] * v[1] - u[1] * v[0];
    };
    const clamp = (c: number) => (c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c);
    const dot = (u: readonly number[], v: readonly number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    for (let a = 0; a < list.inum; a++) {
      const atype = type[a];
      const a3a = partner(id3p[a]), a5a = partner(id5p[a]);
      for (let c = 0; c < 3; c++) { ax[c] = L[9 * a + c]; az[c] = L[9 * a + 6 + c]; }
      const ra = [rbs[3 * a], rbs[3 * a + 1], rbs[3 * a + 2]];
      const jstart = list.firstneigh[a], jend = jstart + list.numneigh[a];
      for (let kk = jstart; kk < jend; kk++) {
        const jj = list.neighbors[kk];
        const factorLJ = special[jj >>> SBBITS];
        const b = jj & NEIGHMASK;
        const btype = type[b];
        const b3b = partner(id3p[b]), b5b = partner(id5p[b]);
        for (let c = 0; c < 3; c++) { bx[c] = L[9 * b + c]; bz[c] = L[9 * b + 6 + c]; }
        const rb = [rbs[3 * b], rbs[3 * b + 1], rbs[3 * b + 2]];
        for (let c = 0; c < 3; c++) delr[c] = x[3 * a + c] + ra[c] - x[3 * b + c] - rb[c];
        const rsq = delr[0] * delr[0] + delr[1] * delr[1] + delr[2] * delr[2];
        const r = Math.sqrt(rsq);
        const rinv = 1.0 / r;
        for (let c = 0; c < 3; c++) drn[c] = delr[c] * rinv;
        const ij = atype * nt + btype;
        const q33 = ((a3a * nt + atype) * nt + btype) * nt + b3b;
        const q55 = ((a5a * nt + atype) * nt + btype) * nt + b5b;
        const kx = P.k[ij], blo = P.blo[ij], bhi = P.bhi[ij];
        const f2_33 = F2(r, kx, T.cut0_33[q33], T.cutlc_33[q33], T.cuthc_33[q33], T.cutlo_33[q33], T.cuthi_33[q33], blo, bhi, T.cutc_33[q33]);
        const f2_55 = F2(r, kx, T.cut0_55[q55], T.cutlc_55[q55], T.cuthc_55[q55], T.cutlo_55[q55], T.cuthi_55[q55], blo, bhi, T.cutc_55[q55]);
        if (f2_33 === 0.0 && f2_55 === 0.0) continue;
        const cost1 = clamp(-dot(ax, bx));
        const theta1 = libmAcos(cost1);
        const f4t1 = F4(theta1, P.a1[ij], P.th1[ij], P.dth1ast[ij], P.b1[ij], P.dth1c[ij]);
        if (f4t1 === 0.0) continue;
        const cost2 = clamp(-dot(ax, drn));
        const theta2 = libmAcos(cost2);
        const f4t2 = F4(theta2, P.a2[ij], P.th2[ij], P.dth2ast[ij], P.b2[ij], P.dth2c[ij]);
        if (f4t2 === 0.0) continue;
        const cost3 = clamp(dot(bx, drn));
        const theta3 = libmAcos(cost3);
        const f4t3 = F4(theta3, P.a3[ij], P.th3[ij], P.dth3ast[ij], P.b3[ij], P.dth3c[ij]);
        if (f4t3 === 0.0) continue;
        const cost4 = clamp(dot(az, bz));
        const theta4 = libmAcos(cost4);
        const f4t4_33 = F4(theta4, T.a4_33[q33], T.th4_33[q33], T.dth4ast_33[q33], T.b4_33[q33], T.dth4c_33[q33]);
        const f4t4_55 = F4(theta4, T.a4_55[q55], T.th4_55[q55], T.dth4ast_55[q55], T.b4_55[q55], T.dth4c_55[q55]);
        if (f4t4_33 === 0.0 && f4t4_55 === 0.0) continue;
        const cost7 = clamp(-dot(az, drn));
        const theta7 = libmAcos(cost7);
        const f4t7_33 = F4(theta7, P.a7[ij], P.th7_33[ij], P.dth7ast[ij], P.b7[ij], P.dth7c[ij]);
        const f4t7_55 = F4(theta7, P.a7[ij], P.th7_55[ij], P.dth7ast[ij], P.b7[ij], P.dth7c[ij]);
        if (f4t7_33 === 0.0 && f4t7_55 === 0.0) continue;
        const cost8 = clamp(dot(bz, drn));
        const theta8 = libmAcos(cost8);
        const f4t8_33 = F4(theta8, P.a8[ij], P.th8_33[ij], P.dth8ast[ij], P.b8[ij], P.dth8c[ij]);
        const f4t8_55 = F4(theta8, P.a8[ij], P.th8_55[ij], P.dth8ast[ij], P.b8[ij], P.dth8c[ij]);
        // energy: f4t1 f4t2 f4t3 (f2_33 f4t4_33 f4t7_33 f4t8_33 + f2_55 f4t4_55 f4t7_55 f4t8_55) factor_lj
        const sum = f2_33 * f4t4_33 * f4t7_33 * f4t8_33 + f2_55 * f4t4_55 * f4t7_55 * f4t8_55;
        const e = f4t1 * f4t2 * f4t3 * sum * factorLJ;
        if (e === 0.0) continue;
        // derivatives (the C++ divides the angle derivatives by sin(theta), the theta4, 7 and 8 ones by it as well)
        const df2_33 = DF2(r, kx, T.cut0_33[q33], T.cutlc_33[q33], T.cuthc_33[q33], T.cutlo_33[q33], T.cuthi_33[q33], blo, bhi);
        const df2_55 = DF2(r, kx, T.cut0_55[q55], T.cutlc_55[q55], T.cuthc_55[q55], T.cutlo_55[q55], T.cuthi_55[q55], blo, bhi);
        const df4t1 = DF4(theta1, P.a1[ij], P.th1[ij], P.dth1ast[ij], P.b1[ij], P.dth1c[ij]) / libmSin(theta1);
        const df4t2 = DF4(theta2, P.a2[ij], P.th2[ij], P.dth2ast[ij], P.b2[ij], P.dth2c[ij]) / libmSin(theta2);
        const df4t3 = DF4(theta3, P.a3[ij], P.th3[ij], P.dth3ast[ij], P.b3[ij], P.dth3c[ij]) / libmSin(theta3);
        const rsint4 = 1.0 / libmSin(theta4);
        const df4t4_33 = DF4(theta4, T.a4_33[q33], T.th4_33[q33], T.dth4ast_33[q33], T.b4_33[q33], T.dth4c_33[q33]) * rsint4;
        const df4t4_55 = DF4(theta4, T.a4_55[q55], T.th4_55[q55], T.dth4ast_55[q55], T.b4_55[q55], T.dth4c_55[q55]) * rsint4;
        const rsint7 = 1.0 / libmSin(theta7);
        const df4t7_33 = DF4(theta7, P.a7[ij], P.th7_33[ij], P.dth7ast[ij], P.b7[ij], P.dth7c[ij]) * rsint7;
        const df4t7_55 = DF4(theta7, P.a7[ij], P.th7_55[ij], P.dth7ast[ij], P.b7[ij], P.dth7c[ij]) * rsint7;
        const rsint8 = 1.0 / libmSin(theta8);
        const df4t8_33 = DF4(theta8, P.a8[ij], P.th8_33[ij], P.dth8ast[ij], P.b8[ij], P.dth8c[ij]) * rsint8;
        const df4t8_55 = DF4(theta8, P.a8[ij], P.th8_55[ij], P.dth8ast[ij], P.b8[ij], P.dth8c[ij]) * rsint8;
        // the sums with one factor differentiated: r (df2), theta4, theta7 and theta8
        const sumR = df2_33 * f4t4_33 * f4t7_33 * f4t8_33 + df2_55 * f4t4_55 * f4t7_55 * f4t8_55;
        const sumT4 = f2_33 * df4t4_33 * f4t7_33 * f4t8_33 + f2_55 * df4t4_55 * f4t7_55 * f4t8_55;
        const sumT7 = f2_33 * f4t4_33 * df4t7_33 * f4t8_33 + f2_55 * f4t4_55 * df4t7_55 * f4t8_55;
        const sumT8 = f2_33 * f4t4_33 * f4t7_33 * df4t8_33 + f2_55 * f4t4_55 * f4t7_55 * df4t8_55;

        delf[0] = 0.0; delf[1] = 0.0; delf[2] = 0.0;
        delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
        deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
        // radial force
        let finc = -f4t1 * f4t2 * f4t3 * sumR * rinv * factorLJ;
        for (let c = 0; c < 3; c++) delf[c] += delr[c] * finc;
        // theta2 and theta3 forces (the energy sum)
        if (theta2 !== 0.0) {
          finc = -f4t1 * df4t2 * f4t3 * sum * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost2 + ax[c]) * finc;
        }
        if (theta3 !== 0.0) {
          finc = -f4t1 * f4t2 * df4t3 * sum * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost3 - bx[c]) * finc;
        }
        // theta7 and theta8 forces
        if (theta7 !== 0.0) {
          finc = -f4t1 * f4t2 * f4t3 * sumT7 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost7 + az[c]) * finc;
        }
        if (theta8 !== 0.0) {
          finc = -f4t1 * f4t2 * f4t3 * sumT8 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost8 - bz[c]) * finc;
        }
        for (let c = 0; c < 3; c++) f[3 * a + c] += delf[c];
        cross(ra, delf, delta);
        for (let c = 0; c < 3; c++) tq[3 * a + c] += delta[c];
        for (let c = 0; c < 3; c++) f[3 * b + c] -= delf[c];
        cross(rb, delf, deltb);
        for (let c = 0; c < 3; c++) tq[3 * b + c] -= deltb[c];
        evdwl += e;
        if (eatom) { eatom[a] += 0.5 * e; eatom[b] += 0.5 * e; }
        if (vatom) {
          const rx = x[3 * a] - x[3 * b], ry = x[3 * a + 1] - x[3 * b + 1], rz = x[3 * a + 2] - x[3 * b + 2];
          const w = [rx * delf[0], ry * delf[1], rz * delf[2], rx * delf[1], rx * delf[2], ry * delf[2]];
          for (let c = 0; c < 6; c++) { vatom[6 * a + c] += 0.5 * w[c]; vatom[6 * b + c] += 0.5 * w[c]; }
        }
        // pure torques (not expressible as r x f)
        delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
        deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
        const pre = f4t1 * f4t2 * f4t3 * factorLJ;
        if (theta1 !== 0.0) {
          const tpair = -df4t1 * f4t2 * f4t3 * sum * factorLJ;
          cross(ax, bx, dir);
          for (let c = 0; c < 3; c++) { delta[c] += dir[c] * tpair; deltb[c] += dir[c] * tpair; }
        }
        if (theta2 !== 0.0) {
          const tpair = -f4t1 * df4t2 * f4t3 * sum * factorLJ;
          cross(ax, drn, dir);
          for (let c = 0; c < 3; c++) delta[c] += dir[c] * tpair;
        }
        if (theta3 !== 0.0) {
          const tpair = -f4t1 * f4t2 * df4t3 * sum * factorLJ;
          cross(bx, drn, dir);
          for (let c = 0; c < 3; c++) deltb[c] += dir[c] * tpair;
        }
        if (theta4 !== 0.0) {
          const tpair = -pre * sumT4;
          cross(bz, az, dir);
          for (let c = 0; c < 3; c++) { delta[c] += dir[c] * tpair; deltb[c] += dir[c] * tpair; }
        }
        if (theta7 !== 0.0) {
          const tpair = -pre * sumT7;
          cross(az, drn, dir);
          for (let c = 0; c < 3; c++) delta[c] += dir[c] * tpair;
        }
        if (theta8 !== 0.0) {
          const tpair = -pre * sumT8;
          cross(bz, drn, dir);
          for (let c = 0; c < 3; c++) deltb[c] += dir[c] * tpair;
        }
        for (let c = 0; c < 3; c++) { tq[3 * a + c] += delta[c]; tq[3 * b + c] -= deltb[c]; }
      }
    }
    pc.nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}
