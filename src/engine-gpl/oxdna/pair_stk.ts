/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna_stk.cpp and
 * pair_oxdna_stk.h (class PairOxdnaStk, pair_style oxdna/stk): the oxDNA stacking interaction between
 * nucleotides that are neighbours along a strand, with the angular modulation of the base and stacking sites.
 * Original authors (from the headers): Oliver Henrich (University of Strathclyde, Glasgow) and Kierran Falloon
 * (University of Strathclyde, Glasgow); described by Ouldridge, Louis and Doye (J Chem Phys 2011). LAMMPS is
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0);
 * this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The stacking term runs over the FENE backbone bonds (the bond list), not over the pair list: the 3' side of a
 * bond is the atom whose 5' partner is the other atom (pair_oxdna_stk.cpp, compute).
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { OXDNA_5P, OXDNA_3P, buildAtomMap, isEllipsoid } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { DF1, DF4, DF5, F1, F4, F5, lrfOf, oxdnaConstants, stackingSiteOxdna1, backboneSite, type StackRule, potentialLines } from './common';
import { sin as libmSin } from '../libm/sincos';

/** pair_oxdna_stk.cpp constructor: the sequence-dependence factors eta_st[i][j] (indexed by 1-based base types mod 4). */
export const ETA_ST: number[][] = [
  [1.11960, 1.01889, 0.98169, 0.94694],
  [1.00852, 0.97804, 1.05913, 0.98169],
  [0.96950, 1.02681, 0.97804, 1.01889],
  [0.99632, 0.96950, 1.00852, 0.96383],
];

/** pair_oxdna_stk.cpp stacking_strength(xi, kappa, T) = xi + kappa T. */
export const stackingStrength = (xi: number, kappa: number, T: number): number => xi + kappa * T;


/**
 * pair_oxdna_stk.cpp coeff(): the lo and hi quadratic-tail coefficients b_st_lo and b_st_hi, their cut_lc and
 * cut_hc, and the shift of the exponential term (the same formulas for oxRNA2 stk, pair_oxrna2_stk.cpp).
 */
export const stackingDerived = (eps: number, a: number, cut0: number, cutC: number, cutLo: number, cutHi: number) => {
  const e = (x: number) => Math.exp(-a * x);
  const ulo = cutLo - cut0, uhi = cutHi - cut0, uc = cutC - cut0;
  const bLo = 2 * a * e(ulo) * 2 * a * e(ulo) * (1 - e(ulo)) * (1 - e(ulo))
    / (4 * ((1 - e(ulo)) * (1 - e(ulo)) - (1 - e(uc)) * (1 - e(uc))));
  const cLc = cutLo - a * e(ulo) * (1 - e(ulo)) / bLo;
  const bHi = 2 * a * e(uhi) * 2 * a * e(uhi) * (1 - e(uhi)) * (1 - e(uhi))
    / (4 * ((1 - e(uhi)) * (1 - e(uhi)) - (1 - e(uc)) * (1 - e(uc))));
  const cHc = cutHi - a * e(uhi) * (1 - e(uhi)) / bHi;
  const tmp = 1 - Math.exp(-(cutC - cut0) * a);
  const shiftOne = eps * tmp * tmp;
  return { bLo, cLc, bHi, cHc, shiftOne };
};

/** The 20 stacking numbers after "I J seqav|seqdep T" (pair_coeff) or after the potential-file name. */
const STK_NAMES = ['xi_st', 'kappa_st', 'a_st', 'cut_st_0', 'cut_st_c', 'cut_st_lo', 'cut_st_hi', 'a_st4', 'theta_st4_0', 'dtheta_st4_ast',
  'a_st5', 'theta_st5_0', 'dtheta_st5_ast', 'a_st6', 'theta_st6_0', 'dtheta_st6_ast', 'a_st1', 'cosphi_st1_ast', 'a_st2', 'cosphi_st2_ast'];

type Tab2 = Float64Array;

export class PairOxdnaStk extends Pair {
  readonly name: string = 'oxdna/stk';
  /** the stacking site rule of the nucleotide model (the oxDNA3 subclass has its own offset) */
  protected stackSite: StackRule = stackingSiteOxdna1;
  // the global virial is the dot product of positions and forces (pair_oxdna_stk.cpp: virial_fdotr_compute)
  virialFdotr = true;
  /** the bond list is the loop; no pair list is needed (pair_oxdna_stk.cpp init_style adds no request) */
  needsHalf = false;
  protected nt = 0;
  /** eta_st[i][j] of the C++ (the oxDNA table; the oxDNA3 subclass has its own sequence table) */
  protected etaSt: number[][] = ETA_ST;
  protected seqdep = false;
  protected eps!: Tab2; protected aSt!: Tab2; protected bLo!: Tab2; protected bHi!: Tab2;
  protected th4!: Tab2; protected a5!: Tab2; protected th5!: Tab2; protected dth5ast!: Tab2; protected b5!: Tab2; protected dth5c!: Tab2;
  protected a6!: Tab2; protected th6!: Tab2; protected dth6ast!: Tab2; protected b6!: Tab2; protected dth6c!: Tab2;
  protected a1!: Tab2; protected cos1ast!: Tab2; protected b1!: Tab2; protected cos1c!: Tab2;
  protected a2!: Tab2; protected cos2ast!: Tab2; protected b2!: Tab2; protected cos2c!: Tab2;
  protected set!: Uint8Array;
  // tetramer tables [i][j][k][l]
  protected c0!: Float64Array; protected clc!: Float64Array; protected chc!: Float64Array; protected clo!: Float64Array; protected chi!: Float64Array;
  protected shiftSt!: Float64Array; protected a4!: Float64Array; protected dth4ast!: Float64Array; protected b4!: Float64Array; protected dth4c!: Float64Array;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nt = ntypes + 1;
    const n = this.nt * this.nt;
    const z = () => new Float64Array(n);
    this.eps = z(); this.aSt = z(); this.bLo = z(); this.bHi = z(); this.th4 = z();
    this.a5 = z(); this.th5 = z(); this.dth5ast = z(); this.b5 = z(); this.dth5c = z();
    this.a6 = z(); this.th6 = z(); this.dth6ast = z(); this.b6 = z(); this.dth6c = z();
    this.a1 = z(); this.cos1ast = z(); this.b1 = z(); this.cos1c = z();
    this.a2 = z(); this.cos2ast = z(); this.b2 = z(); this.cos2c = z();
    this.set = new Uint8Array(n);
    const n4 = this.nt ** 4;
    const z4 = () => new Float64Array(n4);
    this.c0 = z4(); this.clc = z4(); this.chc = z4(); this.clo = z4(); this.chi = z4();
    this.shiftSt = z4(); this.a4 = z4(); this.dth4ast = z4(); this.b4 = z4(); this.dth4c = z4();
  }

  /**
   * pair_oxdna_stk.cpp coeff(): "pair_coeff I J seqav|seqdep T xi kappa a cut_0 cut_c cut_lo cut_hi a4 theta4_0
   * dtheta4_ast a5 theta5_0 dtheta5_ast a6 theta6_0 dtheta6_ast a1 cosphi1_ast a2 cosphi2_ast" (24 words) or
   * "pair_coeff I J seqav|seqdep T potential-file" (5 words).
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 24 && args.length !== 5) throw new StyleError('Incorrect args for pair coefficients in oxdna/stk');
    if (args[2] !== 'seqav' && args[2] !== 'seqdep') throw new StyleError('Incorrect setting, select seqav or seqdep in oxdna/stk');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    typeBounds(args[1], this.ntypes);
    this.seqdep = args[2] === 'seqdep';
    const T = parseNum(args[3], 'T');
    let v: number[];
    if (args.length === 24) {
      v = args.slice(4).map((w, k) => parseNum(w, STK_NAMES[k]));
    } else {
      const text = ctx?.readFile(args[4]) ?? '';
      let found: number[] | null = null;
      for (const w of potentialLines(text, [args[0], args[1]], 'stk')) {
        if (w.length < 3) continue;
        if (w[2] === 'stk') {
          if (w.length < 23) throw new StyleError(`Problem parsing oxDNA potential file: stk needs 20 values in ${args[4]}`);
          found = w.slice(3, 23).map((x, k) => parseNum(x, STK_NAMES[k]));
          break;
        }
      }
      if (!found) throw new StyleError(`No corresponding stk potential found in file ${args[4]} for pair type ${args[0]} ${args[1]}`);
      v = found;
    }
    const [xi, kappa, a, cut0, cutC, cutLo, cutHi, a4, th4_0, dth4ast, a5, th5_0, dth5ast, a6, th6_0, dth6ast, a1, cos1ast, a2, cos2ast] = v;
    const eps = stackingStrength(xi, kappa, T);
    const { bLo, cLc, bHi, cHc, shiftOne } = stackingDerived(eps, a, cut0, cutC, cutLo, cutHi);
    const b4 = a4 * a4 * dth4ast * dth4ast / (1 - a4 * dth4ast * dth4ast);
    const dth4c = 1 / (a4 * dth4ast);
    const b5 = a5 * a5 * dth5ast * dth5ast / (1 - a5 * dth5ast * dth5ast);
    const dth5c = 1 / (a5 * dth5ast);
    const b6 = a6 * a6 * dth6ast * dth6ast / (1 - a6 * dth6ast * dth6ast);
    const dth6c = 1 / (a6 * dth6ast);
    const b1 = a1 * a1 * cos1ast * cos1ast / (1 - a1 * cos1ast * cos1ast);
    const cos1c = 1 / (a1 * cos1ast);
    const b2 = a2 * a2 * cos2ast * cos2ast / (1 - a2 * cos2ast * cos2ast);
    const cos2c = 1 / (a2 * cos2ast);
    const nt = this.nt;
    const mod4 = (t: number) => (t % 4 === 0 ? 4 : t % 4);
    for (let i = ilo; i <= ihi; i++) {
      const im = mod4(i);
      for (let j = ilo; j <= ihi; j++) {
        const jm = mod4(j);
        const q = i * nt + j;
        this.eps[q] = eps;
        if (this.seqdep) this.eps[q] *= this.etaSt[im - 1][jm - 1];
        this.aSt[q] = a; this.bLo[q] = bLo; this.bHi[q] = bHi;
        this.th4[q] = th4_0;
        this.a5[q] = a5; this.th5[q] = th5_0; this.dth5ast[q] = dth5ast; this.b5[q] = b5; this.dth5c[q] = dth5c;
        this.a6[q] = a6; this.th6[q] = th6_0; this.dth6ast[q] = dth6ast; this.b6[q] = b6; this.dth6c[q] = dth6c;
        this.a1[q] = a1; this.cos1ast[q] = cos1ast; this.b1[q] = b1; this.cos1c[q] = cos1c;
        this.a2[q] = a2; this.cos2ast[q] = cos2ast; this.b2[q] = b2; this.cos2c[q] = cos2c;
      }
    }
    // tetramer tables: [i][j][k][l] with i and l from the terminal type 0, j and k in the range
    for (let i = 0; i <= ihi; i++) {
      for (let j = ilo; j <= ihi; j++) {
        const jm = mod4(j);
        for (let k = ilo; k <= ihi; k++) {
          const km = mod4(k);
          for (let l = 0; l <= ihi; l++) {
            const q = ((i * nt + j) * nt + k) * nt + l;
            this.c0[q] = cut0; this.chc[q] = cHc; this.clc[q] = cLc; this.clo[q] = cutLo; this.chi[q] = cutHi;
            this.shiftSt[q] = shiftOne;
            if (this.seqdep) this.shiftSt[q] *= this.etaSt[jm - 1][km - 1];
            this.a4[q] = a4; this.dth4ast[q] = dth4ast; this.b4[q] = b4; this.dth4c[q] = dth4c;
          }
        }
      }
    }
    for (let i = ilo; i <= ihi; i++) for (let j = ilo; j <= ihi; j++) this.set[i * nt + j] = 1;
  }

  init(ctx: StyleContext): void {
    // pair_oxdna_stk.cpp init_style(): "Must use 'atom_style hybrid bond ellipsoid oxdna' with pair style oxdna/stk, ..."
    if (ctx.s && !ctx.s.custom.has(OXDNA_5P)) {
      throw new StyleError("Must use 'atom_style hybrid bond ellipsoid oxdna' with pair style oxdna/stk, oxdna2/stk or oxrna2/stk");
    }
    super.init(ctx);
  }

  /** pair_oxdna_stk.cpp init_one(): cut_max over cut_st_hc[a][i][j][b] for a, b in 0..ntypes-1. */
  initOne(i: number, j: number): number {
    if (!this.set[i * this.nt + j]) throw new StyleError('Coefficient mixing not defined in oxDNA');
    let cutMax = 0.0;
    for (let a = 0; a < this.ntypes; a++) {
      for (let b = 0; b < this.ntypes; b++) {
        cutMax = Math.max(this.chc[((a * this.nt + i) * this.nt + j) * this.nt + b], cutMax);
      }
    }
    return cutMax;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const aMap = s.custom.get(OXDNA_5P);
    if (!aMap || !s.quat || !s.torque) throw new StyleError('pair_style oxdna/stk requires atom_style hybrid bond ellipsoid oxdna');
    const bonds = s.topo.bonds;
    const id5p = aMap.data, id3p = s.custom.get(OXDNA_3P)!.data;
    const x = pc.x, f = pc.f, type = pc.type;
    const nt = this.nt;
    const constants = oxdnaConstants(s.units.style);
    const dxCbk = constants.dx_cbk_oxdna1;
    const torque = s.torque;
    const map = buildAtomMap(s);
    // local reference frames of the owned atoms (fix OXDNA/LRF)
    const nlocal = pc.nlocal;
    const L = new Float64Array(9 * nlocal);
    for (let i = 0; i < nlocal; i++) if (isEllipsoid(s, i)) lrfOf(s.quat, 4 * i, L, 9 * i);
    const eatom = pc.eatom;
    let evdwl = 0;
    const ax = [0, 0, 0], ay = [0, 0, 0], az = [0, 0, 0], bx = [0, 0, 0], by = [0, 0, 0], bz = [0, 0, 0];
    const rac = [0, 0, 0], rbc = [0, 0, 0], rab = [0, 0, 0], rbb = [0, 0, 0];
    const dStk = [0, 0, 0], dStkN = [0, 0, 0], dBk = [0, 0, 0], dBkN = [0, 0, 0];
    const d = [0, 0, 0];
    const xb = [0, 0, 0];
    const addF = (i: number, v: number[], sgn: number) => { f[3 * i] += sgn * v[0]; f[3 * i + 1] += sgn * v[1]; f[3 * i + 2] += sgn * v[2]; };
    const addT = (i: number, v: number[], sgn: number) => { torque[3 * i] += sgn * v[0]; torque[3 * i + 1] += sgn * v[1]; torque[3 * i + 2] += sgn * v[2]; };
    const cross = (u: readonly number[], v: readonly number[], out: number[]) => {
      out[0] = u[1] * v[2] - u[2] * v[1]; out[1] = u[2] * v[0] - u[0] * v[2]; out[2] = u[0] * v[1] - u[1] * v[0];
    };
    const clamp = (c: number) => (c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c);
    const dot = (u: readonly number[], v: readonly number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const delf = [0, 0, 0], delta = [0, 0, 0], deltb = [0, 0, 0], t4dir = [0, 0, 0], t5pdir = [0, 0, 0], t6pdir = [0, 0, 0];
    const cphi1dir = [0, 0, 0], cphi2dir = [0, 0, 0];
    for (let k = 0; k < bonds.n; k++) {
      // bondlist[][0] = atom1, [][1] = atom2 (newton_bond on)
      let a = indexOf(map, bonds.atoms[2 * k]);
      let b = indexOf(map, bonds.atoms[2 * k + 1]);
      // directionality: a -> b is 3' -> 5' when tag[b] == id5p[a]
      if (s.id[b] !== id5p[a]) { const t = a; a = b; b = t; }
      for (let c = 0; c < 3; c++) {
        ax[c] = L[9 * a + c]; ay[c] = L[9 * a + 3 + c]; az[c] = L[9 * a + 6 + c];
        bx[c] = L[9 * b + c]; by[c] = L[9 * b + 3 + c]; bz[c] = L[9 * b + 6 + c];
      }
      // closest image of b relative to a
      d[0] = x[3 * b] - x[3 * a]; d[1] = x[3 * b + 1] - x[3 * a + 1]; d[2] = x[3 * b + 2] - x[3 * a + 2];
      pc.geom.minimumImage(d);
      xb[0] = x[3 * a] + d[0]; xb[1] = x[3 * a + 1] + d[1]; xb[2] = x[3 * a + 2] + d[2];
      // stacking sites
      backboneSite(dxCbk, ax, rab); backboneSite(dxCbk, bx, rbb);
      this.stackSite(constants, ax, ay, az, rac); this.stackSite(constants, bx, by, bz, rbc);
      dStk[0] = xb[0] + rbc[0] - x[3 * a] - rac[0];
      dStk[1] = xb[1] + rbc[1] - x[3 * a + 1] - rac[1];
      dStk[2] = xb[2] + rbc[2] - x[3 * a + 2] - rac[2];
      const a3p = id3p[a] !== 0 ? (type[map[id3p[a]]] ?? 0) : 0;
      const atype = type[a], btype = type[b];
      const b5p = id5p[b] !== 0 ? (type[map[id5p[b]]] ?? 0) : 0;
      const rsqStk = dStk[0] * dStk[0] + dStk[1] * dStk[1] + dStk[2] * dStk[2];
      const rStk = Math.sqrt(rsqStk);
      const rinvStk = 1.0 / rStk;
      for (let c = 0; c < 3; c++) dStkN[c] = dStk[c] * rinvStk;
      // backbone sites (for the cos phi terms)
      for (let c = 0; c < 3; c++) { dBk[c] = xb[c] + rbb[c] - x[3 * a + c] - rab[c]; }
      const rsqBk = dBk[0] * dBk[0] + dBk[1] * dBk[1] + dBk[2] * dBk[2];
      const rBk = Math.sqrt(rsqBk);
      const rinvBk = 1.0 / rBk;
      for (let c = 0; c < 3; c++) dBkN[c] = dBk[c] * rinvBk;

      const qe = atype * nt + btype; // 2D index
      const q4 = ((a3p * nt + atype) * nt + btype) * nt + b5p;
      const f1 = F1(rStk, this.eps[qe], this.aSt[qe], this.c0[q4], this.clc[q4], this.chc[q4], this.clo[q4], this.chi[q4],
        this.bLo[qe], this.bHi[qe], this.shiftSt[q4]);
      if (f1 !== 0.0) {
        // dot products for the angles (the local z axes of a and b)
        const cost4 = clamp(dot(bz, az));
        const theta4 = Math.acos(cost4);
        const f4t4 = F4(theta4, this.a4[q4], this.th4[qe], this.dth4ast[q4], this.b4[q4], this.dth4c[q4]);
        if (f4t4 !== 0.0) {
          const cost5p = clamp(dot(dStkN, bz));
          const theta5p = Math.acos(cost5p);
          const f4t5 = F4(theta5p, this.a5[qe], this.th5[qe], this.dth5ast[qe], this.b5[qe], this.dth5c[qe]);
          if (f4t5 !== 0.0) {
            const cost6p = clamp(dot(dStkN, az));
            const theta6p = Math.acos(cost6p);
            const cosphi1 = clamp(dot(dBkN, by));
            const cosphi2 = clamp(dot(dBkN, ay));
            const f4t6 = F4(theta6p, this.a6[qe], this.th6[qe], this.dth6ast[qe], this.b6[qe], this.dth6c[qe]);
            const f5c1 = F5(-cosphi1, this.a1[qe], -this.cos1ast[qe], this.b1[qe], -this.cos1c[qe]);
            const f5c2 = F5(-cosphi2, this.a2[qe], -this.cos2ast[qe], this.b2[qe], -this.cos2c[qe]);
            evdwl = f1 * f4t4 * f4t5 * f4t6 * f5c1 * f5c2;
            if (evdwl !== 0.0) {
              const df1 = DF1(rStk, this.eps[qe], this.aSt[qe], this.c0[q4], this.clc[q4], this.chc[q4], this.clo[q4], this.chi[q4],
                this.bLo[qe], this.bHi[qe]);
              const df4t4 = DF4(theta4, this.a4[q4], this.th4[qe], this.dth4ast[q4], this.b4[q4], this.dth4c[q4]) / libmSin(theta4);
              const df4t5 = DF4(theta5p, this.a5[qe], this.th5[qe], this.dth5ast[qe], this.b5[qe], this.dth5c[qe]) / libmSin(theta5p);
              const df4t6 = DF4(theta6p, this.a6[qe], this.th6[qe], this.dth6ast[qe], this.b6[qe], this.dth6c[qe]) / libmSin(theta6p);
              const df5c1 = DF5(-cosphi1, this.a1[qe], -this.cos1ast[qe], this.b1[qe], -this.cos1c[qe]);
              const df5c2 = DF5(-cosphi2, this.a2[qe], -this.cos2ast[qe], this.b2[qe], -this.cos2c[qe]);
              delf[0] = 0.0; delf[1] = 0.0; delf[2] = 0.0;
              delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
              deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
              // the first (stacking distance) force
              let finc = -df1 * f4t4 * f4t5 * f4t6 * f5c1 * f5c2;
              delf[0] += dStk[0] * finc; delf[1] += dStk[1] * finc; delf[2] += dStk[2] * finc;
              if (theta5p !== 0.0) {
                finc = -f1 * f4t4 * df4t5 * f4t6 * f5c1 * f5c2 * rinvStk;
                delf[0] += (dStkN[0] * cost5p - bz[0]) * finc;
                delf[1] += (dStkN[1] * cost5p - bz[1]) * finc;
                delf[2] += (dStkN[2] * cost5p - bz[2]) * finc;
              }
              if (theta6p !== 0.0) {
                finc = -f1 * f4t4 * f4t5 * df4t6 * f5c1 * f5c2 * rinvStk;
                delf[0] += (dStkN[0] * cost6p - az[0]) * finc;
                delf[1] += (dStkN[1] * cost6p - az[1]) * finc;
                delf[2] += (dStkN[2] * cost6p - az[2]) * finc;
              }
              // f[a] -= delf, torque[a] -= cross(ra_cstk, delf); f[b] += delf, torque[b] += cross(rb_cstk, delf)
              addF(a, delf, -1);
              cross(rac, delf, delta);
              addF(b, delf, 1);
              cross(rbc, delf, deltb);
              addT(a, delta, -1);
              addT(b, deltb, 1);
              // the cos phi forces (backbone sites)
              delf[0] = 0.0; delf[1] = 0.0; delf[2] = 0.0;
              delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
              deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
              if (cosphi1 !== 0.0) {
                finc = -f1 * f4t4 * f4t5 * f4t6 * df5c1 * f5c2 * rinvBk;
                delf[0] += (dBkN[0] * cosphi1 - by[0]) * finc;
                delf[1] += (dBkN[1] * cosphi1 - by[1]) * finc;
                delf[2] += (dBkN[2] * cosphi1 - by[2]) * finc;
              }
              if (cosphi2 !== 0.0) {
                finc = -f1 * f4t4 * f4t5 * f4t6 * f5c1 * df5c2 * rinvBk;
                delf[0] += (dBkN[0] * cosphi2 - ay[0]) * finc;
                delf[1] += (dBkN[1] * cosphi2 - ay[1]) * finc;
                delf[2] += (dBkN[2] * cosphi2 - ay[2]) * finc;
              }
              addF(a, delf, -1);
              cross(rab, delf, delta);
              addF(b, delf, 1);
              cross(rbb, delf, deltb);
              addT(a, delta, -1);
              addT(b, deltb, 1);
              // torques of the angular terms
              delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
              deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
              if (theta4 !== 0.0) {
                const tpair = -f1 * df4t4 * f4t5 * f4t6 * f5c1 * f5c2;
                cross(az, bz, t4dir);
                for (let c = 0; c < 3; c++) { delta[c] += t4dir[c] * tpair; deltb[c] += t4dir[c] * tpair; }
              }
              if (theta5p !== 0.0) {
                const tpair = -f1 * f4t4 * df4t5 * f4t6 * f5c1 * f5c2;
                cross(dStkN, bz, t5pdir);
                for (let c = 0; c < 3; c++) deltb[c] += t5pdir[c] * tpair;
              }
              if (theta6p !== 0.0) {
                const tpair = -f1 * f4t4 * f4t5 * df4t6 * f5c1 * f5c2;
                cross(dStkN, az, t6pdir);
                for (let c = 0; c < 3; c++) delta[c] -= t6pdir[c] * tpair;
              }
              if (cosphi1 !== 0.0) {
                const tpair = -f1 * f4t4 * f4t5 * f4t6 * df5c1 * f5c2;
                cross(dBkN, by, cphi1dir);
                for (let c = 0; c < 3; c++) deltb[c] += cphi1dir[c] * tpair;
              }
              if (cosphi2 !== 0.0) {
                const tpair = -f1 * f4t4 * f4t5 * f4t6 * f5c1 * df5c2;
                cross(dBkN, ay, cphi2dir);
                for (let c = 0; c < 3; c++) delta[c] -= cphi2dir[c] * tpair;
              }
              addT(a, delta, -1);
              addT(b, deltb, 1);
              // energy tally (ev_tally_xyz of this style: the total, and half per atom)
              pc.acc.evdwl += evdwl;
              if (eatom) { eatom[a] += 0.5 * evdwl; eatom[b] += 0.5 * evdwl; }
            }
          }
        }
      }
    }
  }
}

/** Owned index of an atom ID (the bond list stores IDs). */
const indexOf = (map: Int32Array, id: number): number => {
  const i = id < map.length ? map[id] : -1;
  if (i < 0) throw new StyleError(`bond atom ${id} missing`);
  return i;
};
