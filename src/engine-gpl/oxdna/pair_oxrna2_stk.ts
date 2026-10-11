/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxrna2_stk.cpp and
 * pair_oxrna2_stk.h (class PairOxrna2Stk, pair_style oxrna2/stk): the oxRNA2 stacking interaction, with the
 * stacking sites of the 5' and 3' nucleotide of each bond, the backbone-site angles 9 and 10 of the oxRNA2 model
 * (built from the d3p and d5p direction vectors) and the sequence dependence of the stacking strength (seqdep).
 * Original authors (from the headers): Oliver Henrich (University of Strathclyde, Glasgow) and Kierran Falloon
 * (University of Strathclyde, Glasgow); described by Sulc et al. (oxRNA2, J Chem Phys 2012). LAMMPS is Copyright
 * (2003) Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0); this file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The term runs over the FENE backbone bonds (the bond list): the 3' side of a bond is the atom whose 5' partner
 * is the other atom (pair_oxrna2_stk.cpp, compute). The backbone-site distance uses the oxDNA1 x offset of the
 * backbone site as the C++ does (dx_cbk_oxdna2 is set to get_dx_cbk_oxdna1() there).
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { OXDNA_5P, buildAtomMap, isEllipsoid } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { DF1, DF4, DF5, F1, F4, F5, lrfOf, oxdnaConstants } from './common';
import { stackingDerived, stackingStrength } from './pair_stk';

/** pair_oxrna2_stk.cpp constructor: the sequence-dependence factors eta_st[i-1][j-1] (1-based types). */
const ETA_OXRNA2: number[][] = [
  [0.93851, 0.86331, 0.99407, 0.98804],
  [1.12901, 1.05060, 1.14333, 1.04949],
  [1.15626, 0.90982, 1.06573, 1.12063],
  [0.88850, 0.83252, 0.91705, 0.83818],
];

/** The 23 numbers after "I J seqav|seqdep T" (pair_coeff) or after the potential-file name. */
const OXR_NAMES = ['xi_st', 'kappa_st', 'a_st', 'cut_st_0', 'cut_st_c', 'cut_st_lo', 'cut_st_hi',
  'a_st5', 'theta_st5_0', 'dtheta_st5_ast', 'a_st6', 'theta_st6_0', 'dtheta_st6_ast',
  'a_st9', 'theta_st9_0', 'dtheta_st9_ast', 'a_st10', 'theta_st10_0', 'dtheta_st10_ast',
  'a_st1', 'cosphi_st1_ast', 'a_st2', 'cosphi_st2_ast'];

/** pair_oxrna2_stk.cpp: the direction vectors of the 3' and 5' backbone angles (in the local frame). */
const D3P = [-0.462510, -0.528218, +0.712089];
const D5P = [-0.104402, -0.841783, +0.529624];

type Tab = Float64Array;

export class PairOxrna2Stk extends Pair {
  readonly name: string = 'oxrna2/stk';
  virialFdotr = true;
  needsHalf = false;
  private nt = 0;
  private seqdep = false;
  private t!: Record<string, Tab>;
  private set!: Uint8Array;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nt = ntypes + 1;
    const n = this.nt * this.nt;
    this.t = {};
    for (const k of ['eps', 'a', 'cut0', 'cutc', 'cutlo', 'cuthi', 'cutlc', 'cuthc', 'blo', 'bhi', 'shift',
      'a5', 'th5', 'dth5ast', 'b5', 'dth5c', 'a6', 'th6', 'dth6ast', 'b6', 'dth6c',
      'a9', 'th9', 'dth9ast', 'b9', 'dth9c', 'a10', 'th10', 'dth10ast', 'b10', 'dth10c',
      'a1', 'cos1ast', 'b1', 'cos1c', 'a2', 'cos2ast', 'b2', 'cos2c']) {
      this.t[k] = new Float64Array(n);
    }
    this.set = new Uint8Array(n);
  }

  /**
   * pair_oxrna2_stk.cpp coeff(): "pair_coeff I J seqav|seqdep T xi kappa a cut_0 cut_c cut_lo cut_hi a5 theta5_0
   * dtheta5_ast a6 theta6_0 dtheta6_ast a9 theta9_0 dtheta9_ast a10 theta10_0 dtheta10_ast a1 cosphi1_ast a2
   * cosphi2_ast" (27 words) or "pair_coeff I J seqav|seqdep T file" (5 words).
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 27 && args.length !== 5) throw new StyleError('Incorrect args for pair coefficients in oxrna2/stk');
    if (args[2] !== 'seqav' && args[2] !== 'seqdep') throw new StyleError('Incorrect setting, select seqav or seqdep in oxrna2/stk');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    this.seqdep = args[2] === 'seqdep';
    if (this.seqdep && this.ntypes > 4) throw new StyleError('pair_style oxrna2/stk seqdep is defined for atom types 1 to 4 (A, C, G, U)');
    const T = parseNum(args[3], 'T');
    let v: number[];
    if (args.length === 27) {
      v = args.slice(4).map((w, k) => parseNum(w, OXR_NAMES[k]));
    } else {
      const text = ctx?.readFile(args[4]) ?? '';
      let found: number[] | null = null;
      for (const line of text.split(/\r?\n/)) {
        const w = line.replace(/#.*$/, '').trim().split(/\s+/).filter((x) => x.length);
        if (w.length < 3) continue;
        if (w[0] === args[0] && w[1] === args[1] && w[2] === 'stk') {
          if (w.length < 26) throw new StyleError(`Problem parsing oxDNA potential file: stk needs 23 values in ${args[4]}`);
          found = w.slice(3, 26).map((x, k) => parseNum(x, OXR_NAMES[k]));
          break;
        }
      }
      if (!found) throw new StyleError(`No corresponding stk potential found in file ${args[4]} for pair type ${args[0]} ${args[1]}`);
      v = found;
    }
    const [xi, kappa, a, cut0, cutC, cutLo, cutHi, a5, th5, dth5, a6, th6, dth6, a9, th9, dth9, a10, th10, dth10, a1, cos1, a2, cos2] = v;
    const eps = stackingStrength(xi, kappa, T);
    const { bLo, cLc, bHi, cHc, shiftOne } = stackingDerived(eps, a, cut0, cutC, cutLo, cutHi);
    const bang = (aa: number, ast: number) => aa * aa * ast * ast / (1 - aa * ast * ast);
    const cang = (aa: number, ast: number) => 1 / (aa * ast);
    const nt = this.nt;
    const TT = this.t;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const q = i * nt + j;
        const eta = this.seqdep ? ETA_OXRNA2[i - 1][j - 1] : 1;
        TT.eps[q] = eps * eta;
        TT.a[q] = a; TT.cut0[q] = cut0; TT.cutc[q] = cutC; TT.cutlo[q] = cutLo; TT.cuthi[q] = cutHi;
        TT.cutlc[q] = cLc; TT.cuthc[q] = cHc; TT.blo[q] = bLo; TT.bhi[q] = bHi;
        TT.shift[q] = shiftOne * eta;
        TT.a5[q] = a5; TT.th5[q] = th5; TT.dth5ast[q] = dth5; TT.b5[q] = bang(a5, dth5); TT.dth5c[q] = cang(a5, dth5);
        TT.a6[q] = a6; TT.th6[q] = th6; TT.dth6ast[q] = dth6; TT.b6[q] = bang(a6, dth6); TT.dth6c[q] = cang(a6, dth6);
        TT.a9[q] = a9; TT.th9[q] = th9; TT.dth9ast[q] = dth9; TT.b9[q] = bang(a9, dth9); TT.dth9c[q] = cang(a9, dth9);
        TT.a10[q] = a10; TT.th10[q] = th10; TT.dth10ast[q] = dth10; TT.b10[q] = bang(a10, dth10); TT.dth10c[q] = cang(a10, dth10);
        TT.a1[q] = a1; TT.cos1ast[q] = cos1; TT.b1[q] = bang(a1, cos1); TT.cos1c[q] = cang(a1, cos1);
        TT.a2[q] = a2; TT.cos2ast[q] = cos2; TT.b2[q] = bang(a2, cos2); TT.cos2c[q] = cang(a2, cos2);
        this.set[q] = 1;
      }
    }
  }

  /** pair_oxrna2_stk.cpp init_one(): the (j, i) copies (sequence factors inverted), returns cut_st_hc[i][j]. */
  initOne(i: number, j: number): number {
    const nt = this.nt;
    if (!this.set[i * nt + j]) throw new StyleError('Coefficient mixing not defined in oxRNA');
    const q = i * nt + j, r = j * nt + i;
    const TT = this.t;
    for (const key of Object.keys(TT)) {
      if (key === 'eps' || key === 'shift') continue;
      TT[key][r] = TT[key][q];
    }
    if (this.seqdep) {
      TT.eps[r] = TT.eps[q] / ETA_OXRNA2[i - 1][j - 1] * ETA_OXRNA2[j - 1][i - 1];
      TT.shift[r] = TT.shift[q] / ETA_OXRNA2[i - 1][j - 1] * ETA_OXRNA2[j - 1][i - 1];
    } else {
      TT.eps[r] = TT.eps[q];
      TT.shift[r] = TT.shift[q];
    }
    return TT.cuthc[q];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const id5p = s.custom.get(OXDNA_5P);
    if (!id5p || !s.quat || !s.torque) throw new StyleError('pair_style oxrna2/stk requires atom_style hybrid bond ellipsoid oxdna');
    const bonds = s.topo.bonds;
    const { x, f, type } = pc;
    const nt = this.nt;
    const TT = this.t;
    const constants = oxdnaConstants(s.units.style);
    const dxCbk = constants.dx_cbk_oxdna1, dzCbk = constants.dz_cbk_oxrna2;
    const dx3p = constants.dx_cstk_3p_oxrna2, dy3p = constants.dy_cstk_3p_oxrna2;
    const dx5p = constants.dx_cstk_5p_oxrna2, dy5p = constants.dy_cstk_5p_oxrna2;
    const map = buildAtomMap(s);
    const nlocal = pc.nlocal;
    const L = new Float64Array(9 * nlocal);
    for (let i = 0; i < nlocal; i++) if (isEllipsoid(s, i)) lrfOf(s.quat, 4 * i, L, 9 * i);
    const torque = s.torque;
    let evdwl = 0;
    const id5 = id5p.data;
    const ax = [0, 0, 0], ay = [0, 0, 0], az = [0, 0, 0], bx = [0, 0, 0], by = [0, 0, 0], bz = [0, 0, 0];
    const ra5 = [0, 0, 0], rb3 = [0, 0, 0], rac = [0, 0, 0], rbc = [0, 0, 0];
    const d = [0, 0, 0], xb = [0, 0, 0], drs = [0, 0, 0], drsn = [0, 0, 0], drb = [0, 0, 0], drbn = [0, 0, 0];
    const aux3 = [0, 0, 0], aux5 = [0, 0, 0];
    const delf = [0, 0, 0], delta = [0, 0, 0], deltb = [0, 0, 0], dir = [0, 0, 0];
    const cross = (u: readonly number[], v: readonly number[], out: number[]) => {
      out[0] = u[1] * v[2] - u[2] * v[1]; out[1] = u[2] * v[0] - u[0] * v[2]; out[2] = u[0] * v[1] - u[1] * v[0];
    };
    const clamp = (c: number) => (c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c);
    const dot = (u: readonly number[], v: readonly number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    for (let k = 0; k < bonds.n; k++) {
      let a = map[bonds.atoms[2 * k]];
      let b = map[bonds.atoms[2 * k + 1]];
      if (s.id[b] !== id5[a]) { const t = a; a = b; b = t; }
      for (let c = 0; c < 3; c++) {
        ax[c] = L[9 * a + c]; ay[c] = L[9 * a + 3 + c]; az[c] = L[9 * a + 6 + c];
        bx[c] = L[9 * b + c]; by[c] = L[9 * b + 3 + c]; bz[c] = L[9 * b + 6 + c];
      }
      // closest image of b relative to a
      d[0] = x[3 * b] - x[3 * a]; d[1] = x[3 * b + 1] - x[3 * a + 1]; d[2] = x[3 * b + 2] - x[3 * a + 2];
      pc.geom.minimumImage(d);
      xb[0] = x[3 * a] + d[0]; xb[1] = x[3 * a + 1] + d[1]; xb[2] = x[3 * a + 2] + d[2];
      // stacking sites: 5' site of a (ax, ay), 3' site of b (bx, by)
      for (let c = 0; c < 3; c++) {
        ra5[c] = dx5p * ax[c] + dy5p * ay[c];
        rb3[c] = dx3p * bx[c] + dy3p * by[c];
      }
      for (let c = 0; c < 3; c++) drs[c] = xb[c] + rb3[c] - x[3 * a + c] - ra5[c];
      const atype = type[a], btype = type[b];
      const rsqS = drs[0] * drs[0] + drs[1] * drs[1] + drs[2] * drs[2];
      const rS = Math.sqrt(rsqS);
      const rinvS = 1.0 / rS;
      for (let c = 0; c < 3; c++) drsn[c] = drs[c] * rinvS;
      // backbone sites (oxRNA2 model) and their distance
      for (let c = 0; c < 3; c++) {
        rac[c] = dxCbk * ax[c] + dzCbk * az[c];
        rbc[c] = dxCbk * bx[c] + dzCbk * bz[c];
      }
      for (let c = 0; c < 3; c++) drb[c] = xb[c] + rbc[c] - x[3 * a + c] - rac[c];
      const rsqB = drb[0] * drb[0] + drb[1] * drb[1] + drb[2] * drb[2];
      const rB = Math.sqrt(rsqB);
      const rinvB = 1.0 / rB;
      for (let c = 0; c < 3; c++) drbn[c] = drb[c] * rinvB;
      const ij = atype * nt + btype;
      const f1 = F1(rS, TT.eps[ij], TT.a[ij], TT.cut0[ij], TT.cutlc[ij], TT.cuthc[ij], TT.cutlo[ij], TT.cuthi[ij], TT.blo[ij], TT.bhi[ij], TT.shift[ij]);
      if (f1 === 0.0) continue;
      const cost5p = clamp(dot(drsn, bz));
      const theta5p = Math.acos(cost5p);
      const f4t5 = F4(theta5p, TT.a5[ij], TT.th5[ij], TT.dth5ast[ij], TT.b5[ij], TT.dth5c[ij]);
      if (f4t5 === 0.0) continue;
      const cost6p = clamp(dot(drsn, az));
      const theta6p = Math.acos(cost6p);
      for (let c = 0; c < 3; c++) {
        aux5[c] = D5P[0] * ax[c] + D5P[1] * ay[c] + D5P[2] * az[c];
        aux3[c] = D3P[0] * bx[c] + D3P[1] * by[c] + D3P[2] * bz[c];
      }
      const cost9 = clamp(dot(drbn, aux3));
      const theta9 = Math.acos(cost9);
      const cost10 = clamp(dot(drbn, aux5));
      const theta10 = Math.acos(cost10);
      const cosphi1 = clamp(dot(drbn, by));
      const cosphi2 = clamp(dot(drbn, ay));
      const f4t6 = F4(theta6p, TT.a6[ij], TT.th6[ij], TT.dth6ast[ij], TT.b6[ij], TT.dth6c[ij]);
      const f4t9 = F4(theta9, TT.a9[ij], TT.th9[ij], TT.dth9ast[ij], TT.b9[ij], TT.dth9c[ij]);
      const f4t10 = F4(theta10, TT.a10[ij], TT.th10[ij], TT.dth10ast[ij], TT.b10[ij], TT.dth10c[ij]);
      const f5c1 = F5(-cosphi1, TT.a1[ij], -TT.cos1ast[ij], TT.b1[ij], -TT.cos1c[ij]);
      const f5c2 = F5(-cosphi2, TT.a2[ij], -TT.cos2ast[ij], TT.b2[ij], -TT.cos2c[ij]);
      const e = f1 * f4t5 * f4t6 * f4t9 * f4t10 * f5c1 * f5c2;
      if (e === 0.0) continue;
      const df1 = DF1(rS, TT.eps[ij], TT.a[ij], TT.cut0[ij], TT.cutlc[ij], TT.cuthc[ij], TT.cutlo[ij], TT.cuthi[ij], TT.blo[ij], TT.bhi[ij]);
      const df4t5 = DF4(theta5p, TT.a5[ij], TT.th5[ij], TT.dth5ast[ij], TT.b5[ij], TT.dth5c[ij]) / Math.sin(theta5p);
      const df4t6 = DF4(theta6p, TT.a6[ij], TT.th6[ij], TT.dth6ast[ij], TT.b6[ij], TT.dth6c[ij]) / Math.sin(theta6p);
      const df4t9 = DF4(theta9, TT.a9[ij], TT.th9[ij], TT.dth9ast[ij], TT.b9[ij], TT.dth9c[ij]) / Math.sin(theta9);
      const df4t10 = DF4(theta10, TT.a10[ij], TT.th10[ij], TT.dth10ast[ij], TT.b10[ij], TT.dth10c[ij]) / Math.sin(theta10);
      const df5c1 = DF5(-cosphi1, TT.a1[ij], -TT.cos1ast[ij], TT.b1[ij], -TT.cos1c[ij]);
      const df5c2 = DF5(-cosphi2, TT.a2[ij], -TT.cos2ast[ij], TT.b2[ij], -TT.cos2c[ij]);
      // radial force along the stacking vector, then the angles 5' and 6' (the stacking sites)
      delf[0] = 0.0; delf[1] = 0.0; delf[2] = 0.0;
      delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
      deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
      let finc = -df1 * f4t5 * f4t6 * f4t9 * f4t10 * f5c1 * f5c2;
      for (let c = 0; c < 3; c++) delf[c] += drs[c] * finc;
      if (theta5p !== 0.0) {
        finc = -f1 * df4t5 * f4t6 * f4t9 * f4t10 * f5c1 * f5c2 * rinvS;
        for (let c = 0; c < 3; c++) delf[c] += (drsn[c] * cost5p - bz[c]) * finc;
      }
      if (theta6p !== 0.0) {
        finc = -f1 * f4t5 * df4t6 * f4t9 * f4t10 * f5c1 * f5c2 * rinvS;
        for (let c = 0; c < 3; c++) delf[c] += (drsn[c] * cost6p - az[c]) * finc;
      }
      // f[a] -= delf, torque[a] -= cross(ra5, delf); f[b] += delf, torque[b] += cross(rb3, delf)
      for (let c = 0; c < 3; c++) f[3 * a + c] -= delf[c];
      cross(ra5, delf, delta);
      for (let c = 0; c < 3; c++) torque[3 * a + c] -= delta[c];
      for (let c = 0; c < 3; c++) f[3 * b + c] += delf[c];
      cross(rb3, delf, deltb);
      for (let c = 0; c < 3; c++) torque[3 * b + c] += deltb[c];
      evdwl += e;
      // forces between the backbone sites (angles 9 and 10, cosphi1 and cosphi2)
      delf[0] = 0.0; delf[1] = 0.0; delf[2] = 0.0;
      delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
      deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
      if (theta9 !== 0.0) {
        finc = -f1 * f4t5 * f4t6 * df4t9 * f4t10 * f5c1 * f5c2 * rinvB;
        for (let c = 0; c < 3; c++) delf[c] += (drbn[c] * cost9 - aux3[c]) * finc;
      }
      if (theta10 !== 0.0) {
        finc = -f1 * f4t5 * f4t6 * f4t9 * df4t10 * f5c1 * f5c2 * rinvB;
        for (let c = 0; c < 3; c++) delf[c] += (drbn[c] * cost10 - aux5[c]) * finc;
      }
      if (cosphi1 !== 0.0) {
        finc = -f1 * f4t5 * f4t6 * f4t9 * f4t10 * df5c1 * f5c2 * rinvB;
        for (let c = 0; c < 3; c++) delf[c] += (drbn[c] * cosphi1 - by[c]) * finc;
      }
      if (cosphi2 !== 0.0) {
        finc = -f1 * f4t5 * f4t6 * f4t9 * f4t10 * f5c1 * df5c2 * rinvB;
        for (let c = 0; c < 3; c++) delf[c] += (drbn[c] * cosphi2 - ay[c]) * finc;
      }
      for (let c = 0; c < 3; c++) f[3 * a + c] -= delf[c];
      cross(rac, delf, delta);
      for (let c = 0; c < 3; c++) f[3 * b + c] += delf[c];
      cross(rbc, delf, deltb);
      for (let c = 0; c < 3; c++) { torque[3 * a + c] -= delta[c]; torque[3 * b + c] += deltb[c]; }
      // pure torques of the same angles
      delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
      deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
      if (theta5p !== 0.0) {
        const tpair = -f1 * df4t5 * f4t6 * f4t9 * f4t10 * f5c1 * f5c2;
        cross(drsn, bz, dir);
        for (let c = 0; c < 3; c++) deltb[c] += dir[c] * tpair;
      }
      if (theta6p !== 0.0) {
        const tpair = -f1 * f4t5 * df4t6 * f4t9 * f4t10 * f5c1 * f5c2;
        cross(drsn, az, dir);
        for (let c = 0; c < 3; c++) delta[c] -= dir[c] * tpair;
      }
      if (theta9 !== 0.0) {
        const tpair = -f1 * f4t5 * f4t6 * df4t9 * f4t10 * f5c1 * f5c2;
        cross(drbn, aux3, dir);
        for (let c = 0; c < 3; c++) deltb[c] += dir[c] * tpair;
      }
      if (theta10 !== 0.0) {
        const tpair = -f1 * f4t5 * f4t6 * f4t9 * df4t10 * f5c1 * f5c2;
        cross(drbn, aux5, dir);
        for (let c = 0; c < 3; c++) delta[c] -= dir[c] * tpair;
      }
      if (cosphi1 !== 0.0) {
        const tpair = -f1 * f4t5 * f4t6 * f4t9 * f4t10 * df5c1 * f5c2;
        cross(drbn, by, dir);
        for (let c = 0; c < 3; c++) deltb[c] += dir[c] * tpair;
      }
      if (cosphi2 !== 0.0) {
        const tpair = -f1 * f4t5 * f4t6 * f4t9 * f4t10 * f5c1 * df5c2;
        cross(drbn, ay, dir);
        for (let c = 0; c < 3; c++) delta[c] -= dir[c] * tpair;
      }
      for (let c = 0; c < 3; c++) { torque[3 * a + c] -= delta[c]; torque[3 * b + c] += deltb[c]; }
    }
    pc.acc.evdwl += evdwl;
  }
}

