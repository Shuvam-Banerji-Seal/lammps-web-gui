/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna2_coaxstk.cpp and
 * pair_oxdna2_coaxstk.h (class PairOxdna2Coaxstk, pair_style oxdna2/coaxstk): the oxDNA2 coaxial stacking, with
 * the F6 angular term of the first angle, the direction-dependent stacking strength k (3'-5' or 5'-3' order of
 * the two nucleotides), and the stacking sites of the oxDNA1 model. Original authors (from the headers): Oliver
 * Henrich (University of Strathclyde, Glasgow) and Kierran Falloon (University of Strathclyde, Glasgow);
 * described by Snodin et al. (J Chem Phys 142, 234901, 2015). LAMMPS is Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The C++ computes a cosine (cosphi3) from the backbone sites and never uses it in the energy or the forces; it
 * is left out here. The force of the first and fourth angles is not in the C++ force part (only in the torques);
 * this port keeps that.
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { OXDNA_3P, OXDNA_5P, isEllipsoid } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { DF2, DF4, DF6, F2, F4, F6, lrfOf, oxdnaConstants, stackingSiteOxdna1, type StackRule, potentialLines } from './common';
import { sinFma as libmSin } from '../libm/sincos_fma';
import { acos as libmAcos } from '../libm/acos';

const CX2_NAMES = ['k_cxst', 'cut_cxst_0', 'cut_cxst_c', 'cut_cxst_lo', 'cut_cxst_hi',
  'a_cxst1', 'theta_cxst1_0', 'dtheta_cxst1_ast', 'a_cxst4', 'theta_cxst4_0', 'dtheta_cxst4_ast',
  'a_cxst5', 'theta_cxst5_0', 'dtheta_cxst5_ast', 'a_cxst6', 'theta_cxst6_0', 'dtheta_cxst6_ast',
  'AA_cxst1', 'BB_cxst1'];

const PI = Math.PI;
type Tab = Float64Array;

export class PairOxdna2Coaxstk extends Pair {
  readonly name: string = 'oxdna2/coaxstk';
  virialFdotr = true;
  /** stacking site rule of the nucleotide model (the oxDNA3 subclass has its own offset) */
  protected stackSite: StackRule = stackingSiteOxdna1;
  /** eta_cxst[i][j] of the C++ (1 for oxDNA2; the oxDNA3 subclass sets its sequence table): k_cxst[i][j] = k eta[i%4][j%4] */
  protected etaCxst: number[][] = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]];
  private nt = 0;
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
    for (const k of ['k', 'cut0', 'cutc', 'cutlo', 'cuthi', 'cutlc', 'cuthc', 'blo', 'bhi',
      'a1', 'th1', 'dth1ast', 'b1', 'dth1c', 'a4', 'th4', 'dth4ast', 'b4', 'dth4c', 'a5', 'th5', 'dth5ast', 'b5', 'dth5c',
      'a6', 'th6', 'dth6ast', 'b6', 'dth6c', 'AA', 'BB']) {
      this.t[k] = new Float64Array(n);
    }
    this.set = new Uint8Array(n);
  }

  /**
   * pair_oxdna2_coaxstk.cpp coeff(): "pair_coeff I J <19 numbers>" (narg 21) or "pair_coeff I J file" (narg 3).
   * b_cxst_lo = 0.25 (lo - 0)^2 / (0.5 (lo - 0)^2 - 0.5 (0 - c)^2), cut_lc = lo - 0.5 (lo - 0) / b_lo.
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 21 && args.length !== 3) throw new StyleError('Incorrect args for pair coefficients in oxdna2/coaxstk');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    let v: number[];
    if (args.length === 21) {
      v = args.slice(2).map((w, k) => parseNum(w, CX2_NAMES[k]));
    } else {
      const text = ctx?.readFile(args[2]) ?? '';
      let found: number[] | null = null;
      for (const w of potentialLines(text, [args[0], args[1]], 'coaxstk')) {
        if (w.length < 3) continue;
        if (w[2] === 'coaxstk') {
          if (w.length < 22) throw new StyleError(`Problem parsing oxdna potential file: coaxstk needs 19 values in ${args[2]}`);
          found = w.slice(3, 22).map((x, k) => parseNum(x, CX2_NAMES[k]));
          break;
        }
      }
      if (!found) throw new StyleError(`No corresponding coaxstk potential found in file ${args[2]} for pair type ${args[0]} ${args[1]}`);
      v = found;
    }
    const [k, cut0, cutC, cutLo, cutHi, a1, th1, dth1, a4, th4, dth4, a5, th5, dth5, a6, th6, dth6, AA, BB] = v;
    const dlo = cutLo - cut0, dhi = cutHi - cut0;
    const dcc = 0.5 * (cut0 - cutC) * (cut0 - cutC);
    const bLo = 0.25 * dlo * dlo / (0.5 * dlo * dlo - dcc);
    const cLc = cutLo - 0.5 * dlo / bLo;
    const bHi = 0.25 * dhi * dhi / (0.5 * dhi * dhi - dcc);
    const cHc = cutHi - 0.5 * dhi / bHi;
    const bang = (aa: number, ast: number) => aa * aa * ast * ast / (1 - aa * ast * ast);
    const cang = (aa: number, ast: number) => 1 / (aa * ast);
    const b1 = bang(a1, dth1), c1 = cang(a1, dth1);
    const b4 = bang(a4, dth4), c4 = cang(a4, dth4);
    const b5 = bang(a5, dth5), c5 = cang(a5, dth5);
    const b6 = bang(a6, dth6), c6 = cang(a6, dth6);
    const T = this.t;
    const nt = this.nt;
    // k_cxst[i][j] = k eta_cxst[imod4-1][jmod4-1] for every ordered pair of the range (pair_oxdna2_coaxstk.cpp coeff)
    const mod4 = (t: number) => (t % 4 === 0 ? 4 : t % 4);
    for (let i = ilo; i <= ihi; i++) {
      for (let j = jlo; j <= jhi; j++) T.k[i * nt + j] = k * this.etaCxst[mod4(i) - 1][mod4(j) - 1];
    }
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const q = i * nt + j;
        const set = (key: string, val: number) => { T[key][q] = val; };
        set('cut0', cut0); set('cutc', cutC); set('cutlo', cutLo); set('cuthi', cutHi);
        set('cutlc', cLc); set('cuthc', cHc); set('blo', bLo); set('bhi', bHi);
        set('a1', a1); set('th1', th1); set('dth1ast', dth1); set('b1', b1); set('dth1c', c1);
        set('a4', a4); set('th4', th4); set('dth4ast', dth4); set('b4', b4); set('dth4c', c4);
        set('a5', a5); set('th5', th5); set('dth5ast', dth5); set('b5', b5); set('dth5c', c5);
        set('a6', a6); set('th6', th6); set('dth6ast', dth6); set('b6', b6); set('dth6c', c6);
        set('AA', AA); set('BB', BB);
        this.set[q] = 1;
      }
    }
  }

  /** pair_oxdna2_coaxstk.cpp init_one(): symmetric (j, i) copies; returns cut_cxst_hc[i][j]. */
  initOne(i: number, j: number): number {
    const nt = this.nt;
    if (!this.set[i * nt + j]) throw new StyleError('Coefficient mixing not defined in oxDNA');
    const q = i * nt + j, r = j * nt + i;
    // k_cxst is not symmetric in the C++ (eta_cxst is not): it is set for every ordered pair in coeff()
    for (const key of Object.keys(this.t)) if (key !== 'k') this.t[key][r] = this.t[key][q];
    return this.t.cuthc[q];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const id3p = s.custom.get(OXDNA_3P), id5p = s.custom.get(OXDNA_5P);
    if (!id3p || !id5p || !s.quat || !s.torque) throw new StyleError('pair_style oxdna2/coaxstk requires atom_style hybrid bond ellipsoid oxdna');
    const list = pc.half!;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nall = pc.nall;
    const nt = this.nt;
    const T = this.t;
    const constants = oxdnaConstants(s.units.style);
    const special = pc.specialLJ;
    const tq = new Float64Array(3 * nall);
    const L = new Float64Array(9 * nall);
    for (let k = 0; k < nall; k++) {
      const o = owner[k];
      if (isEllipsoid(s, o)) lrfOf(s.quat, 4 * o, L, 9 * k);
    }
    const rst = new Float64Array(3 * nall);
    const tmp = [0, 0, 0];
    for (let k = 0; k < nall; k++) {
      this.stackSite(constants, L.subarray(9 * k, 9 * k + 3), L.subarray(9 * k + 3, 9 * k + 6), L.subarray(9 * k + 6, 9 * k + 9), tmp);
      rst[3 * k] = tmp[0]; rst[3 * k + 1] = tmp[1]; rst[3 * k + 2] = tmp[2];
    }
    const interior = (o: number) => id3p.data[o] !== 0 && id5p.data[o] !== 0;
    let evdwl = 0;
    const delr = [0, 0, 0], drn = [0, 0, 0], delf = [0, 0, 0], delta = [0, 0, 0], deltb = [0, 0, 0], dir = [0, 0, 0];
    const ax = [0, 0, 0], az = [0, 0, 0], bx = [0, 0, 0], bz = [0, 0, 0];
    const cross = (u: readonly number[], v: readonly number[], out: number[]) => {
      out[0] = u[1] * v[2] - u[2] * v[1]; out[1] = u[2] * v[0] - u[0] * v[2]; out[2] = u[0] * v[1] - u[1] * v[0];
    };
    const clamp = (c: number) => (c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c);
    const dot = (u: readonly number[], v: readonly number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    for (let a = 0; a < list.inum; a++) {
      if (interior(a)) continue;
      const atype = type[a];
      for (let c = 0; c < 3; c++) { ax[c] = L[9 * a + c]; az[c] = L[9 * a + 6 + c]; }
      const jstart = list.firstneigh[a], jend = jstart + list.numneigh[a];
      for (let kk = jstart; kk < jend; kk++) {
        const jj = list.neighbors[kk];
        const factorLJ = special[jj >>> SBBITS];
        const b = jj & NEIGHMASK;
        const ob = owner[b];
        if (interior(ob)) continue;
        const btype = type[b];
        for (let c = 0; c < 3; c++) { bx[c] = L[9 * b + c]; bz[c] = L[9 * b + 6 + c]; }
        for (let c = 0; c < 3; c++) delr[c] = x[3 * a + c] + rst[3 * a + c] - x[3 * b + c] - rst[3 * b + c];
        const rsq = delr[0] * delr[0] + delr[1] * delr[1] + delr[2] * delr[2];
        const r = Math.sqrt(rsq);
        const rinv = 1.0 / r;
        for (let c = 0; c < 3; c++) drn[c] = delr[c] * rinv;
        const ij = atype * nt + btype;
        const cost1 = clamp(-dot(ax, bx));
        const theta1 = libmAcos(cost1);
        const theta1p = 2 * PI - theta1;
        const f4f6t1 = F4(theta1, T.a1[ij], T.th1[ij], T.dth1ast[ij], T.b1[ij], T.dth1c[ij]) + F6(theta1, T.AA[ij], T.BB[ij]);
        if (f4f6t1 === 0.0) continue;
        const cost4 = clamp(dot(az, bz));
        const theta4 = libmAcos(cost4);
        const f4t4 = F4(theta4, T.a4[ij], T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij])
          + F4(theta4, T.a4[ij], PI - T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij]);
        if (f4t4 === 0.0) continue;
        const cost5 = clamp(dot(drn, az));
        const theta5 = libmAcos(cost5);
        const theta5p = PI - theta5;
        const f4t5 = F4(theta5, T.a5[ij], T.th5[ij], T.dth5ast[ij], T.b5[ij], T.dth5c[ij])
          + F4(theta5p, T.a5[ij], T.th5[ij], T.dth5ast[ij], T.b5[ij], T.dth5c[ij]);
        if (f4t5 === 0.0) continue;
        const cost6 = clamp(dot(drn, bz));
        const theta6 = libmAcos(cost6);
        const theta6p = PI - theta6;
        const f4t6 = F4(theta6, T.a6[ij], T.th6[ij], T.dth6ast[ij], T.b6[ij], T.dth6c[ij])
          + F4(theta6p, T.a6[ij], T.th6[ij], T.dth6ast[ij], T.b6[ij], T.dth6c[ij]);
        // stacking strength by the direction of a -> b: 5'->3' keeps the types, 3'->5' swaps them
        const idA5 = id5p.data[a], idA3 = id3p.data[a], idB5 = id5p.data[ob], idB3 = id3p.data[ob];
        let kab: number;
        if (idA5 === 0 && idB3 === 0) kab = T.k[ij];
        else if (idA3 === 0 && idB5 === 0) kab = T.k[btype * nt + atype];
        else kab = 0.5 * (T.k[ij] + T.k[btype * nt + atype]);
        const f2 = F2(r, kab, T.cut0[ij], T.cutlc[ij], T.cuthc[ij], T.cutlo[ij], T.cuthi[ij], T.blo[ij], T.bhi[ij], T.cutc[ij]);
        const e = f2 * f4f6t1 * f4t4 * f4t5 * f4t6 * factorLJ;
        if (e === 0.0) continue;
        const df2 = DF2(r, kab, T.cut0[ij], T.cutlc[ij], T.cuthc[ij], T.cutlo[ij], T.cuthi[ij], T.blo[ij], T.bhi[ij]);
        const df4f6t1 = (DF4(theta1, T.a1[ij], T.th1[ij], T.dth1ast[ij], T.b1[ij], T.dth1c[ij]) + DF6(theta1, T.AA[ij], T.BB[ij])) / libmSin(theta1);
        const df4t4 = (DF4(theta4, T.a4[ij], T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij])
          + DF4(theta4, T.a4[ij], PI - T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij])) / libmSin(theta4);
        const df4t5 = (DF4(theta5, T.a5[ij], T.th5[ij], T.dth5ast[ij], T.b5[ij], T.dth5c[ij])
          - DF4(theta5p, T.a5[ij], T.th5[ij], T.dth5ast[ij], T.b5[ij], T.dth5c[ij])) / libmSin(theta5);
        const df4t6 = (DF4(theta6, T.a6[ij], T.th6[ij], T.dth6ast[ij], T.b6[ij], T.dth6c[ij])
          - DF4(theta6p, T.a6[ij], T.th6[ij], T.dth6ast[ij], T.b6[ij], T.dth6c[ij])) / libmSin(theta6);
        delf[0] = 0.0; delf[1] = 0.0; delf[2] = 0.0;
        delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
        deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
        let finc = -df2 * f4f6t1 * f4t4 * f4t5 * f4t6 * rinv * factorLJ;
        for (let c = 0; c < 3; c++) delf[c] += delr[c] * finc;
        if (theta5 !== 0.0 && theta5p !== 0.0) {
          finc = -f2 * f4f6t1 * f4t4 * df4t5 * f4t6 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost5 - az[c]) * finc;
        }
        if (theta6 !== 0.0 && theta6p !== 0.0) {
          finc = -f2 * f4f6t1 * f4t4 * f4t5 * df4t6 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost6 - bz[c]) * finc;
        }
        // f[a] += delf, torque[a] += cross(ra_cstk, delf); f[b] -= delf, torque[b] -= cross(rb_cstk, delf)
        for (let c = 0; c < 3; c++) f[3 * a + c] += delf[c];
        cross([rst[3 * a], rst[3 * a + 1], rst[3 * a + 2]], delf, delta);
        for (let c = 0; c < 3; c++) tq[3 * a + c] += delta[c];
        for (let c = 0; c < 3; c++) f[3 * b + c] -= delf[c];
        cross([rst[3 * b], rst[3 * b + 1], rst[3 * b + 2]], delf, deltb);
        for (let c = 0; c < 3; c++) tq[3 * b + c] -= deltb[c];
        evdwl += e;
        delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
        deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
        if (theta1 !== 0.0 && theta1p !== 0.0) {
          const tpair = -f2 * df4f6t1 * f4t4 * f4t5 * f4t6 * factorLJ;
          cross(ax, bx, dir);
          for (let c = 0; c < 3; c++) { delta[c] += dir[c] * tpair; deltb[c] += dir[c] * tpair; }
        }
        if (theta4 !== 0.0) {
          const tpair = -f2 * f4f6t1 * df4t4 * f4t5 * f4t6 * factorLJ;
          cross(bz, az, dir);
          for (let c = 0; c < 3; c++) { delta[c] += dir[c] * tpair; deltb[c] += dir[c] * tpair; }
        }
        if (theta5 !== 0.0 && theta5p !== 0.0) {
          const tpair = -f2 * f4f6t1 * f4t4 * df4t5 * f4t6 * factorLJ;
          cross(drn, az, dir);
          for (let c = 0; c < 3; c++) delta[c] += dir[c] * tpair;
        }
        if (theta6 !== 0.0 && theta6p !== 0.0) {
          const tpair = -f2 * f4f6t1 * f4t4 * f4t5 * df4t6 * factorLJ;
          cross(drn, bz, dir);
          for (let c = 0; c < 3; c++) deltb[c] -= dir[c] * tpair;
        }
        for (let c = 0; c < 3; c++) { tq[3 * a + c] += delta[c]; tq[3 * b + c] -= deltb[c]; }
      }
    }
    pc.nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}
