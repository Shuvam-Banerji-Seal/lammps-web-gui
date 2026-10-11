/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna_xstk.cpp and
 * pair_oxdna_xstk.h (class PairOxdnaXstk, pair_style oxdna/xstk): the oxDNA cross-stacking interaction between
 * the base sites of nucleotides on opposite strands, with the angular factors of the base-pair geometry, and
 * the symmetric angle sums f4(theta) + f4(pi - theta) of the parallel and antiparallel orientations.
 * Original authors (from the headers): Oliver Henrich (University of Strathclyde, Glasgow) and Kierran Falloon
 * (University of Strathclyde, Glasgow); described by Ouldridge, Louis and Doye (J Chem Phys 2011). LAMMPS is
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0);
 * this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { OXDNA_5P, isEllipsoid } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { DF2, DF4, F2, F4, baseSite, lrfOf, oxdnaConstants } from './common';

/** The 23 numbers after "I J" (pair_coeff) or after the potential-file name, in order. */
const XST_NAMES = ['k_xst', 'cut_xst_0', 'cut_xst_c', 'cut_xst_lo', 'cut_xst_hi',
  'a_xst1', 'theta_xst1_0', 'dtheta_xst1_ast', 'a_xst2', 'theta_xst2_0', 'dtheta_xst2_ast',
  'a_xst3', 'theta_xst3_0', 'dtheta_xst3_ast', 'a_xst4', 'theta_xst4_0', 'dtheta_xst4_ast',
  'a_xst7', 'theta_xst7_0', 'dtheta_xst7_ast', 'a_xst8', 'theta_xst8_0', 'dtheta_xst8_ast'];

const PI = Math.PI;
type Tab = Float64Array;

export class PairOxdnaXstk extends Pair {
  readonly name: string = 'oxdna/xstk';
  /** the oxDNA model has the theta4 factor (a4, theta4_0, dtheta4); the oxRNA2 subclass drops it */
  protected withTheta4 = true;
  virialFdotr = true;
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
    for (const k of ['k', 'cut0', 'cutc', 'cutlo', 'cuthi', 'cutlc', 'cuthc', 'blo', 'bhi', 'cutsqhc',
      'a1', 'th1', 'dth1ast', 'b1', 'dth1c', 'a2', 'th2', 'dth2ast', 'b2', 'dth2c', 'a3', 'th3', 'dth3ast', 'b3', 'dth3c',
      'a4', 'th4', 'dth4ast', 'b4', 'dth4c', 'a7', 'th7', 'dth7ast', 'b7', 'dth7c', 'a8', 'th8', 'dth8ast', 'b8', 'dth8c']) {
      this.t[k] = new Float64Array(n);
    }
    this.set = new Uint8Array(n);
  }

  /**
   * pair_oxdna_xstk.cpp coeff(): "pair_coeff I J <23 numbers>" (narg 25) or "pair_coeff I J file" (narg 3).
   * b_xst_lo = 0.25 (lo - 0)^2 / (0.5 (lo - 0)^2 - k 0.5 (0 - c)^2 / k), cut_lc = lo - 0.5 (lo - 0) / b_lo.
   */
  coeff(args: string[], ctx?: StyleContext): void {
    const nNum = this.withTheta4 ? 23 : 20;
    if (args.length !== nNum + 2 && args.length !== 3) throw new StyleError('Incorrect args for pair coefficients in oxdna/xstk');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    let v: number[];
    if (args.length === nNum + 2) {
      v = args.slice(2).map((w, k) => parseNum(w, XST_NAMES[k]));
      // oxRNA2 has no a4 group: its 20 numbers are placed in the 23-number layout with zeros for it
      if (!this.withTheta4) v = [...v.slice(0, 14), 0, 0, 0, ...v.slice(14)];
    } else {
      const text = ctx?.readFile(args[2]) ?? '';
      let found: number[] | null = null;
      for (const line of text.split(/\r?\n/)) {
        const w = line.replace(/#.*$/, '').trim().split(/\s+/).filter((x) => x.length);
        if (w.length < 3) continue;
        if (w[0] === args[0] && w[1] === args[1] && w[2] === 'xstk') {
          if (w.length < 26) throw new StyleError(`Problem parsing oxdna potential file: xstk needs 23 values in ${args[2]}`);
          found = w.slice(3, 26).map((x, k) => parseNum(x, XST_NAMES[k]));
          break;
        }
      }
      if (!found) throw new StyleError(`No corresponding xstk potential found in file ${args[2]} for pair type ${args[0]} ${args[1]}`);
      v = found;
    }
    const [k, cut0, cutC, cutLo, cutHi, a1, th1, dth1, a2, th2, dth2, a3, th3, dth3, a4, th4, dth4, a7, th7, dth7, a8, th8, dth8] = v;
    const dlo = cutLo - cut0, dhi = cutHi - cut0;
    const dcc = k * 0.5 * (cut0 - cutC) * (cut0 - cutC) / k;
    const bLo = 0.25 * dlo * dlo / (0.5 * dlo * dlo - dcc);
    const cLc = cutLo - 0.5 * dlo / bLo;
    const bHi = 0.25 * dhi * dhi / (0.5 * dhi * dhi - dcc);
    const cHc = cutHi - 0.5 * dhi / bHi;
    const bang = (aa: number, ast: number) => aa * aa * ast * ast / (1 - aa * ast * ast);
    const cang = (aa: number, ast: number) => 1 / (aa * ast);
    const T = this.t;
    const nt = this.nt;
    const b1 = bang(a1, dth1), c1 = cang(a1, dth1);
    const b2 = bang(a2, dth2), c2 = cang(a2, dth2);
    const b3 = bang(a3, dth3), c3 = cang(a3, dth3);
    const b4 = bang(a4, dth4), c4 = cang(a4, dth4);
    const b7 = bang(a7, dth7), c7 = cang(a7, dth7);
    const b8 = bang(a8, dth8), c8 = cang(a8, dth8);
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const q = i * nt + j, r = j * nt + i;
        const set = (key: string, val: number) => { T[key][q] = val; T[key][r] = val; };
        set('k', k); set('cut0', cut0); set('cutc', cutC); set('cutlo', cutLo); set('cuthi', cutHi);
        set('cutlc', cLc); set('cuthc', cHc); set('blo', bLo); set('bhi', bHi);
        set('a1', a1); set('th1', th1); set('dth1ast', dth1); set('b1', b1); set('dth1c', c1);
        set('a2', a2); set('th2', th2); set('dth2ast', dth2); set('b2', b2); set('dth2c', c2);
        set('a3', a3); set('th3', th3); set('dth3ast', dth3); set('b3', b3); set('dth3c', c3);
        set('a4', a4); set('th4', th4); set('dth4ast', dth4); set('b4', b4); set('dth4c', c4);
        set('a7', a7); set('th7', th7); set('dth7ast', dth7); set('b7', b7); set('dth7c', c7);
        set('a8', a8); set('th8', th8); set('dth8ast', dth8); set('b8', b8); set('dth8c', c8);
        this.set[q] = 1;
      }
    }
  }

  /** pair_oxdna_xstk.cpp init_one(): returns cut_xst_hc[i][j]. */
  initOne(i: number, j: number): number {
    const nt = this.nt;
    if (!this.set[i * nt + j]) throw new StyleError('Coefficient mixing not defined in oxDNA');
    const q = i * nt + j;
    this.t.cutsqhc[q] = this.t.cuthc[q] * this.t.cuthc[q];
    return this.t.cuthc[q];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.custom.has(OXDNA_5P) || !s.quat || !s.torque) throw new StyleError('pair_style oxdna/xstk requires atom_style hybrid bond ellipsoid oxdna');
    const list = pc.half!;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nall = pc.nall;
    const nt = this.nt;
    const T = this.t;
    const constants = oxdnaConstants(s.units.style);
    const dxCbs = constants.dx_cbs_oxdna1;
    const special = pc.specialLJ;
    const tq = new Float64Array(3 * nall);
    const eatom = pc.eatom, vatom = pc.vatom;
    const L = new Float64Array(9 * nall);
    for (let k = 0; k < nall; k++) {
      const o = owner[k];
      if (isEllipsoid(s, o)) lrfOf(s.quat, 4 * o, L, 9 * k);
    }
    const rbs = new Float64Array(3 * nall);
    const tmp = [0, 0, 0];
    for (let k = 0; k < nall; k++) {
      baseSite(dxCbs, L.subarray(9 * k, 9 * k + 3), tmp);
      rbs[3 * k] = tmp[0]; rbs[3 * k + 1] = tmp[1]; rbs[3 * k + 2] = tmp[2];
    }
    let evdwl = 0;
    const delr = [0, 0, 0], drn = [0, 0, 0], delf = [0, 0, 0], delta = [0, 0, 0], deltb = [0, 0, 0], dir = [0, 0, 0];
    const ax = [0, 0, 0], ay = [0, 0, 0], az = [0, 0, 0], bx = [0, 0, 0], by = [0, 0, 0], bz = [0, 0, 0];
    const cross = (u: readonly number[], v: readonly number[], out: number[]) => {
      out[0] = u[1] * v[2] - u[2] * v[1]; out[1] = u[2] * v[0] - u[0] * v[2]; out[2] = u[0] * v[1] - u[1] * v[0];
    };
    const clamp = (c: number) => (c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c);
    const dot = (u: readonly number[], v: readonly number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    for (let a = 0; a < list.inum; a++) {
      for (let c = 0; c < 3; c++) { ax[c] = L[9 * a + c]; ay[c] = L[9 * a + 3 + c]; az[c] = L[9 * a + 6 + c]; }
      const jstart = list.firstneigh[a], jend = jstart + list.numneigh[a];
      for (let kk = jstart; kk < jend; kk++) {
        const jj = list.neighbors[kk];
        const factorLJ = special[jj >>> SBBITS];
        const b = jj & NEIGHMASK;
        const atype = type[a], btype = type[b];
        for (let c = 0; c < 3; c++) { bx[c] = L[9 * b + c]; by[c] = L[9 * b + 3 + c]; bz[c] = L[9 * b + 6 + c]; }
        for (let c = 0; c < 3; c++) delr[c] = x[3 * a + c] + rbs[3 * a + c] - x[3 * b + c] - rbs[3 * b + c];
        const rsq = delr[0] * delr[0] + delr[1] * delr[1] + delr[2] * delr[2];
        const r = Math.sqrt(rsq);
        const rinv = 1.0 / r;
        for (let c = 0; c < 3; c++) drn[c] = delr[c] * rinv;
        const ij = atype * nt + btype;
        const f2 = F2(r, T.k[ij], T.cut0[ij], T.cutlc[ij], T.cuthc[ij], T.cutlo[ij], T.cuthi[ij], T.blo[ij], T.bhi[ij], T.cutc[ij]);
        if (f2 === 0.0) continue;
        let cost1 = -dot(ax, bx); cost1 = clamp(cost1);
        const theta1 = Math.acos(cost1);
        const f4t1 = F4(theta1, T.a1[ij], T.th1[ij], T.dth1ast[ij], T.b1[ij], T.dth1c[ij]);
        if (f4t1 === 0.0) continue;
        let cost2 = -dot(ax, drn); cost2 = clamp(cost2);
        const theta2 = Math.acos(cost2);
        const f4t2 = F4(theta2, T.a2[ij], T.th2[ij], T.dth2ast[ij], T.b2[ij], T.dth2c[ij]);
        if (f4t2 === 0.0) continue;
        let cost3 = dot(bx, drn); cost3 = clamp(cost3);
        const theta3 = Math.acos(cost3);
        const f4t3 = F4(theta3, T.a3[ij], T.th3[ij], T.dth3ast[ij], T.b3[ij], T.dth3c[ij]);
        if (f4t3 === 0.0) continue;
        // the theta4 factor (oxDNA only): f4t4 = 1 otherwise, and the product with 1 leaves every bit unchanged
        let f4t4 = 1.0, theta4 = 0.0, theta4p = 0.0;
        if (this.withTheta4) {
          const cost4 = clamp(dot(az, bz));
          theta4 = Math.acos(cost4);
          theta4p = PI - theta4;
          f4t4 = F4(theta4, T.a4[ij], T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij])
            + F4(theta4p, T.a4[ij], T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij]);
          if (f4t4 === 0.0) continue;
        }
        let cost7 = -dot(az, drn); cost7 = clamp(cost7);
        const theta7 = Math.acos(cost7);
        const theta7p = PI - theta7;
        const f4t7 = F4(theta7, T.a7[ij], T.th7[ij], T.dth7ast[ij], T.b7[ij], T.dth7c[ij])
          + F4(theta7p, T.a7[ij], T.th7[ij], T.dth7ast[ij], T.b7[ij], T.dth7c[ij]);
        if (f4t7 === 0.0) continue;
        let cost8 = dot(bz, drn); cost8 = clamp(cost8);
        const theta8 = Math.acos(cost8);
        const theta8p = PI - theta8;
        const f4t8 = F4(theta8, T.a8[ij], T.th8[ij], T.dth8ast[ij], T.b8[ij], T.dth8c[ij])
          + F4(theta8p, T.a8[ij], T.th8[ij], T.dth8ast[ij], T.b8[ij], T.dth8c[ij]);
        const e = f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factorLJ;
        if (e === 0.0) continue;
        const df2 = DF2(r, T.k[ij], T.cut0[ij], T.cutlc[ij], T.cuthc[ij], T.cutlo[ij], T.cuthi[ij], T.blo[ij], T.bhi[ij]);
        const df4t1 = DF4(theta1, T.a1[ij], T.th1[ij], T.dth1ast[ij], T.b1[ij], T.dth1c[ij]) / Math.sin(theta1);
        const df4t2 = DF4(theta2, T.a2[ij], T.th2[ij], T.dth2ast[ij], T.b2[ij], T.dth2c[ij]) / Math.sin(theta2);
        const df4t3 = DF4(theta3, T.a3[ij], T.th3[ij], T.dth3ast[ij], T.b3[ij], T.dth3c[ij]) / Math.sin(theta3);
        const df4t4 = this.withTheta4
          ? (DF4(theta4, T.a4[ij], T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij])
            - DF4(theta4p, T.a4[ij], T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij])) / Math.sin(theta4)
          : 0.0;
        const df4t7 = (DF4(theta7, T.a7[ij], T.th7[ij], T.dth7ast[ij], T.b7[ij], T.dth7c[ij])
          - DF4(theta7p, T.a7[ij], T.th7[ij], T.dth7ast[ij], T.b7[ij], T.dth7c[ij])) / Math.sin(theta7);
        const df4t8 = (DF4(theta8, T.a8[ij], T.th8[ij], T.dth8ast[ij], T.b8[ij], T.dth8c[ij])
          - DF4(theta8p, T.a8[ij], T.th8[ij], T.dth8ast[ij], T.b8[ij], T.dth8c[ij])) / Math.sin(theta8);
        delf[0] = 0.0; delf[1] = 0.0; delf[2] = 0.0;
        delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
        deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
        let finc = -df2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * rinv * factorLJ;
        for (let c = 0; c < 3; c++) delf[c] += delr[c] * finc;
        if (theta2 !== 0.0) {
          finc = -f2 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost2 + ax[c]) * finc;
        }
        if (theta3 !== 0.0) {
          finc = -f2 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost3 - bx[c]) * finc;
        }
        if (theta7 !== 0.0) {
          finc = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost7 + az[c]) * finc;
        }
        if (theta8 !== 0.0) {
          finc = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost8 - bz[c]) * finc;
        }
        for (let c = 0; c < 3; c++) f[3 * a + c] += delf[c];
        cross([rbs[3 * a], rbs[3 * a + 1], rbs[3 * a + 2]], delf, delta);
        for (let c = 0; c < 3; c++) tq[3 * a + c] += delta[c];
        for (let c = 0; c < 3; c++) f[3 * b + c] -= delf[c];
        cross([rbs[3 * b], rbs[3 * b + 1], rbs[3 * b + 2]], delf, deltb);
        for (let c = 0; c < 3; c++) tq[3 * b + c] -= deltb[c];
        evdwl += e;
        if (eatom) { eatom[a] += 0.5 * e; eatom[b] += 0.5 * e; }
        if (vatom) {
          const rx = x[3 * a] - x[3 * b], ry = x[3 * a + 1] - x[3 * b + 1], rz = x[3 * a + 2] - x[3 * b + 2];
          const w = [rx * delf[0], ry * delf[1], rz * delf[2], rx * delf[1], rx * delf[2], ry * delf[2]];
          for (let c = 0; c < 6; c++) { vatom[6 * a + c] += 0.5 * w[c]; vatom[6 * b + c] += 0.5 * w[c]; }
        }
        delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
        deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
        if (theta1 !== 0.0) {
          const tpair = -f2 * df4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factorLJ;
          cross(ax, bx, dir);
          for (let c = 0; c < 3; c++) { delta[c] += dir[c] * tpair; deltb[c] += dir[c] * tpair; }
        }
        if (theta2 !== 0.0) {
          const tpair = -f2 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factorLJ;
          cross(ax, drn, dir);
          for (let c = 0; c < 3; c++) delta[c] += dir[c] * tpair;
        }
        if (theta3 !== 0.0) {
          const tpair = -f2 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * factorLJ;
          cross(bx, drn, dir);
          for (let c = 0; c < 3; c++) deltb[c] += dir[c] * tpair;
        }
        if (theta4 !== 0.0 && theta4p !== 0.0) {
          const tpair = -f2 * f4t1 * f4t2 * f4t3 * df4t4 * f4t7 * f4t8 * factorLJ;
          cross(bz, az, dir);
          for (let c = 0; c < 3; c++) { delta[c] += dir[c] * tpair; deltb[c] += dir[c] * tpair; }
        }
        if (theta7 !== 0.0) {
          const tpair = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * factorLJ;
          cross(az, drn, dir);
          for (let c = 0; c < 3; c++) delta[c] += dir[c] * tpair;
        }
        if (theta8 !== 0.0) {
          const tpair = -f2 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * factorLJ;
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

/** pair_style oxrna2/xstk (PairOxrna2Xstk): the cross-stacking term without the theta4 factor (pair_oxrna2_xstk.cpp). */
export class PairOxrna2Xstk extends PairOxdnaXstk {
  override readonly name: string = 'oxrna2/xstk';
  protected override withTheta4 = false;
}
