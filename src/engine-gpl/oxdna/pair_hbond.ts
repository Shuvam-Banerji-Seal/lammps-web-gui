/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna_hbond.cpp (and pair_oxrna2_hbond.cpp, the oxRNA2 alpha table) and
 * pair_oxdna_hbond.h (class PairOxdnaHbond, pair_style oxdna/hbond): the oxDNA hydrogen-bonding interaction
 * between the base sites of two nucleotides, with the angular factors f4 of the base-pairing geometry.
 * Original authors (from the headers): Oliver Henrich (University of Strathclyde, Glasgow) and Kierran Falloon
 * (University of Strathclyde, Glasgow); described by Ouldridge, Louis and Doye (J Chem Phys 2011). LAMMPS is
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0);
 * this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Unique base pairing (the custom idc vector of pair_oxdna_hbond.cpp init_style, enabled by a fix named Basepairs)
 * is refused by name. The per-type coefficients are symmetrised in initOne
 * (pair_oxdna_hbond.cpp init_one writes the j,i entries), as in the C++.
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { OXDNA_5P, isEllipsoid } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { DF1, DF4, F1, F4, baseSiteOxdna1, lrfOf, oxdnaConstants, type OxdnaConstants, potentialLines } from './common';
import { sin as libmSin } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';

/** pair_oxdna_hbond.cpp constructor: alpha_hb[i][j] (sequence dependence of the hydrogen bonds, oxDNA). */
const ALPHA_HB_OXDNA: number[][] = [
  [1.00000, 1.00000, 1.00000, 0.82915],
  [1.00000, 1.00000, 1.15413, 1.00000],
  [1.00000, 1.15413, 1.00000, 1.00000],
  [0.82915, 1.00000, 1.00000, 1.00000],
];

/** The 24 numbers after "I J seqav|seqdep" (pair_coeff) or after the potential-file name, in this order. */
const HB_NAMES = ['epsilon_hb', 'a_hb', 'cut_hb_0', 'cut_hb_c', 'cut_hb_lo', 'cut_hb_hi',
  'a_hb1', 'theta_hb1_0', 'dtheta_hb1_ast', 'a_hb2', 'theta_hb2_0', 'dtheta_hb2_ast',
  'a_hb3', 'theta_hb3_0', 'dtheta_hb3_ast', 'a_hb4', 'theta_hb4_0', 'dtheta_hb4_ast',
  'a_hb7', 'theta_hb7_0', 'dtheta_hb7_ast', 'a_hb8', 'theta_hb8_0', 'dtheta_hb8_ast'];

/** Per-type-pair table, (ntypes+1)^2 entries. */
type Tab = Float64Array;

export class PairOxdnaHbond extends Pair {
  readonly name: string = 'oxdna/hbond';
  virialFdotr = true;
  /** alpha_hb[i][j] of the model (the oxRNA2 subclass sets its own table) */
  protected alpha: number[][] = ALPHA_HB_OXDNA;
  /** base site rule of the nucleotide model (the oxDNA3 subclass takes the base type into account) */
  protected baseSiteOf: (c: OxdnaConstants, type: number, e1: ArrayLike<number>, out: number[]) => void = baseSiteOxdna1;
  protected nt = 0;
  protected seqdep = false;
  protected t!: Record<string, Tab>;
  protected set!: Uint8Array;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nt = ntypes + 1;
    const n = this.nt * this.nt;
    this.t = {};
    for (const k of ['epsilon', 'a', 'cut0', 'cutc', 'cutlo', 'cuthi', 'cutlc', 'cuthc', 'blo', 'bhi', 'shift', 'cutsqhc',
      'a1', 'th1', 'dth1ast', 'b1', 'dth1c', 'a2', 'th2', 'dth2ast', 'b2', 'dth2c', 'a3', 'th3', 'dth3ast', 'b3', 'dth3c',
      'a4', 'th4', 'dth4ast', 'b4', 'dth4c', 'a7', 'th7', 'dth7ast', 'b7', 'dth7c', 'a8', 'th8', 'dth8ast', 'b8', 'dth8c']) {
      this.t[k] = new Float64Array(n);
    }
    this.set = new Uint8Array(n);
  }

  /**
   * pair_oxdna_hbond.cpp coeff(): "pair_coeff I J seqav|seqdep <24 numbers>" or "pair_coeff I J seqav|seqdep file".
   * The b, cut_lc, cut_hc and shift values come from the same formulas as the stacking style.
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 27 && args.length !== 4) throw new StyleError('Incorrect args for pair coefficients in oxdna/hbond');
    if (args[2] !== 'seqav' && args[2] !== 'seqdep') throw new StyleError('Incorrect setting, select seqav or seqdep in oxdna/hbond');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    this.seqdep = args[2] === 'seqdep';
    let v: number[];
    if (args.length === 27) {
      v = args.slice(3).map((w, k) => parseNum(w, HB_NAMES[k]));
    } else {
      const text = ctx?.readFile(args[3]) ?? '';
      let found: number[] | null = null;
      for (const w of potentialLines(text, [args[0], args[1]], 'hbond')) {
        if (w.length < 3) continue;
        if (w[2] === 'hbond') {
          if (w.length < 27) throw new StyleError(`Problem parsing oxDNA potential file: hbond needs 24 values in ${args[3]}`);
          found = w.slice(3, 27).map((x, k) => parseNum(x, HB_NAMES[k]));
          break;
        }
      }
      if (!found) throw new StyleError(`No corresponding hbond potential found in file ${args[3]} for pair type ${args[0]} ${args[1]}`);
      v = found;
    }
    const [eps, a, cut0, cutC, cutLo, cutHi, a1, th1, dth1, a2, th2, dth2, a3, th3, dth3, a4, th4, dth4, a7, th7, dth7, a8, th8, dth8] = v;
    const e = (x: number) => libmExp(-a * x);
    const ulo = cutLo - cut0, uhi = cutHi - cut0, uc = cutC - cut0;
    const bLo = 2 * a * e(ulo) * 2 * a * e(ulo) * (1 - e(ulo)) * (1 - e(ulo))
      / (4 * ((1 - e(ulo)) * (1 - e(ulo)) - (1 - e(uc)) * (1 - e(uc))));
    const cLc = cutLo - a * e(ulo) * (1 - e(ulo)) / bLo;
    const bHi = 2 * a * e(uhi) * 2 * a * e(uhi) * (1 - e(uhi)) * (1 - e(uhi))
      / (4 * ((1 - e(uhi)) * (1 - e(uhi)) - (1 - e(uc)) * (1 - e(uc))));
    const cHc = cutHi - a * e(uhi) * (1 - e(uhi)) / bHi;
    const tmp = 1 - libmExp(-(cutC - cut0) * a);
    const shiftOne = eps * tmp * tmp;
    const bang = (aa: number, ast: number) => aa * aa * ast * ast / (1 - aa * ast * ast);
    const cang = (aa: number, ast: number) => 1 / (aa * ast);
    const b1 = bang(a1, dth1), c1 = cang(a1, dth1);
    const b2 = bang(a2, dth2), c2 = cang(a2, dth2);
    const b3 = bang(a3, dth3), c3 = cang(a3, dth3);
    const b4 = bang(a4, dth4), c4 = cang(a4, dth4);
    const b7 = bang(a7, dth7), c7 = cang(a7, dth7);
    const b8 = bang(a8, dth8), c8 = cang(a8, dth8);
    const nt = this.nt;
    const mod4 = (t: number) => (t % 4 === 0 ? 4 : t % 4);
    const T = this.t;
    for (let i = ilo; i <= ihi; i++) {
      const im = mod4(i);
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const jm = mod4(j);
        const q = i * nt + j;
        T.epsilon[q] = eps;
        if (this.seqdep) T.epsilon[q] *= this.alpha[im - 1][jm - 1];
        T.a[q] = a; T.cut0[q] = cut0; T.cutc[q] = cutC; T.cutlo[q] = cutLo; T.cuthi[q] = cutHi;
        T.cutlc[q] = cLc; T.cuthc[q] = cHc; T.blo[q] = bLo; T.bhi[q] = bHi;
        T.shift[q] = shiftOne;
        if (this.seqdep) T.shift[q] *= this.alpha[im - 1][jm - 1];
        T.a1[q] = a1; T.th1[q] = th1; T.dth1ast[q] = dth1; T.b1[q] = b1; T.dth1c[q] = c1;
        T.a2[q] = a2; T.th2[q] = th2; T.dth2ast[q] = dth2; T.b2[q] = b2; T.dth2c[q] = c2;
        T.a3[q] = a3; T.th3[q] = th3; T.dth3ast[q] = dth3; T.b3[q] = b3; T.dth3c[q] = c3;
        T.a4[q] = a4; T.th4[q] = th4; T.dth4ast[q] = dth4; T.b4[q] = b4; T.dth4c[q] = c4;
        T.a7[q] = a7; T.th7[q] = th7; T.dth7ast[q] = dth7; T.b7[q] = b7; T.dth7c[q] = c7;
        T.a8[q] = a8; T.th8[q] = th8; T.dth8ast[q] = dth8; T.b8[q] = b8; T.dth8c[q] = c8;
        this.set[q] = 1;
      }
    }
  }

  /** pair_oxdna_hbond.cpp init_one(): symmetric copies of the (i, j) entries, returns cut_hb_hc[i][j]. */
  initOne(i: number, j: number): number {
    const nt = this.nt;
    if (!this.set[i * nt + j]) throw new StyleError('Coefficient mixing not defined in oxDNA');
    const im = i % 4 === 0 ? 4 : i % 4, jm = j % 4 === 0 ? 4 : j % 4;
    const q = i * nt + j, r = j * nt + i;
    const T = this.t;
    for (const k of Object.keys(T)) {
      if (k === 'epsilon' || k === 'shift') continue;
      T[k][r] = T[k][q];
    }
    if (this.seqdep) {
      T.epsilon[r] = T.epsilon[q] / this.alpha[im - 1][jm - 1] * this.alpha[jm - 1][im - 1];
      T.shift[r] = T.shift[q] / this.alpha[im - 1][jm - 1] * this.alpha[jm - 1][im - 1];
    } else {
      T.epsilon[r] = T.epsilon[q];
      T.shift[r] = T.shift[q];
    }
    T.cutsqhc[q] = T.cuthc[q] * T.cuthc[q];
    T.cutsqhc[r] = T.cutsqhc[q];
    this.set[r] = 1;
    return T.cuthc[q];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.custom.has(OXDNA_5P) || !s.quat || !s.torque) throw new StyleError('pair_style oxdna/hbond requires atom_style hybrid bond ellipsoid oxdna');
    const list = pc.half!;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nall = pc.nall;
    const nt = this.nt;
    const T = this.t;
    const constants = oxdnaConstants(s.units.style);
    const special = pc.specialLJ;
    const tq = new Float64Array(3 * nall);
    const eatom = pc.eatom, vatom = pc.vatom;
    // local reference frames (owned atoms, copied to ghosts) and the base-site offsets
    const L = new Float64Array(9 * nall);
    for (let k = 0; k < nall; k++) {
      const o = owner[k];
      if (isEllipsoid(s, o)) lrfOf(s.quat, 4 * o, L, 9 * k);
    }
    const rbs = new Float64Array(3 * nall);
    const tmp = [0, 0, 0];
    for (let k = 0; k < nall; k++) {
      this.baseSiteOf(constants, type[k], L.subarray(9 * k, 9 * k + 3), tmp);
      rbs[3 * k] = tmp[0]; rbs[3 * k + 1] = tmp[1]; rbs[3 * k + 2] = tmp[2];
    }
    // unique base pairing (pair_oxdna_hbond.cpp compute, the idc vector of fix Basepairs, i_idc): a pair is skipped
    // unless one atom's complement ID is the other atom's tag, or one of the two IDs is not positive
    const idc = s.custom.get('idc')?.data ?? null;
    let evdwl = 0;
    const delr = [0, 0, 0], drn = [0, 0, 0], delf = [0, 0, 0], delta = [0, 0, 0], deltb = [0, 0, 0], dir = [0, 0, 0];
    const ax = [0, 0, 0], ay = [0, 0, 0], az = [0, 0, 0], bx = [0, 0, 0], by = [0, 0, 0], bz = [0, 0, 0];
    const cross = (u: readonly number[], v: readonly number[], out: number[]) => {
      out[0] = u[1] * v[2] - u[2] * v[1]; out[1] = u[2] * v[0] - u[0] * v[2]; out[2] = u[0] * v[1] - u[1] * v[0];
    };
    const clamp = (c: number) => (c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c);
    const dot = (u: readonly number[], v: readonly number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    for (let ia = 0; ia < list.inum; ia++) {
      const a = ia;
      const atype = type[a];
      for (let c = 0; c < 3; c++) {
        ax[c] = L[9 * a + c]; ay[c] = L[9 * a + 3 + c]; az[c] = L[9 * a + 6 + c];
      }
      const jstart = list.firstneigh[a], jend = jstart + list.numneigh[a];
      for (let kk = jstart; kk < jend; kk++) {
        const jj = list.neighbors[kk];
        const factorLJ = special[jj >>> SBBITS];
        const b = jj & NEIGHMASK;
        const btype = type[b];
        if (idc) {
          const ida = idc[a], idb = idc[owner[b]];
          if (ida !== s.id[owner[b]] && idb !== s.id[a] && ida > 0 && idb > 0) continue;
        }
        for (let c = 0; c < 3; c++) {
          bx[c] = L[9 * b + c]; by[c] = L[9 * b + 3 + c]; bz[c] = L[9 * b + 6 + c];
        }
        for (let c = 0; c < 3; c++) delr[c] = x[3 * a + c] + rbs[3 * a + c] - x[3 * b + c] - rbs[3 * b + c];
        const rsq = delr[0] * delr[0] + delr[1] * delr[1] + delr[2] * delr[2];
        const r = Math.sqrt(rsq);
        const rinv = 1.0 / r;
        for (let c = 0; c < 3; c++) drn[c] = delr[c] * rinv;
        const ij = atype * nt + btype;
        const eps = T.epsilon[ij], aa = T.a[ij], cut0 = T.cut0[ij], cutlc = T.cutlc[ij], cuthc = T.cuthc[ij];
        const cutlo = T.cutlo[ij], cuthi = T.cuthi[ij], blo = T.blo[ij], bhi = T.bhi[ij], shift = T.shift[ij];
        const f1 = F1(r, eps, aa, cut0, cutlc, cuthc, cutlo, cuthi, blo, bhi, shift);
        if (f1 === 0.0) continue;
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
        let cost4 = dot(az, bz); cost4 = clamp(cost4);
        const theta4 = Math.acos(cost4);
        const f4t4 = F4(theta4, T.a4[ij], T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij]);
        if (f4t4 === 0.0) continue;
        let cost7 = -dot(az, drn); cost7 = clamp(cost7);
        const theta7 = Math.acos(cost7);
        const f4t7 = F4(theta7, T.a7[ij], T.th7[ij], T.dth7ast[ij], T.b7[ij], T.dth7c[ij]);
        if (f4t7 === 0.0) continue;
        let cost8 = dot(bz, drn); cost8 = clamp(cost8);
        const theta8 = Math.acos(cost8);
        const f4t8 = F4(theta8, T.a8[ij], T.th8[ij], T.dth8ast[ij], T.b8[ij], T.dth8c[ij]);
        const e = f1 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factorLJ;
        if (e === 0.0) continue;
        const df1 = DF1(r, eps, aa, cut0, cutlc, cuthc, cutlo, cuthi, blo, bhi);
        const df4t1 = DF4(theta1, T.a1[ij], T.th1[ij], T.dth1ast[ij], T.b1[ij], T.dth1c[ij]) / libmSin(theta1);
        const df4t2 = DF4(theta2, T.a2[ij], T.th2[ij], T.dth2ast[ij], T.b2[ij], T.dth2c[ij]) / libmSin(theta2);
        const df4t3 = DF4(theta3, T.a3[ij], T.th3[ij], T.dth3ast[ij], T.b3[ij], T.dth3c[ij]) / libmSin(theta3);
        const df4t4 = DF4(theta4, T.a4[ij], T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij]) / libmSin(theta4);
        const df4t7 = DF4(theta7, T.a7[ij], T.th7[ij], T.dth7ast[ij], T.b7[ij], T.dth7c[ij]) / libmSin(theta7);
        const df4t8 = DF4(theta8, T.a8[ij], T.th8[ij], T.dth8ast[ij], T.b8[ij], T.dth8c[ij]) / libmSin(theta8);
        delf[0] = 0.0; delf[1] = 0.0; delf[2] = 0.0;
        delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
        deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
        // force along delr_bsbs, then the angular force terms
        let finc = -df1 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factorLJ;
        for (let c = 0; c < 3; c++) delf[c] += delr[c] * finc;
        if (theta2 !== 0.0) {
          finc = -f1 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost2 + ax[c]) * finc;
        }
        if (theta3 !== 0.0) {
          finc = -f1 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost3 - bx[c]) * finc;
        }
        if (theta7 !== 0.0) {
          finc = -f1 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost7 + az[c]) * finc;
        }
        if (theta8 !== 0.0) {
          finc = -f1 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * rinv * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost8 - bz[c]) * finc;
        }
        // f[a] += delf, torque[a] += cross(ra, delf); f[b] -= delf, torque[b] -= cross(rb, delf)
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
        // torque terms of the angles (the dir vectors are the cross products of the local axes)
        delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
        deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
        if (theta1 !== 0.0) {
          const tpair = -f1 * df4t1 * f4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factorLJ;
          cross(ax, bx, dir);
          for (let c = 0; c < 3; c++) { delta[c] += dir[c] * tpair; deltb[c] += dir[c] * tpair; }
        }
        if (theta2 !== 0.0) {
          const tpair = -f1 * f4t1 * df4t2 * f4t3 * f4t4 * f4t7 * f4t8 * factorLJ;
          cross(ax, drn, dir);
          for (let c = 0; c < 3; c++) delta[c] += dir[c] * tpair;
        }
        if (theta3 !== 0.0) {
          const tpair = -f1 * f4t1 * f4t2 * df4t3 * f4t4 * f4t7 * f4t8 * factorLJ;
          cross(bx, drn, dir);
          for (let c = 0; c < 3; c++) deltb[c] += dir[c] * tpair;
        }
        if (theta4 !== 0.0) {
          const tpair = -f1 * f4t1 * f4t2 * f4t3 * df4t4 * f4t7 * f4t8 * factorLJ;
          cross(bz, az, dir);
          for (let c = 0; c < 3; c++) { delta[c] += dir[c] * tpair; deltb[c] += dir[c] * tpair; }
        }
        if (theta7 !== 0.0) {
          const tpair = -f1 * f4t1 * f4t2 * f4t3 * f4t4 * df4t7 * f4t8 * factorLJ;
          cross(az, drn, dir);
          for (let c = 0; c < 3; c++) delta[c] += dir[c] * tpair;
        }
        if (theta8 !== 0.0) {
          const tpair = -f1 * f4t1 * f4t2 * f4t3 * f4t4 * f4t7 * df4t8 * factorLJ;
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

/** pair_style oxrna2/hbond (PairOxrna2Hbond): the oxRNA2 base-pairing strengths alpha_hb (pair_oxrna2_hbond.cpp). */
export class PairOxrna2Hbond extends PairOxdnaHbond {
  override readonly name: string = 'oxrna2/hbond';
  // A:0 C:1 G:2 U:3, 5'- [i][j] -3'
  protected override alpha: number[][] = [
    [1.00000, 1.00000, 1.00000, 0.94253],
    [1.00000, 1.00000, 1.22288, 1.00000],
    [1.00000, 1.22288, 1.00000, 0.58655],
    [0.94253, 1.00000, 0.58655, 1.00000],
  ];
}
