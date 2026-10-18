/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna_coaxstk.cpp and
 * pair_oxdna_coaxstk.h (class PairOxdnaCoaxstk, pair_style oxdna/coaxstk): the oxDNA coaxial stacking
 * interaction between stacked nucleotides (stacking sites), with the angular factors of the helix and the
 * cosine term cos(phi3) built from the backbone and base frames. Original authors (from the headers): Oliver
 * Henrich (University of Strathclyde, Glasgow) and Kierran Falloon (University of Strathclyde, Glasgow);
 * described by Ouldridge, Louis and Doye (J Chem Phys 2011). LAMMPS is Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Atoms with both a 3' and a 5' partner (the interior of a strand) take no coaxial stacking; the C++ skips them
 * (the "continue" lines of compute), and so does this port.
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { OXDNA_3P, OXDNA_5P, isEllipsoid } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { DF2, DF4, DF5, F2, F4, F5, backboneSite, lrfOf, oxdnaConstants, stackingSite } from './common';
import { sin as libmSin } from '../libm/sincos';
import { acos as libmAcos } from '../libm/acos';

/** The 21 numbers after "I J" (pair_coeff) or after the potential-file name, in order. */
const CX_NAMES = ['k_cxst', 'cut_cxst_0', 'cut_cxst_c', 'cut_cxst_lo', 'cut_cxst_hi',
  'a_cxst1', 'theta_cxst1_0', 'dtheta_cxst1_ast', 'a_cxst4', 'theta_cxst4_0', 'dtheta_cxst4_ast',
  'a_cxst5', 'theta_cxst5_0', 'dtheta_cxst5_ast', 'a_cxst6', 'theta_cxst6_0', 'dtheta_cxst6_ast',
  'a_cxst3p', 'cosphi_cxst3p_ast', 'a_cxst4p', 'cosphi_cxst4p_ast'];

const PI = Math.PI;
type Tab = Float64Array;

export class PairOxdnaCoaxstk extends Pair {
  readonly name = 'oxdna/coaxstk';
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
      'a1', 'th1', 'dth1ast', 'b1', 'dth1c', 'a4', 'th4', 'dth4ast', 'b4', 'dth4c', 'a5', 'th5', 'dth5ast', 'b5', 'dth5c',
      'a6', 'th6', 'dth6ast', 'b6', 'dth6c', 'a3p', 'cos3past', 'b3p', 'cos3pc', 'a4p', 'cos4past', 'b4p', 'cos4pc']) {
      this.t[k] = new Float64Array(n);
    }
    this.set = new Uint8Array(n);
  }

  /**
   * pair_oxdna_coaxstk.cpp coeff(): "pair_coeff I J <21 numbers>" (narg 23) or "pair_coeff I J file" (narg 3).
   * b_cxst3p = a^2 cos*^2 / (1 - a cos*^2) and cosphi_cxst3p_c = 1 / (a cos*), as for the angle terms.
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 23 && args.length !== 3) throw new StyleError('Incorrect args for pair coefficients in oxdna/coaxstk');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    let v: number[];
    if (args.length === 23) {
      v = args.slice(2).map((w, k) => parseNum(w, CX_NAMES[k]));
    } else {
      const text = ctx?.readFile(args[2]) ?? '';
      let found: number[] | null = null;
      for (const line of text.split(/\r?\n/)) {
        const w = line.replace(/#.*$/, '').trim().split(/\s+/).filter((x) => x.length);
        if (w.length < 3) continue;
        if (w[0] === args[0] && w[1] === args[1] && w[2] === 'coaxstk') {
          if (w.length < 24) throw new StyleError(`Problem parsing oxdna potential file: coaxstk needs 21 values in ${args[2]}`);
          found = w.slice(3, 24).map((x, k) => parseNum(x, CX_NAMES[k]));
          break;
        }
      }
      if (!found) throw new StyleError(`No corresponding coaxstk potential found in file ${args[2]} for pair type ${args[0]} ${args[1]}`);
      v = found;
    }
    const [k, cut0, cutC, cutLo, cutHi, a1, th1, dth1, a4, th4, dth4, a5, th5, dth5, a6, th6, dth6, a3p, cos3p, a4p, cos4p] = v;
    const dlo = cutLo - cut0, dhi = cutHi - cut0;
    const dcc = k * 0.5 * (cut0 - cutC) * (cut0 - cutC) / k;
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
    const b3p = bang(a3p, cos3p), c3p = cang(a3p, cos3p);
    const b4p = bang(a4p, cos4p), c4p = cang(a4p, cos4p);
    const T = this.t;
    const nt = this.nt;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const q = i * nt + j;
        const set = (key: string, val: number) => { T[key][q] = val; };
        set('k', k); set('cut0', cut0); set('cutc', cutC); set('cutlo', cutLo); set('cuthi', cutHi);
        set('cutlc', cLc); set('cuthc', cHc); set('blo', bLo); set('bhi', bHi);
        set('a1', a1); set('th1', th1); set('dth1ast', dth1); set('b1', b1); set('dth1c', c1);
        set('a4', a4); set('th4', th4); set('dth4ast', dth4); set('b4', b4); set('dth4c', c4);
        set('a5', a5); set('th5', th5); set('dth5ast', dth5); set('b5', b5); set('dth5c', c5);
        set('a6', a6); set('th6', th6); set('dth6ast', dth6); set('b6', b6); set('dth6c', c6);
        set('a3p', a3p); set('cos3past', cos3p); set('b3p', b3p); set('cos3pc', c3p);
        set('a4p', a4p); set('cos4past', cos4p); set('b4p', b4p); set('cos4pc', c4p);
        this.set[q] = 1;
      }
    }
  }

  /**
   * pair_oxdna_coaxstk.cpp init_one(): the symmetric (j, i) copies of the coefficients, returns cut_cxst_hc[i][j].
   */
  initOne(i: number, j: number): number {
    const nt = this.nt;
    if (!this.set[i * nt + j]) throw new StyleError('Coefficient mixing not defined in oxDNA');
    const q = i * nt + j, r = j * nt + i;
    for (const key of Object.keys(this.t)) {
      if (key === 'cutsqhc') continue;
      this.t[key][r] = this.t[key][q];
    }
    this.t.cutsqhc[q] = this.t.cuthc[q] * this.t.cuthc[q];
    this.t.cutsqhc[r] = this.t.cutsqhc[q];
    this.set[r] = 1;
    return this.t.cuthc[q];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.custom.has(OXDNA_5P) || !s.quat || !s.torque) throw new StyleError('pair_style oxdna/coaxstk requires atom_style hybrid bond ellipsoid oxdna');
    const list = pc.half!;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nall = pc.nall;
    const nt = this.nt;
    const T = this.t;
    const constants = oxdnaConstants(s.units.style);
    const dxCbk = constants.dx_cbk_oxdna1, dxCstk = constants.dx_cstk_oxdna1;
    const special = pc.specialLJ;
    const id3p = s.custom.get(OXDNA_3P)!.data, id5p = s.custom.get(OXDNA_5P)!.data;
    const tq = new Float64Array(3 * nall);
    const eatom = pc.eatom, vatom = pc.vatom;
    const L = new Float64Array(9 * nall);
    for (let k = 0; k < nall; k++) {
      const o = owner[k];
      if (isEllipsoid(s, o)) lrfOf(s.quat, 4 * o, L, 9 * k);
    }
    // stacking and backbone site offsets of every atom (owned and ghost)
    const rst = new Float64Array(3 * nall), rbk = new Float64Array(3 * nall);
    const tmp = [0, 0, 0];
    for (let k = 0; k < nall; k++) {
      const e = L.subarray(9 * k, 9 * k + 3);
      stackingSite(dxCstk, e, tmp); rst[3 * k] = tmp[0]; rst[3 * k + 1] = tmp[1]; rst[3 * k + 2] = tmp[2];
      backboneSite(dxCbk, e, tmp); rbk[3 * k] = tmp[0]; rbk[3 * k + 1] = tmp[1]; rbk[3 * k + 2] = tmp[2];
    }
    const gamma = dxCbk - dxCstk;
    const gammacub = gamma * gamma * gamma;
    let evdwl = 0;
    const delr = [0, 0, 0], drn = [0, 0, 0], delf = [0, 0, 0], delta = [0, 0, 0], deltb = [0, 0, 0], delt = [0, 0, 0];
    const dbk = [0, 0, 0], dbkn = [0, 0, 0];
    const ax = [0, 0, 0], ay = [0, 0, 0], az = [0, 0, 0], bx = [0, 0, 0], by = [0, 0, 0], bz = [0, 0, 0];
    const v1 = [0, 0, 0], v2 = [0, 0, 0], v3 = [0, 0, 0], dir = [0, 0, 0];
    const cross = (u: readonly number[], v: readonly number[], out: number[]) => {
      out[0] = u[1] * v[2] - u[2] * v[1]; out[1] = u[2] * v[0] - u[0] * v[2]; out[2] = u[0] * v[1] - u[1] * v[0];
    };
    const clamp = (c: number) => (c > 1.0 ? 1.0 : c < -1.0 ? -1.0 : c);
    const dot = (u: readonly number[], v: readonly number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const interior = (k: number) => id3p[k] !== 0 && id5p[k] !== 0;
    for (let a = 0; a < list.inum; a++) {
      if (interior(a)) continue;
      const atype = type[a];
      for (let c = 0; c < 3; c++) { ax[c] = L[9 * a + c]; ay[c] = L[9 * a + 3 + c]; az[c] = L[9 * a + 6 + c]; }
      const jstart = list.firstneigh[a], jend = jstart + list.numneigh[a];
      for (let kk = jstart; kk < jend; kk++) {
        const jj = list.neighbors[kk];
        const factorLJ = special[jj >>> SBBITS];
        const b = jj & NEIGHMASK;
        const ob = owner[b];
        if (interior(ob)) continue;
        const btype = type[b];
        for (let c = 0; c < 3; c++) { bx[c] = L[9 * b + c]; by[c] = L[9 * b + 3 + c]; bz[c] = L[9 * b + 6 + c]; }
        // stacking distance
        for (let c = 0; c < 3; c++) delr[c] = x[3 * a + c] + rst[3 * a + c] - x[3 * b + c] - rst[3 * b + c];
        const rsqStk = delr[0] * delr[0] + delr[1] * delr[1] + delr[2] * delr[2];
        const rStk = Math.sqrt(rsqStk);
        const rinvStk = 1.0 / rStk;
        for (let c = 0; c < 3; c++) drn[c] = delr[c] * rinvStk;
        // backbone distance (for the cos phi3 term)
        for (let c = 0; c < 3; c++) dbk[c] = x[3 * a + c] + rbk[3 * a + c] - x[3 * b + c] - rbk[3 * b + c];
        const rsqBk = dbk[0] * dbk[0] + dbk[1] * dbk[1] + dbk[2] * dbk[2];
        const rBk = Math.sqrt(rsqBk);
        const rinvBk = 1.0 / rBk;
        for (let c = 0; c < 3; c++) dbkn[c] = dbk[c] * rinvBk;
        const ij = atype * nt + btype;
        let cost1 = -dot(ax, bx); cost1 = clamp(cost1);
        const theta1 = libmAcos(cost1);
        const theta1p = 2 * PI - theta1;
        const f4t1 = F4(theta1, T.a1[ij], T.th1[ij], T.dth1ast[ij], T.b1[ij], T.dth1c[ij])
          + F4(theta1p, T.a1[ij], T.th1[ij], T.dth1ast[ij], T.b1[ij], T.dth1c[ij]);
        if (f4t1 === 0.0) continue;
        let cost4 = dot(az, bz); cost4 = clamp(cost4);
        const theta4 = libmAcos(cost4);
        const f4t4 = F4(theta4, T.a4[ij], T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij])
          + F4(theta4, T.a4[ij], PI - T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij]);
        if (f4t4 === 0.0) continue;
        let cost5 = dot(drn, az); cost5 = clamp(cost5);
        const theta5 = libmAcos(cost5);
        const theta5p = PI - theta5;
        const f4t5 = F4(theta5, T.a5[ij], T.th5[ij], T.dth5ast[ij], T.b5[ij], T.dth5c[ij])
          + F4(theta5p, T.a5[ij], T.th5[ij], T.dth5ast[ij], T.b5[ij], T.dth5c[ij]);
        if (f4t5 === 0.0) continue;
        let cost6 = dot(drn, bz); cost6 = clamp(cost6);
        const theta6 = libmAcos(cost6);
        const theta6p = PI - theta6;
        const f4t6 = F4(theta6, T.a6[ij], T.th6[ij], T.dth6ast[ij], T.b6[ij], T.dth6c[ij])
          + F4(theta6p, T.a6[ij], T.th6[ij], T.dth6ast[ij], T.b6[ij], T.dth6c[ij]);
        cross(dbkn, ax, v1);
        const cosphi3 = clamp(dot(drn, v1));
        const f2 = F2(rStk, T.k[ij], T.cut0[ij], T.cutlc[ij], T.cuthc[ij], T.cutlo[ij], T.cuthi[ij], T.blo[ij], T.bhi[ij], T.cutc[ij]);
        const f5c3 = F5(cosphi3, T.a3p[ij], T.cos3past[ij], T.b3p[ij], T.cos3pc[ij]);
        const e = f2 * f4t1 * f4t4 * f4t5 * f4t6 * f5c3 * f5c3 * factorLJ;
        if (e === 0.0) continue;
        const df2 = DF2(rStk, T.k[ij], T.cut0[ij], T.cutlc[ij], T.cuthc[ij], T.cutlo[ij], T.cuthi[ij], T.blo[ij], T.bhi[ij]);
        const df4t1 = (DF4(theta1, T.a1[ij], T.th1[ij], T.dth1ast[ij], T.b1[ij], T.dth1c[ij])
          - DF4(theta1p, T.a1[ij], T.th1[ij], T.dth1ast[ij], T.b1[ij], T.dth1c[ij])) / libmSin(theta1);
        const df4t4 = (DF4(theta4, T.a4[ij], T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij])
          + DF4(theta4, T.a4[ij], PI - T.th4[ij], T.dth4ast[ij], T.b4[ij], T.dth4c[ij])) / libmSin(theta4);
        const df4t5 = (DF4(theta5, T.a5[ij], T.th5[ij], T.dth5ast[ij], T.b5[ij], T.dth5c[ij])
          - DF4(theta5p, T.a5[ij], T.th5[ij], T.dth5ast[ij], T.b5[ij], T.dth5c[ij])) / libmSin(theta5);
        const df4t6 = (DF4(theta6, T.a6[ij], T.th6[ij], T.dth6ast[ij], T.b6[ij], T.dth6c[ij])
          - DF4(theta6p, T.a6[ij], T.th6[ij], T.dth6ast[ij], T.b6[ij], T.dth6c[ij])) / libmSin(theta6);
        const df5c3 = DF5(cosphi3, T.a3p[ij], T.cos3past[ij], T.b3p[ij], T.cos3pc[ij]);
        delf[0] = 0.0; delf[1] = 0.0; delf[2] = 0.0;
        delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
        deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
        let finc = -df2 * f4t1 * f4t4 * f4t5 * f4t6 * f5c3 * f5c3 * rinvStk * factorLJ;
        for (let c = 0; c < 3; c++) delf[c] += delr[c] * finc;
        if (theta5 !== 0.0 && theta5p !== 0.0) {
          finc = -f2 * f4t1 * f4t4 * df4t5 * f4t6 * f5c3 * f5c3 * rinvStk * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost5 - az[c]) * finc;
        }
        if (theta6 !== 0.0 && theta6p !== 0.0) {
          finc = -f2 * f4t1 * f4t4 * f4t5 * df4t6 * f5c3 * f5c3 * rinvStk * factorLJ;
          for (let c = 0; c < 3; c++) delf[c] += (drn[c] * cost6 - bz[c]) * finc;
        }
        if (cosphi3 !== 0.0) {
          finc = -f2 * f4t1 * f4t4 * f4t5 * f4t6 * 2.0 * f5c3 * df5c3 * factorLJ;
          const rinvBkCub = rinvBk * rinvBk * rinvBk;
          const aybx = dot(ay, bx), azbx = dot(az, bx);
          const rax = dot(drn, ax), ray = dot(drn, ay), raz = dot(drn, az), rbx = dot(drn, bx);
          const fac = raz * aybx - ray * azbx;
          const dcdrax = -gamma * gamma * fac * rStk * rinvBkCub;
          const dcdray = -gamma * azbx * rinvBk;
          const dcdraz = gamma * aybx * rinvBk;
          const dcdrbx = gamma * gamma * fac * rStk * rinvBkCub;
          const dcdr = -gamma * fac * (gamma * (rax - rbx) + rStk) * rinvBkCub;
          for (let c = 0; c < 3; c++) {
            delf[c] += (drn[c] * dcdr + ((ax[c] - drn[c] * rax) * dcdrax + (ay[c] - drn[c] * ray) * dcdray
              + (az[c] - drn[c] * raz) * dcdraz + (bx[c] - drn[c] * rbx) * dcdrbx) * rinvStk) * finc * factorLJ;
          }
        }
        // f[a] += delf, torque[a] += cross(ra_cstk, delf); f[b] -= delf, torque[b] -= cross(rb_cstk, delf)
        for (let c = 0; c < 3; c++) f[3 * a + c] += delf[c];
        cross([rst[3 * a], rst[3 * a + 1], rst[3 * a + 2]], delf, delta);
        for (let c = 0; c < 3; c++) tq[3 * a + c] += delta[c];
        for (let c = 0; c < 3; c++) f[3 * b + c] -= delf[c];
        cross([rst[3 * b], rst[3 * b + 1], rst[3 * b + 2]], delf, deltb);
        for (let c = 0; c < 3; c++) tq[3 * b + c] -= deltb[c];
        evdwl += e;
        if (eatom) { eatom[a] += 0.5 * e; eatom[b] += 0.5 * e; }
        if (vatom) {
          const rx = x[3 * a] - x[3 * b], ry = x[3 * a + 1] - x[3 * b + 1], rz = x[3 * a + 2] - x[3 * b + 2];
          const w = [rx * delf[0], ry * delf[1], rz * delf[2], rx * delf[1], rx * delf[2], ry * delf[2]];
          for (let c = 0; c < 6; c++) { vatom[6 * a + c] += 0.5 * w[c]; vatom[6 * b + c] += 0.5 * w[c]; }
        }
        // torques of the angular and cos phi3 terms
        delta[0] = 0.0; delta[1] = 0.0; delta[2] = 0.0;
        deltb[0] = 0.0; deltb[1] = 0.0; deltb[2] = 0.0;
        if (theta1 !== 0.0 && theta1p !== 0.0) {
          const tpair = -f2 * df4t1 * f4t4 * f4t5 * f4t6 * f5c3 * f5c3 * factorLJ;
          cross(ax, bx, dir);
          for (let c = 0; c < 3; c++) { delta[c] += dir[c] * tpair; deltb[c] += dir[c] * tpair; }
        }
        if (theta4 !== 0.0) {
          const tpair = -f2 * f4t1 * df4t4 * f4t5 * f4t6 * f5c3 * f5c3 * factorLJ;
          cross(bz, az, dir);
          for (let c = 0; c < 3; c++) { delta[c] += dir[c] * tpair; deltb[c] += dir[c] * tpair; }
        }
        if (theta5 !== 0.0 && theta5p !== 0.0) {
          const tpair = -f2 * f4t1 * f4t4 * df4t5 * f4t6 * f5c3 * f5c3 * factorLJ;
          cross(drn, az, dir);
          for (let c = 0; c < 3; c++) delta[c] += dir[c] * tpair;
        }
        if (theta6 !== 0.0 && theta6p !== 0.0) {
          const tpair = -f2 * f4t1 * f4t4 * f4t5 * df4t6 * f5c3 * f5c3 * factorLJ;
          cross(drn, bz, dir);
          for (let c = 0; c < 3; c++) deltb[c] -= dir[c] * tpair;
        }
        if (cosphi3 !== 0.0) {
          const rinvBkCub = rinvBk * rinvBk * rinvBk;
          const aybx = dot(ay, bx), azbx = dot(az, bx);
          const ray = dot(drn, ay), raz = dot(drn, az);
          const fac = raz * aybx - ray * azbx;
          const dcdaxbx = gammacub * fac * rinvBkCub;
          const dcdaybx = gamma * raz * rinvBk;
          const dcdazbx = -gamma * ray * rinvBk;
          const dcdrax = -gamma * gamma * fac * rStk * rinvBkCub;
          const dcdray = -gamma * azbx * rinvBk;
          const dcdraz = gamma * aybx * rinvBk;
          const dcdrbx = gamma * gamma * fac * rStk * rinvBkCub;
          const tpair = -f2 * f4t1 * f4t4 * f4t5 * f4t6 * 2.0 * f5c3 * df5c3 * factorLJ;
          cross(ax, bx, v1); cross(ay, bx, v2); cross(az, bx, v3);
          for (let c = 0; c < 3; c++) {
            delt[c] = (v1[c] * dcdaxbx + v2[c] * dcdaybx + v3[c] * dcdazbx) * tpair;
            delta[c] += delt[c]; deltb[c] += delt[c];
          }
          cross(ax, drn, v1); cross(ay, drn, v2); cross(az, drn, v3);
          for (let c = 0; c < 3; c++) delta[c] += (v1[c] * dcdrax + v2[c] * dcdray + v3[c] * dcdraz) * tpair;
          cross(bx, drn, v1);
          for (let c = 0; c < 3; c++) deltb[c] -= v1[c] * dcdrbx * tpair;
        }
        for (let c = 0; c < 3; c++) { tq[3 * a + c] += delta[c]; tq[3 * b + c] -= deltb[c]; }
      }
    }
    pc.nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}
