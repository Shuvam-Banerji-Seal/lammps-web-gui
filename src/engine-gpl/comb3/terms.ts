/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_comb3.cpp and src/MANYBODY/pair_comb3.h (the short-range and bond-order
 * functions: comb_fc, comb_fc_d, comb_fccc..comb_fccch_d, comb_fa, comb_bij_d, coord, cntri_int, comb_gijk,
 * comb_gijk_d, zeta, repulsive, selfp6p, ep6p, fp6p, force_zeta (without the radical and torsion parts),
 * attractive, comb_zetaterm_d, costheta_d, self), at commit b332f28. Authors credited in the C++ file
 * headers: Tzu-Ray Shan (Sandia) and the COMB3 authors. Copyright (2003) Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';
import { powint } from '../comb/params';
import type { Comb3Lib, Comb3Param } from './params';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

const MY_PI = Math.PI;
const MY_PI2 = Math.PI / 2;

const dot3 = (a: Float64Array, b: Float64Array): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Result of comb_gijk_d: the angular derivative and the coordination-derivative coefficient. */
export interface GijkD { gijkD: number; com3jk: number }

/**
 * The short-range kernel of pair_style comb3. `lib` holds the library splines (null when no carbon is in the
 * potential, as in the C++ read_lib). `xcc`, `xch`, `xco` are the per-atom partial coordinations of the
 * owned and ghost atoms (xcctmp, xchtmp, xcotmp), set by the pair style before the terms are evaluated.
 */
export class Comb3Kernel {
  lib: Comb3Lib | null = null;
  maxx = 0; maxy = 0; maxz = 0;
  xcc = new Float64Array(0);
  xch = new Float64Array(0);
  xco = new Float64Array(0);
  private readonly dcosdri = new Float64Array(3);
  private readonly dcosdrj = new Float64Array(3);
  private readonly dcosdrk = new Float64Array(3);
  private readonly rijHat = new Float64Array(3);
  private readonly rikHat = new Float64Array(3);
  /** Results of coord (pcorn ... dxcoij). */
  private readonly cd = new Float64Array(5);

  private requireLib(): Comb3Lib {
    if (!this.lib) throw new StyleError('pair_style comb3: the coordination and angle splines need the library file lib.comb3, which is read only when an element of pair_coeff is C');
    return this.lib;
  }

  /** comb_fc: cosine switch between bigr - bigd and bigr + bigd (note: cos, as in the C++). */
  fc(r: number, p: Comb3Param): number {
    const rInn = p.bigr - p.bigd;
    const rOut = p.bigr + p.bigd;
    if (r <= rInn) return 1.0;
    if (r >= rOut) return 0.0;
    return 0.5 * (1.0 + libmCos(MY_PI * (r - rInn) / (rOut - rInn)));
  }

  fcD(r: number, p: Comb3Param): number {
    const rInn = p.bigr - p.bigd;
    const rOut = p.bigr + p.bigd;
    if (r <= rInn) return 0.0;
    if (r >= rOut) return 0.0;
    return -MY_PI2 / (rOut - rInn) * libmSin(MY_PI * (r - rInn) / (rOut - rInn));
  }

  /** Coordination switches with the cut-offs of lib.comb3 (ccutoff[0..5]): comb_fccc, comb_fcch, comb_fccch. */
  private switchCut(x: number, c1: number, c2: number): number {
    if (x <= c1) return 1.0;
    if (x >= c2) return 0.0;
    return 0.5 * (1.0 + libmCos(MY_PI * (x - c1) / (c2 - c1)));
  }

  private switchCutD(x: number, c1: number, c2: number): number {
    if (x <= c1) return 0.0;
    if (x >= c2) return 0.0;
    return -MY_PI2 / (c2 - c1) * libmSin(MY_PI * (x - c1) / (c2 - c1));
  }

  fccc(x: number): number { const c = this.requireLib().ccutoff; return this.switchCut(x, c[0], c[1]); }
  fcccD(x: number): number { const c = this.requireLib().ccutoff; return this.switchCutD(x, c[0], c[1]); }
  fccch(x: number): number { const c = this.requireLib().ccutoff; return this.switchCut(x, c[4], c[5]); }
  fccchD(x: number): number { const c = this.requireLib().ccutoff; return this.switchCutD(x, c[4], c[5]); }

  /** comb_fcch / comb_fcch_d: the radical coordination switch (ccutoff[2], ccutoff[3]). */
  fcch(x: number): number { const c = this.requireLib().ccutoff; return this.switchCut(x, c[2], c[3]); }
  fcchD(x: number): number { const c = this.requireLib().ccutoff; return this.switchCutD(x, c[2], c[3]); }

  /** comb_gijk: angular function (ang_flag 0: polynomial; 1: spline blend; 2: chain blend), times pcross. */
  gijk(costheta: number, p: Comb3Param, ncoTmp: number): number {
    const rmu1 = costheta;
    const rmu2 = rmu1 * rmu1, rmu3 = rmu2 * rmu1, rmu4 = rmu3 * rmu1, rmu5 = rmu4 * rmu1, rmu6 = rmu5 * rmu1;
    const co6 = p.pcos6 * rmu6, co5 = p.pcos5 * rmu5, co4 = p.pcos4 * rmu4, co3 = p.pcos3 * rmu3;
    const co2 = p.pcos2 * rmu2, co1 = p.pcos1 * rmu1, co0 = p.pcos0;
    const pcross = p.pcross;
    if (p.ang_flag === 1) {
      const lib = this.requireLib();
      const qtheta = this.fccc(ncoTmp);
      const astep = 2.0 / lib.ntab;
      const rrmu = (rmu1 + 1.0) / astep;
      const k = Math.trunc(rrmu);
      const gmu1 = co6 + co5 + co4 + co3 + co2 + co1 + co0;
      const gmu2 = lib.pang[k] + (lib.pang[k + 1] - lib.pang[k]) * (rrmu - k);
      const gmu = gmu2 + qtheta * (gmu1 - gmu2);
      return gmu * pcross;
    } else if (p.ang_flag === 2) {
      const lib = this.requireLib();
      const a = lib.chA;
      const ch6 = a[6] * rmu6, ch5 = a[5] * rmu5, ch4 = a[4] * rmu4, ch3 = a[3] * rmu3;
      const ch2 = a[2] * rmu2, ch1 = a[1] * rmu1, ch0 = a[0];
      const qtheta = this.fccch(ncoTmp);
      const gmu1 = co6 + co5 + co4 + co3 + co2 + co1 + co0;
      const gmu2 = ch6 + ch5 + ch4 + ch3 + ch2 + ch1 + ch0;
      const gmu = gmu2 + qtheta * (gmu1 - gmu2);
      return gmu * pcross;
    }
    const gmu = co6 + co5 + co4 + co3 + co2 + co1 + co0;
    return gmu * pcross;
  }

  /** comb_gijk_d: derivative of gijk with respect to cos(theta), and the coordination-derivative coefficient. */
  gijkD(costheta: number, p: Comb3Param, ncoTmp: number): GijkD {
    const rmu1 = costheta;
    const rmu2 = rmu1 * rmu1, rmu3 = rmu2 * rmu1, rmu4 = rmu3 * rmu1, rmu5 = rmu4 * rmu1, rmu6 = rmu5 * rmu1;
    const co6 = p.pcos6, co5 = p.pcos5, co4 = p.pcos4, co3 = p.pcos3, co2 = p.pcos2, co1 = p.pcos1, co0 = p.pcos0;
    const pcross = p.pcross;
    if (p.ang_flag === 1) {
      const lib = this.requireLib();
      const qtheta = this.fccc(ncoTmp);
      const dqtheta = this.fcccD(ncoTmp);
      const astep = 2.0 / lib.ntab;
      const rrmu = (rmu1 + 1.0) / astep;
      const k = Math.trunc(rrmu);
      const gmu1 = rmu6 * co6 + rmu5 * co5 + rmu4 * co4 + rmu3 * co3 + rmu2 * co2 + rmu1 * co1 + co0;
      const dgmu1 = 6.0 * rmu5 * co6 + 5.0 * rmu4 * co5 + 4.0 * rmu3 * co4 + 3.0 * rmu2 * co3 + 2.0 * rmu1 * co2 + co1;
      const gmu2 = lib.pang[k] + (lib.pang[k + 1] - lib.pang[k]) * (rrmu - k);
      const dgmu2 = lib.dpang[k] + (lib.dpang[k + 1] - lib.dpang[k]) * (rrmu - k);
      return { gijkD: pcross * (dgmu2 + qtheta * (dgmu1 - dgmu2)), com3jk: dqtheta * (gmu1 - gmu2) };
    } else if (p.ang_flag === 2) {
      const lib = this.requireLib();
      const a = lib.chA;
      const qtheta = this.fccch(ncoTmp);
      const dqtheta = this.fccchD(ncoTmp);
      const gmu1 = rmu6 * co6 + rmu5 * co5 + rmu4 * co4 + rmu3 * co3 + rmu2 * co2 + rmu1 * co1 + co0;
      const dgmu1 = 6.0 * rmu5 * co6 + 5.0 * rmu4 * co5 + 4.0 * rmu3 * co4 + 3.0 * rmu2 * co3 + 2.0 * rmu1 * co2 + co1;
      const gmu2 = rmu6 * a[6] + rmu5 * a[5] + rmu4 * a[4] + rmu3 * a[3] + rmu2 * a[2] + rmu1 * a[1] + a[0];
      const dgmu2 = 6.0 * rmu5 * a[6] + 5.0 * rmu4 * a[5] + 4.0 * rmu3 * a[4] + 3.0 * rmu2 * a[3] + 2.0 * rmu1 * a[2] + a[1];
      return { gijkD: pcross * (dgmu2 + qtheta * (dgmu1 - dgmu2)), com3jk: dqtheta * (gmu1 - gmu2) };
    }
    return {
      gijkD: pcross * (6.0 * rmu5 * co6 + 5.0 * rmu4 * co5 + 4.0 * rmu3 * co4 + 3.0 * rmu2 * co3 + 2.0 * rmu1 * co2 + co1),
      com3jk: 0.0,
    };
  }

  /** self: self energy of one ion with charge qi (chi, dj, dk, dl, qmin, qmax). */
  self(p: Comb3Param, qi: number): number {
    const s1 = p.chi, s2 = p.dj, s3 = p.dk, s4 = p.dl;
    const qmin = p.qmin, qmax = p.qmax;
    const cmin = 100.0, cmax = 100.0;
    let selfTmp = qi * (s1 + qi * (s2 + qi * (s3 + qi * s4)));
    if (qi < qmin) selfTmp += cmin * powint(qi - qmin, 4);
    if (qi > qmax) selfTmp += cmax * powint(qi - qmax, 4);
    return selfTmp;
  }

  /** comb_fa: attractive (bond-order-free) term; writes att_eng and att_force. */
  fa(r: number, pi: Comb3Param, pj: Comb3Param, qi: number, qj: number, out: { eng: number; force: number }): void {
    const alfij1 = pi.alpha1, alfij2 = pi.alpha2, alfij3 = pi.alpha3;
    const pbij1 = pi.bigB1, pbij2 = pi.bigB2, pbij3 = pi.bigB3;
    const Di = pi.DU + Math.pow(Math.abs(pi.bD * (pi.QU - qi)), pi.nD);
    const Dj = pj.DU + Math.pow(Math.abs(pj.bD * (pj.QU - qj)), pj.nD);
    const YYBn = pi.aB - Math.abs(powint(pi.bB * (qi - pi.Qo), 10));
    const YYBj = pj.aB - Math.abs(powint(pj.bB * (qj - pj.Qo), 10));
    if (YYBn * YYBj > 0.0) {
      const AlfDiAlfDj = Math.exp(0.5 * (pi.alfi * Di + pj.alfi * Dj));
      const sq = Math.sqrt(YYBn * YYBj);
      const Bsi = (pbij1 * Math.exp(-alfij1 * r) + pbij2 * Math.exp(-alfij2 * r) + pbij3 * Math.exp(-alfij3 * r)) * sq * AlfDiAlfDj;
      out.eng = -Bsi * this.fc(r, pi);
      out.force = -(Bsi * this.fcD(r, pi) - this.fc(r, pi) * sq * AlfDiAlfDj *
        (alfij1 * pbij1 * Math.exp(-alfij1 * r) + alfij2 * pbij2 * Math.exp(-alfij2 * r) + alfij3 * pbij3 * Math.exp(-alfij3 * r)));
    } else {
      out.eng = 0.0;
      out.force = 0.0;
    }
  }

  /**
   * comb_bij_d: bond order b_ij = (1 + zeta^n + pcorn)^(-1/2n) with its derivatives. Writes tbij and the five
   * prefactor derivatives into `o` (o[0] = tbij, o[1] = tbij1, o[2..5] = tbij2..5).
   */
  bijD(zet: number, p: Comb3Param, r: number, i: number, xcn: number, o: Float64Array): void {
    this.coord(p, r, i, xcn);
    const [pcorn, dpcorn, dxccij, dxchij, dxcoij] = this.cd;
    const zetang = zet;
    const powN = p.powern;
    let zeta = Math.pow(zetang, powN) + pcorn;
    const tmpTbij = powN * Math.pow(zetang, powN - 1.0);
    let tbij: number, tbij1: number;
    if ((1.0 + zeta) < 0.1) {
      zeta = 0.1 - 1.0;
      tbij = Math.pow(1.0 + zeta, -0.5 / powN);
      tbij1 = 0.0;
    } else if (zeta > p.c1) {
      tbij = Math.pow(zeta, -0.5 / powN);
      tbij1 = -0.5 / powN * Math.pow(zeta, (-0.5 / powN - 1.0));
    } else if (zeta > p.c2) {
      tbij = Math.pow(zeta, -0.5 / powN) - 0.5 / powN * Math.pow(zeta, (-0.5 / powN - 1.0));
      tbij1 = -0.5 / powN / zeta;
    } else if (Math.abs(zeta) < p.c4) {
      tbij = 1.0;
      tbij1 = 0.0;
    } else if (Math.abs(zeta) < p.c3) {
      tbij = 1.0 - zeta / (2.0 * powN);
      tbij1 = -1 / (2.0 * powN);
    } else {
      tbij = Math.pow(1.0 + zeta, -0.5 / powN);
      tbij1 = -0.5 / powN * Math.pow(1.0 + zeta, (-1.0 - 0.5 / powN));
    }
    o[0] = tbij;
    o[2] = tbij1 * dpcorn;
    o[3] = tbij1 * dxccij;
    o[4] = tbij1 * dxchij;
    o[5] = tbij1 * dxcoij;
    o[1] = tbij1 * tmpTbij;
  }

  /**
   * coord: the coordination correction pcorn of atom i (and its derivatives with respect to the partial
   * coordinations), written into this.cd = [pcorn, dpcorn, dxccij, dxchij, dxcoij].
   */
  coord(p: Comb3Param, r: number, i: number, xcn: number): void {
    const cd = this.cd;
    const triFlag = p.pcn_flag;
    const jeleGp = p.jelementgp;
    const pan = p.pcna, pbn = p.pcnb, pcn = p.pcnc, pdn = p.pcnd;
    let xcccn = this.xcc[i], xchcn = this.xch[i], xcocn = this.xco[i];
    const fcr = this.fc(r, p);
    let xcntot = -fcr * p.pcross + xcn;
    cd.fill(0);
    if (xcntot < 0.0) xcntot = 0.0;
    if (triFlag > 0) {
      const lib = this.requireLib();
      if (jeleGp === 1) xcccn = xcccn - fcr * p.pcross;
      if (jeleGp === 2) xchcn = xchcn - fcr * p.pcross;
      if (jeleGp === 3) xcocn = xcocn - fcr * p.pcross;
      if (xcccn < 0.0) xcccn = 0.0;
      if (xchcn < 0.0) xchcn = 0.0;
      if (xcocn < 0.0) xcocn = 0.0;
      if (xcccn > this.maxx) xcccn = this.maxx;
      if (xchcn > this.maxy) xchcn = this.maxy;
      if (xcocn > this.maxz) xcocn = this.maxz;
      const xcntritot = xcccn + xchcn + xcocn;
      const t = triFlag - 1;
      if (xcntritot > lib.maxxcn[t]) {
        cd[0] = lib.vmaxxcn[t] + (xcntot - lib.maxxcn[t]) * lib.dvmaxxcn[t];
        cd[2] = cd[3] = cd[4] = lib.dvmaxxcn[t];
      } else {
        const ixmin = Math.trunc(xcccn + 1.0e-12);
        const iymin = Math.trunc(xchcn + 1.0e-12);
        const izmin = Math.trunc(xcocn + 1.0e-12);
        if (Math.abs(ixmin - xcccn) > 1.0e-8 || Math.abs(iymin - xchcn) > 1.0e-8 || Math.abs(izmin - xcocn) > 1.0e-8) {
          this.cntriInt(triFlag, xcccn, xchcn, xcocn, ixmin, iymin, izmin);
        } else {
          const at = ((t * 5 + ixmin) * 5 + iymin) * 5 + izmin;
          cd[0] = lib.pcnGrid[at];
          cd[2] = lib.pcnGridx[at];
          cd[3] = lib.pcnGridy[at];
          cd[4] = lib.pcnGridz[at];
        }
      }
    } else {
      cd[0] = pan * xcntot + pbn * Math.exp(pcn * xcntot) + pdn;
      cd[1] = pan + pbn * pcn * Math.exp(pcn * xcntot);
    }
  }

  /** cntri_int: tricubic interpolation of the coordination spline (pcn_cubs); writes into this.cd. */
  private cntriInt(triFlag: number, xval: number, yval: number, zval: number, ixmin0: number, iymin0: number, izmin0: number): void {
    const lib = this.requireLib();
    let ixmin = ixmin0, iymin = iymin0, izmin = izmin0;
    if (ixmin >= this.maxx - 1) ixmin = this.maxx - 1;
    if (iymin >= this.maxy - 1) iymin = this.maxy - 1;
    if (izmin >= this.maxz - 1) izmin = this.maxz - 1;
    const cd = this.cd;
    cd[0] = 0.0; cd[2] = 0.0; cd[3] = 0.0; cd[4] = 0.0;
    const base = ((((triFlag - 1) * 4 + ixmin) * 4 + iymin) * 4 + izmin) * 64;
    for (let j = 0; j < 64; j++) {
      const e0 = Math.trunc(j / 16), e1 = Math.trunc(j / 4) % 4, e2 = j % 4;
      const x = lib.pcnCubs[base + j] * Math.pow(xval, e0) * Math.pow(yval, e1) * Math.pow(zval, e2);
      cd[0] += x;
      if (xval > 1.0e-8) cd[2] += x * e0 / xval;
      if (yval > 1.0e-8) cd[3] += x * e1 / yval;
      if (zval > 1.0e-8) cd[4] += x * e2 / zval;
    }
  }

  /** zeta: bond-order angular and exponential term of the bond i-j with the neighbor k (pair_comb3.cpp zeta). */
  zeta(pij: Comb3Param, pik: Comb3Param, rsqij: number, rsqik: number, delrij: Float64Array, delrik: Float64Array, xcn: number): number {
    const rij = Math.sqrt(rsqij);
    if (rij > pij.bigr + pij.bigd) return 0.0;
    const rik = Math.sqrt(rsqik);
    const costheta = dot3(delrij, delrik) / (rij * rik);
    const rlm3 = pij.beta;
    const arg = powint(rlm3 * (rij - rik), pij.powermint);
    let exDelr: number;
    if (arg > 69.0776) exDelr = 1.e30;
    else if (arg < -69.0776) exDelr = 0.0;
    else exDelr = Math.exp(arg);
    return this.fc(rik, pik) * this.gijk(costheta, pij, xcn) * exDelr;
  }

  /** repulsive: pure two-body repulsion (with the additional repulsion of addrep); returns fforce (force / r) and eng. */
  repulsive(pi: Comb3Param, pj: Comb3Param, rsq: number, iq: number, jq: number, out: { fforce: number; eng: number }): void {
    const romi = pi.addrep;
    const rrcs = pi.bigr + pi.bigd;
    const addr = pi.addrepr;
    const r = Math.sqrt(rsq);
    if (r > rrcs) return;
    const tmpFc = this.fc(r, pi);
    const tmpFcD = this.fcD(r, pi);
    const Di = pi.DU + Math.pow(Math.abs(pi.bD * (pi.QU - iq)), pi.nD);
    const Dj = pj.DU + Math.pow(Math.abs(pj.bD * (pj.QU - jq)), pj.nD);
    const bigA = pi.bigA;
    const rlm1 = pi.lambda;
    const fcdA = tmpFcD - tmpFc * rlm1;
    const LamDiLamDj = Math.exp(0.5 * (pi.lami * Di + pj.lami * Dj) - rlm1 * r);
    const caj = bigA * LamDiLamDj;
    let fforce = -caj * fcdA;
    let vrcs = 1.0, fvrcs = 0.0;
    if (romi !== 0.0 && r < addr) {
      vrcs += romi * (1.0 - r / addr) * (1.0 - r / addr);
      fvrcs = romi * 2.0 * (r / addr - 1.0) / addr;
      fforce = fforce * vrcs - caj * tmpFc * vrcs * fvrcs;
    }
    fforce /= r;
    out.fforce = fforce;
    out.eng = caj * tmpFc * vrcs;
  }

  /** selfp6p: two-body Legendre-polynomial (p6p) energy; adds its force/r to `force` and returns the energy. */
  selfp6p(pi: Comb3Param, pj: Comb3Param, rsq: number, out: { eng: number; force: number }): void {
    const r = Math.sqrt(rsq);
    const fcj = this.fc(r, pi);
    const fcjD = this.fcD(r, pi);
    const comtti = pi.p6p0 + pi.p6p1 + pi.p6p2 + pi.p6p3 + pi.p6p4 + pi.p6p5 + pi.p6p6;
    const comttj = pj.p6p0 + pj.p6p1 + pj.p6p2 + pj.p6p3 + pj.p6p4 + pj.p6p5 + pj.p6p6;
    out.eng = 0.5 * fcj * (comtti + comttj);
    out.force += 0.5 * fcjD * (comtti + comttj) / r;
  }

  /** ep6p: three-body Legendre-polynomial energy of the angle j-i-k. */
  ep6p(pj: Comb3Param, pk: Comb3Param, rsqij: number, rsqik: number, delrij: Float64Array, delrik: Float64Array): number {
    const rij = Math.sqrt(rsqij);
    const rik = Math.sqrt(rsqik);
    const rmu = dot3(delrij, delrik) / (rij * rik);
    const fcj = this.fc(rij, pj);
    const fck = this.fc(rik, pk);
    const rmu2 = rmu * rmu, rmu3 = rmu2 * rmu, rmu4 = rmu3 * rmu, rmu5 = rmu4 * rmu, rmu6 = rmu5 * rmu;
    const comtt = pj.p6p0 + pj.p6p1 * rmu + pj.p6p2 * rmu2 + pj.p6p3 * rmu3 + pj.p6p4 * rmu4 + pj.p6p5 * rmu5 + pj.p6p6 * rmu6;
    return 0.5 * fck * comtt * fcj;
  }

  /** fp6p: forces of ep6p on the three atoms (drilp: i, drjlp: j, drklp: k), written into the given vectors. */
  fp6p(pij: Comb3Param, pik: Comb3Param, rsqij: number, rsqik: number, delrij: Float64Array, delrik: Float64Array,
    drilp: Float64Array, drjlp: Float64Array, drklp: Float64Array): void {
    const rij = Math.sqrt(rsqij), rik = Math.sqrt(rsqik);
    const rmu = dot3(delrij, delrik) / (rij * rik);
    const fcj = this.fc(rij, pij), fck = this.fc(rik, pik);
    const fcjD = this.fcD(rij, pij), fckD = this.fcD(rik, pik);
    const rmu2 = rmu * rmu, rmu3 = rmu2 * rmu, rmu4 = rmu3 * rmu, rmu5 = rmu4 * rmu, rmu6 = rmu5 * rmu;
    const pp = pij;
    const lp0 = pp.p6p0, lp1 = pp.p6p1 * rmu, lp2 = pp.p6p2 * rmu2, lp3 = pp.p6p3 * rmu3;
    const lp4 = pp.p6p4 * rmu4, lp5 = pp.p6p5 * rmu5, lp6 = pp.p6p6 * rmu6;
    const lp1D = pp.p6p1, lp2D = pp.p6p2 * 2.0 * rmu, lp3D = pp.p6p3 * 3.0 * rmu2;
    const lp4D = pp.p6p4 * 4.0 * rmu3, lp5D = pp.p6p5 * 5.0 * rmu4, lp6D = pp.p6p6 * 6.0 * rmu5;
    const comtt = lp0 + lp1 + lp2 + lp3 + lp4 + lp5 + lp6;
    const comttD = lp1D + lp2D + lp3D + lp4D + lp5D + lp6D;
    const com4k = fcj * fckD * comtt;
    const com5 = fcj * fck * comttD;
    const com5k = fck * comtt * fcjD;
    const ffj1 = 0.5 * (-com5 / (rij * rik));
    const ffj2 = 0.5 * (com5 * rmu / rsqij - com5k / rij);
    const ffk1 = ffj1;
    const ffk2 = 0.5 * (-com4k / rik + com5 * rmu / rsqik);
    for (let c = 0; c < 3; c++) {
      drjlp[c] = ffj2 * delrij[c] + ffj1 * delrik[c];
      drklp[c] = ffk2 * delrik[c] + ffk1 * delrij[c];
    }
    for (let c = 0; c < 3; c++) drilp[c] = -(drjlp[c] + drklp[c]);
  }

  /** attractive: the three-body force of the bond-order term (comb_zetaterm_d with unit vectors). */
  attractive(pi: Comb3Param, pj: Comb3Param, pk: Comb3Param, pref: Float64Array,
    rsqij: number, rsqik: number, delrij: Float64Array, delrik: Float64Array,
    fi: Float64Array, fj: Float64Array, fk: Float64Array, xcn: number): void {
    const rijinv = 1.0 / Math.sqrt(rsqij);
    const rikinv = 1.0 / Math.sqrt(rsqik);
    const rij = Math.sqrt(rsqij), rik = Math.sqrt(rsqik);
    const rijHat = this.rijHat, rikHat = this.rikHat;
    for (let c = 0; c < 3; c++) {
      rijHat[c] = rijinv * delrij[c];
      rikHat[c] = rikinv * delrik[c];
    }
    this.zetaTermD(pref, rijHat, rij, rikHat, rik, fi, fj, fk, pi, pj, pk, xcn);
  }

  /** comb_zetaterm_d: forces of the bond-order term on atoms i, j, k (dri, drj, drk), from the five prefactors. */
  private zetaTermD(pref: Float64Array, rijHat: Float64Array, rij: number, rikHat: Float64Array, rik: number,
    dri: Float64Array, drj: Float64Array, drk: Float64Array,
    pi: Comb3Param, pj: Comb3Param, pk: Comb3Param, xcn: number): void {
    const mint = pi.powermint;
    const pcrossi = pi.pcross, pcrossj = pj.pcross, pcrossk = pk.pcross;
    const icontrol = pi.pcn_flag;
    const dfcI = this.fcD(rij, pi);
    const fcK = this.fc(rik, pk);
    const dfcK = this.fcD(rik, pk);
    const rlm3 = pi.beta;
    const tmp = powint(rlm3 * (rij - rik), mint);
    let exDelr: number;
    if (tmp > 69.0776) exDelr = 1.e30;
    else if (tmp < -69.0776) exDelr = 0.0;
    else exDelr = Math.exp(tmp);
    exDelr *= pcrossi;
    const cosTheta = dot3(rijHat, rikHat);
    const gijk = this.gijk(cosTheta, pi, xcn);
    const { gijkD, com3jk } = this.gijkD(cosTheta, pi, xcn);
    this.costhetaD(rijHat, rij, rikHat, rik, this.dcosdri, this.dcosdrj, this.dcosdrk);
    const dcosdri = this.dcosdri, dcosdrj = this.dcosdrj, dcosdrk = this.dcosdrk;
    let com6: number;
    if (icontrol > 0) {
      if (pi.kelementgp === 1) com6 = pref[2] * pcrossk * dfcK;
      else if (pi.kelementgp === 2) com6 = pref[3] * pcrossk * dfcK;
      else com6 = pref[4] * pcrossk * dfcK;
      // the radical and torsion derivative terms are carried by prefac_ij2 (pradx)
      if (pi.rad_flag >= 1 || pi.tor_flag !== 0) com6 += pref[1] * pcrossk * dfcK;
    } else {
      com6 = pref[1] * pcrossi * dfcK;
    }
    let com3j: number, com3k: number;
    if (pi.ang_flag === 1 || pi.ang_flag === 2) {
      com3j = com3jk * exDelr * pcrossk * pcrossj * fcK * dfcI;
      com3k = com3jk * exDelr * pcrossk * pcrossk * fcK * dfcK;
    } else {
      com3j = 0.0;
      com3k = 0.0;
    }
    const exDelrD = mint * powint(rlm3, mint) * powint(rij - rik, mint - 1) * exDelr;
    const p1 = pref[0];
    const c1 = -dfcK * gijk * exDelr;
    const c2 = fcK * gijkD * exDelr;
    const c3p = fcK * gijk * exDelrD;
    const c3m = -fcK * gijk * exDelrD;
    for (let c = 0; c < 3; c++) {
      let v = c1 * rikHat[c];
      v += c2 * dcosdri[c];
      v += c3p * rikHat[c];
      v += c3m * rijHat[c];
      v += -com3k * rikHat[c];
      v += -com3j * rijHat[c];
      dri[c] = p1 * v - com6 * rikHat[c];
    }
    const d2 = fcK * gijkD * exDelr;
    const d3 = fcK * gijk * exDelrD;
    for (let c = 0; c < 3; c++) {
      let v = d2 * dcosdrj[c];
      v += d3 * rijHat[c];
      v += com3j * rijHat[c];
      drj[c] = p1 * v;
    }
    const e1 = dfcK * gijk * exDelr;
    const e2 = fcK * gijkD * exDelr;
    const e3 = -fcK * gijk * exDelrD;
    for (let c = 0; c < 3; c++) {
      let v = e1 * rikHat[c];
      v += e2 * dcosdrk[c];
      v += e3 * rikHat[c];
      v += com3k * rikHat[c];
      drk[c] = p1 * v + com6 * rikHat[c];
    }
  }

  /** costheta_d: derivatives of cos(theta) with respect to the positions of atoms i, j, k. */
  costhetaD(rijHat: Float64Array, rij: number, rikHat: Float64Array, rik: number,
    dri: Float64Array, drj: Float64Array, drk: Float64Array): void {
    const cosTheta = dot3(rijHat, rikHat);
    for (let c = 0; c < 3; c++) {
      drj[c] = (-cosTheta * rijHat[c] + rikHat[c]) * (1.0 / rij);
      drk[c] = (-cosTheta * rikHat[c] + rijHat[c]) * (1.0 / rik);
    }
    for (let c = 0; c < 3; c++) dri[c] = -(drj[c] + drk[c]);
  }

}
