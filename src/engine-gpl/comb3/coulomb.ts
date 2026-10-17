/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_comb3.cpp (PairComb3::direct, field, qfo_self, qfo_direct, qfo_field, qfo_short,
 * comb_fc_curl, comb_fc_curl_d) and src/MANYBODY/pair_comb3.h, at commit b332f28. Authors credited in the C++
 * file headers: Tzu-Ray Shan (Sandia) and the COMB3 authors. Copyright (2003) Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE). The polarization (dipole) terms are not
 * included (pair_style comb3 polar_off).
 */
import { powint } from '../comb/params';
import type { Comb3Param } from './params';
import type { Comb3Tables, TriPoint } from './tables';
import type { Comb3Kernel } from './terms';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

const MY_PI = Math.PI;
const MY_PI2 = Math.PI / 2;
const MY_PIS = 1.77245385090551602729; // math_const.h (sqrt(pi) rounded once; Math.sqrt(Math.PI) is one ulp below)

/** comb_fc_curl: switch of the curl (carbon-type) correction between curlcut1 and curlcut2. */
export const fcCurl = (rocn: number, p: Comb3Param): number => {
  const a = p.curlcut1, b = p.curlcut2;
  if (rocn <= a) return 1.0;
  if (rocn >= b) return 0.0;
  return 0.5 * (1.0 + libmCos(MY_PI * (rocn - a) / (b - a)));
};

/** comb_fc_curl_d: derivative of fcCurl. */
export const fcCurlD = (rocn: number, p: Comb3Param): number => {
  const a = p.curlcut1, b = p.curlcut2;
  if (rocn <= a) return 0.0;
  if (rocn >= b) return 0.0;
  return -MY_PI2 / (b - a) * libmSin(MY_PI * (rocn - a) / (b - a));
};

/** Coulomb (direct, field) and charge-force terms of pair_style comb3, with the tables of tables.ts. */
export class Comb3Charge {
  /** Set by the pair style before each compute (the tables and the conversion factor qqr2e). */
  tb: Comb3Tables | null = null;
  esucon = 0;

  constructor(private readonly kern: Comb3Kernel) {}

  private need(): Comb3Tables {
    if (!this.tb) throw new Error('pair_style comb3: Coulomb tables are not set');
    return this.tb;
  }

  /** Three-point sum of a table of one row stride (sr1..3 weights at rows mr1..3). */
  private sum(tp: TriPoint, tab: Float64Array, stride: number, col: number): number {
    return tp.sr1 * tab[tp.mr1 * stride + col] + tp.sr2 * tab[tp.mr2 * stride + col] + tp.sr3 * tab[tp.mr3 * stride + col];
  }

  /** erfc(alf r) (erpaw[.][0]) or its Gaussian partner (erpaw[.][1]) at the three points. */
  private erp(tp: TriPoint, which: 0 | 1): number {
    const e = this.need().erpaw;
    return tp.sr1 * e[2 * tp.mr1 + which] + tp.sr2 * e[2 * tp.mr2 + which] + tp.sr3 * e[2 * tp.mr3 + which];
  }

  /**
   * direct: 1/r Coulomb energy and force (force divided by r) of the pair, with the Slater overlap terms.
   * xcoi, xcoj: the partial coordinations of the two atoms (for the curl correction of group-2 elements).
   */
  direct(pi: Comb3Param, pj: Comb3Param, tp: TriPoint, rsq: number, iq: number, jq: number,
    fac11: number, fac11e: number, xcoi: number, xcoj: number): { pot: number; pd: number } {
    const tb = this.need();
    const nt = tb.nntypes;
    const kern = this.kern;
    const esucon = this.esucon;
    let curli = pi.curl;
    let curlj = pj.curl;
    const inti = pi.ielement, intj = pj.ielement;
    const inty = tb.intype[inti * tb.nel + intj];
    const curlij0 = pi.curl0, curlji0 = pj.curl0;
    const ielegp = pi.ielementgp, jelegp = pj.ielementgp;
    const r = Math.sqrt(rsq);
    const r3 = r * rsq;
    const alf = 0.20;
    const alfdpi = 2.0 * alf / MY_PIS;
    let icurl = 0, jcurl = 0, curlij1 = 0, curlji1 = 0;
    if (ielegp === 2 && curli > curlij0) { icurl = 1; curlij1 = curli; }
    if (jelegp === 2 && curlj > curlji0) { jcurl = 1; curlji1 = curlj; }
    let dcurlij = 0.0, dcurlji = 0.0;
    if (icurl === 1 || jcurl === 1) {
      const fcp1j = kern.fcD(r, pi);
      if (icurl === 1) {
        curli = curlij1 + (curlij0 - curlij1) * fcCurl(xcoi, pi);
        dcurlij = fcp1j * (curlij0 - curlij1) * fcCurlD(xcoi, pi);
      }
      if (jcurl === 1) {
        curlj = curlji1 + (curlji0 - curlji1) * fcCurl(xcoj, pj);
        dcurlji = fcp1j * (curlji0 - curlji1) * fcCurlD(xcoj, pj);
      }
    }
    const erfcc = this.erp(tp, 0);
    const erfcd = this.erp(tp, 1);
    const afbn = this.sum(tp, tb.afb, nt, inti);
    const afbj = this.sum(tp, tb.afb, nt, intj);
    const fafbnl = this.sum(tp, tb.fafb, nt, inty);
    const dafbn = this.sum(tp, tb.dafb, nt, inti);
    const dafbj = this.sum(tp, tb.dafb, nt, intj);
    const dfafbnl = this.sum(tp, tb.dfafb, nt, inty);
    // energy
    const potij = erfcc / r * esucon - fac11e;
    const sme1n = iq * curlj * (afbn - fafbnl) * esucon;
    const sme1j = jq * curli * (afbj - fafbnl) * esucon;
    const sme2 = (potij + fafbnl * esucon) * iq * jq;
    const pot = 1.0 * (sme1n + sme1j + sme2);
    // force / r
    const dvdrr = (erfcc / r3 + alfdpi * erfcd / rsq) * esucon - fac11;
    let smf1n = iq * curlj * (dafbn - dfafbnl) * esucon / r;
    let smf1j = jq * curli * (dafbj - dfafbnl) * esucon / r;
    if (jcurl === 1 && ielegp === 3 && dcurlji !== 0.0) smf1n += dcurlji * iq * (afbn - fafbnl) * esucon / r;
    if (icurl === 1 && jelegp === 3 && dcurlij !== 0.0) smf1j += dcurlij * jq * (afbj - fafbnl) * esucon / r;
    const smf2 = dvdrr + dfafbnl * esucon / r;
    const pd = 1.0 * iq * jq * smf2 + smf1n + smf1j;
    return { pot, pd };
  }

  /** field: the field correction to the self energy (energy and force / r, to be subtracted from the direct force). */
  field(pi: Comb3Param, pj: Comb3Param, rsq: number, iq: number, jq: number): { eng: number; f: number } {
    const r = Math.sqrt(rsq);
    const r3 = r * r * r, r4 = r3 * r, r5 = r4 * r;
    const rc = pi.lcut;
    const rc2 = rc * rc, rc3 = rc * rc * rc, rc4 = rc3 * rc, rc5 = rc4 * rc;
    const cmi1 = pi.cmn1, cmi2 = pi.cmn2, cmj1 = pj.cmn1, cmj2 = pj.cmn2;
    const pcmi1 = pi.pcmn1, pcmi2 = pi.pcmn2;
    const rf3i = r3 / (r3 * r3 + pcmi1 * pcmi1 * pcmi1);
    const rcf3i = rc3 / (rc3 * rc3 + pcmi1 * pcmi1 * pcmi1);
    const rf5i = r5 / (r5 * r5 + powint(pcmi2, 5));
    const rcf5i = rc5 / (rc5 * rc5 + powint(pcmi2, 5));
    const drf3i = 3 / r * rf3i - 6 * rsq * rf3i * rf3i;
    const drcf3i = 3 / rc * rcf3i - 6 * rc2 * rcf3i * rcf3i;
    const drf5i = 5 / r * rf5i - 10 * r4 * rf5i * rf5i;
    const drcf5i = 5 / rc * rcf5i - 10 * rc4 * rcf5i * rcf5i;
    const rf3 = rf3i - rcf3i - (r - rc) * drcf3i;
    const rf5 = rf5i - rcf5i - (r - rc) * drcf5i;
    const drf4 = drf3i - drcf3i;
    const drf6 = drf5i - drcf5i;
    const smpn = jq * (cmi1 * rf3 + jq * cmi2 * rf5);
    const smpl = iq * (cmj1 * rf3 + iq * cmj2 * rf5);
    const eng = 1.0 * (smpn + smpl);
    const rfx1 = jq * (cmi1 * drf4 + jq * cmi2 * drf6) / r;
    const rfx2 = iq * (cmj1 * drf4 + iq * cmj2 * drf6) / r;
    return { eng, f: -1.0 * (rfx1 + rfx2) };
  }

  /** qfo_self: derivative of the self energy with respect to the charge. */
  qfoSelf(p: Comb3Param, qi: number): number {
    const s1 = p.chi, s2 = p.dj, s3 = p.dk, s4 = p.dl;
    const qmin = p.qmin, qmax = p.qmax;
    const cmin = 100.0, cmax = 100.0;
    let selfD = s1 + qi * (2.0 * s2 + qi * (3.0 * s3 + qi * 4.0 * s4));
    if (qi < qmin) selfD += 4.0 * cmin * (qi - qmin) * (qi - qmin) * (qi - qmin);
    if (qi > qmax) selfD += 4.0 * cmax * (qi - qmax) * (qi - qmax) * (qi - qmax);
    return selfD;
  }

  /** qfo_direct: charge forces of the 1/r term (fqij for atom i, fqji for atom j). */
  qfoDirect(pi: Comb3Param, pj: Comb3Param, tp: TriPoint, rsq: number, fac11e: number, iq: number, jq: number,
    xcoi: number, xcoj: number): { fqij: number; fqji: number } {
    const tb = this.need();
    const nt = tb.nntypes;
    const esucon = this.esucon;
    let curli = pi.curl;
    let curlj = pj.curl;
    const inti = pi.ielement, intj = pj.ielement;
    const inty = tb.intype[inti * tb.nel + intj];
    const curlij0 = pi.curl0, curlji0 = pj.curl0;
    const ielegp = pi.ielementgp, jelegp = pj.ielementgp;
    const r = Math.sqrt(rsq);
    if (ielegp === 2 && curli > curlij0) curli = curli + (curlij0 - curli) * fcCurl(xcoi, pi);
    if (jelegp === 2 && curlj > curlji0) curlj = curlj + (curlji0 - curlj) * fcCurl(xcoj, pj);
    const erfcc = this.erp(tp, 0);
    const fafbnl = this.sum(tp, tb.fafb, nt, inty);
    const afbn = this.sum(tp, tb.afb, nt, inti);
    const afbj = this.sum(tp, tb.afb, nt, intj);
    const vm = erfcc / r * esucon - fac11e;
    const vmfafb = vm + esucon * fafbnl;
    const sme1n = curlj * (afbn - fafbnl) * esucon;
    const sme1j = curli * (afbj - fafbnl) * esucon;
    return { fqij: 1.0 * (jq * vmfafb + sme1n), fqji: 1.0 * (iq * vmfafb + sme1j) };
  }

  /** qfo_field: charge forces of the field correction. */
  qfoField(pi: Comb3Param, pj: Comb3Param, rsq: number, iq: number, jq: number): { fqij: number; fqji: number } {
    const r = Math.sqrt(rsq);
    const r3 = r * rsq;
    const r5 = r3 * rsq;
    const rc = pi.lcut;
    const rc2 = rc * rc, rc3 = rc * rc * rc, rc4 = rc3 * rc, rc5 = rc4 * rc;
    const cmi1 = pi.cmn1, cmi2 = pi.cmn2, cmj1 = pj.cmn1, cmj2 = pj.cmn2;
    const pcmi1 = pi.pcmn1, pcmi2 = pi.pcmn2;
    const rf3i = r3 / (r3 * r3 + pcmi1 * pcmi1 * pcmi1);
    const rcf3i = rc3 / (rc3 * rc3 + pcmi1 * pcmi1 * pcmi1);
    const rf5i = r5 / (r5 * r5 + powint(pcmi2, 5));
    const rcf5i = rc5 / (rc5 * rc5 + powint(pcmi2, 5));
    const drcf3i = 3 / rc * rcf3i - 6 * rc2 * rcf3i * rcf3i;
    const drcf5i = 5 / rc * rcf5i - 10 * rc4 * rcf5i * rcf5i;
    const rf3 = rf3i - rcf3i - (r - rc) * drcf3i;
    const rf5 = rf5i - rcf5i - (r - rc) * drcf5i;
    return { fqij: 1.0 * cmj1 * rf3 + 2.0 * iq * cmj2 * rf5, fqji: 1.0 * cmi1 * rf3 + 2.0 * jq * cmi2 * rf5 };
  }

  /**
   * qfo_short: charge forces of the bond-order (attractive) term; bij is the bond order of the pair from the
   * last compute (bbij[i][nj]). The charges are clamped to [QL - 0.2, QU + 0.2] as in the C++.
   */
  qfoShort(pi: Comb3Param, pj: Comb3Param, rsq: number, iq0: number, jq0: number, bij: number, fc: number):
    { fqij: number; fqji: number } {
    const rlm1 = pi.lambda;
    const alfij1 = pi.alpha1, alfij2 = pi.alpha2, alfij3 = pi.alpha3;
    const pbij1 = pi.bigB1, pbij2 = pi.bigB2, pbij3 = pi.bigB3;
    const r = Math.sqrt(rsq);
    const tmpFc = fc;
    const QUchi = (pi.QU - iq0) * pi.bD;
    const QUchj = (pj.QU - jq0) * pj.bD;
    const QOchi = (iq0 - pi.Qo) * pi.bB;
    const QOchj = (jq0 - pj.Qo) * pj.bB;
    let Di: number, dDi: number, Bsi: number, dBsi: number;
    if (iq0 < pi.QL - 0.2) {
      Di = pi.DL; dDi = 0.0; Bsi = 0.0; dBsi = 0.0;
    } else if (iq0 > pi.QU + 0.2) {
      Di = pi.DU; dDi = 0.0; Bsi = 0.0; dBsi = 0.0;
    } else {
      Di = pi.DU + libmPow(QUchi, pi.nD);
      dDi = -pi.nD * pi.bD * libmPow(QUchi, pi.nD - 1.0);
      Bsi = pi.aB - libmPow(QOchi, 10);
      dBsi = -pi.bB * 10.0 * powint(QOchi, 9);
    }
    let Dj: number, dDj: number, Bsj: number, dBsj: number;
    if (jq0 < pj.QL - 0.2) {
      Dj = pj.DL; dDj = 0.0; Bsj = 0.0; dBsj = 0.0;
    } else if (jq0 > pj.QU + 0.2) {
      Dj = pj.DU; dDj = 0.0; Bsj = 0.0; dBsj = 0.0;
    } else {
      Dj = pj.DU + libmPow(QUchj, pj.nD);
      dDj = -pj.nD * pj.bD * libmPow(QUchj, pj.nD - 1.0);
      Bsj = pj.aB - libmPow(QOchj, 10);
      dBsj = -pj.bB * 10.0 * powint(QOchj, 9);
    }
    const LamDiLamDj = libmExp(0.5 * (pi.lami * Di + pj.lami * Dj) - rlm1 * r);
    const caj = 0.5 * tmpFc * pi.bigA * LamDiLamDj;
    let cbqpn: number, cbqpj: number;
    if (Bsi * Bsj > 0.0) {
      const AlfDiAlfDj = libmExp(0.5 * (pi.alfi * Di + pj.alfi * Dj));
      const cbj = -0.5 * tmpFc * bij * Math.sqrt(Bsi * Bsj) * AlfDiAlfDj *
        (pbij1 * libmExp(-alfij1 * r) + pbij2 * libmExp(-alfij2 * r) + pbij3 * libmExp(-alfij3 * r));
      cbqpn = cbj * (pi.alfi * dDi + dBsi / Bsi);
      cbqpj = cbj * (pj.alfi * dDj + dBsj / Bsj);
    } else {
      cbqpn = 0.0; cbqpj = 0.0;
    }
    const caqpn = caj * pi.lami * dDi;
    const caqpj = caj * pj.lami * dDj;
    return { fqij: 1.0 * (caqpn + cbqpn), fqji: 1.0 * (caqpj + cbqpj) };
  }
}
