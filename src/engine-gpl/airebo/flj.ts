/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_airebo.cpp (PairAIREBO::FLJ, PairAIREBO::bondorderLJ) and
 * src/MANYBODY/pair_airebo_morse.cpp (the Morse branch of FLJ), at commit b332f28.
 * The force and energy expressions keep the C++ operation order (the LJ bond-order
 * forces of bondorderLJ differ in form from the REBO bondorder: they are reproduced as written).
 * Authors credited in the C++ file headers: Ase Henry (MIT); bugfixes and optimizations:
 * Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer (Temple U), Markus Hoehnerbach
 * (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M: Thomas C. O'Connor (JHU) 2014.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { NEIGHMASK } from '../../engine/neighbor';
import { sp, sp2, type Ref } from './splines';
import { TOL, kron, gSpline, piRCSpline, TijSpline, newRef, type ReboCore } from './rebo';
import { bondorderPijCross, pijEval } from './bc';
import { exp as libmExp } from '../libm/exp';
import { vTally2, vTally3, vTally4 } from './tally';

/** Parameters of the LJ / Morse term, precomputed in init_one (AIREBO side). */
export interface LJParams {
  /** cutljsq[ii*2+jj] = (cutlj * sigma)^2 */
  cutljsq: Float64Array;
  lj1: Float64Array;
  lj2: Float64Array;
  lj3: Float64Array;
  lj4: Float64Array;
  sigcut: number;
  sigmin: number;
  morseflag: boolean;
  cut3rebo: number;
}

/** REBO core plus the LJ term data and the LJ pair list of owned atoms. */
export interface LJCore extends ReboCore {
  ljp: LJParams;
  /**
   * LJ neighbor rows of owned atoms: row i is ljList[ljFirst[i] .. ljFirst[i] + ljNum[i]), entries masked with
   * NEIGHMASK (the full neighbor list, or the rows of a binned search); flj skips unmapped atoms and pairs beyond
   * cutljsq.
   */
  ljFirst: Int32Array;
  ljNum: Int32Array;
  ljList: Int32Array;
  /** Epsilon of the torsion term per element pair (ii*2+jj) (TORSION uses it). */
  epsilonT: Float64Array;
}

const v3 = (): Float64Array => new Float64Array(3);
const rdwij = newRef(), rdS = newRef();
const dN2 = new Float64Array(2);
const dN3 = new Float64Array(3);
const dN2PIJ = new Float64Array(2), dN2PJI = new Float64Array(2), dN3piRC = new Float64Array(3), dN3Tij = new Float64Array(3);
const rdgdc = newRef(), rdgdN = newRef();
const rwik = newRef(), rwjl = newRef(), rdw21 = newRef(), rdw34 = newRef(), rdNki = newRef(), rdNlj = newRef();
const rdwkn = newRef(), rdwln = newRef(), rdStb = newRef(), rdStr = newRef(), rdts = newRef(), rdtl = newRef();
const rik = v3(), rjl = v3(), rkn = v3(), rln = v3(), r32 = v3(), r23 = v3(), r21 = v3(), r34 = v3(), rjk = v3(), ril = v3();
const dcosjikdri = v3(), dcosijldri = v3(), dcosjikdrk = v3(), dcosjikdrj = v3(), dcosijldrj = v3(), dcosijldrl = v3();
const fi = v3(), fj = v3(), fk = v3(), fl = v3();
const rji = v3(), rki = v3(), rlj = v3(), r13 = v3(), r43 = v3();
const cross321 = v3(), cross234 = v3(), dt2dik = v3(), dt2djl = v3(), dt2dij = v3();
const F23 = v3(), F12 = v3(), F34 = v3(), F31 = v3(), F24 = v3(), f1 = v3(), f2 = v3(), f3 = v3(), f4 = v3();

/** f[a] += (x, y, z). */
const addF = (f: Float64Array, a: number, x: number, y: number, z: number): void => {
  f[3 * a] += x;
  f[3 * a + 1] += y;
  f[3 * a + 2] += z;
};

/**
 * PairAIREBO::bondorderLJ: the bond order of the LJ connectivity switch. Returns Stb (the
 * bond-order switch of the LJ term, Sp2 of b_ij between bLJmin and bLJmax); VA is the LJ
 * attraction scaled by its derivative; forces are added to f. rij is the vector x_i - x_j,
 * rijmag its length; rijmagMod is rcmin of the element pair (the C++ rijmag_mod).
 */
export const bondorderLJ = (
  core: LJCore, i: number, j: number, rijmagMod: number, VA0: number, rij: Float64Array, rijmag0: number,
): number => {
  const { P, nC, nH, x, f, mt, rstart, rlist } = core;
  const vatom = core.vatom;
  let VA = VA0;
  let rijmag = rijmag0;
  const atomi = i, atomj = j;
  const itype = mt[atomi], jtype = mt[atomj];
  const rmin = P.rcmin, rmax = P.rcmax, rmaxp = P.rcmaxp, rho = P.rho;
  const Nmin = P.Nmin, Nmax = P.Nmax;
  const pidx = (a: number, b: number) => a * 2 + b;

  const wij = sp(rijmag, rmin[pidx(itype, jtype)], rmax[pidx(itype, jtype)], rdwij);
  const NijC = nC[atomi] - (wij * kron(jtype, 0));
  const NijH = nH[atomi] - (wij * kron(jtype, 1));
  const NjiC = nC[atomj] - (wij * kron(itype, 0));
  const NjiH = nH[atomj] - (wij * kron(itype, 1));
  let tmp = 0.0, tmp2 = 0.0, tmp3 = 0.0;
  let dgdc = 0.0, dgdN = 0.0;
  let NconjtmpI = 0.0, NconjtmpJ = 0.0;
  let Etmp = 0.0;
  let Stb = 0.0, dStb = 0.0;

  for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
    const atomk = rlist[kk];
    if (atomk === atomj) continue;
    const ktype = mt[atomk];
    rik[0] = x[3 * atomi] - x[3 * atomk];
    rik[1] = x[3 * atomi + 1] - x[3 * atomk + 1];
    rik[2] = x[3 * atomi + 2] - x[3 * atomk + 2];
    const rikmag = Math.sqrt((rik[0] * rik[0]) + (rik[1] * rik[1]) + (rik[2] * rik[2]));
    const lamdajik = 4.0 * kron(itype, 1) * ((rho[ktype * 2 + 1] - rikmag) - (rho[jtype * 2 + 1] - rijmagMod));
    const wik = sp(rikmag, rmin[pidx(itype, ktype)], rmax[pidx(itype, ktype)], rdS);
    const Nki = nC[atomk] - (wik * kron(itype, 0)) + nH[atomk] - (wik * kron(itype, 1));
    let cosjik = ((rij[0] * rik[0]) + (rij[1] * rik[1]) + (rij[2] * rik[2])) / (rijmag * rikmag);
    cosjik = Math.min(cosjik, 1.0);
    cosjik = Math.max(cosjik, -1.0);
    const g = gSpline(core, cosjik, NijC + NijH, itype, rdgdc, rdgdN);
    dgdN = rdgdN.v;
    Etmp += (wik * g * libmExp(lamdajik));
    tmp3 += (wik * dgdN * libmExp(lamdajik));
    NconjtmpI = NconjtmpI + (kron(ktype, 0) * wik * sp(Nki, Nmin, Nmax, rdS));
  }
  dN2PIJ[0] = 0.0;
  dN2PIJ[1] = 0.0;
  const PijS = pijEval(core, NijC, NijH, NjiC, NjiH, itype, jtype, dN2PIJ);
  const pij = 1.0 / Math.sqrt(1.0 + Etmp + PijS);
  const tmppij = -0.5 * (pij * pij * pij);
  const tmp3pij = tmp3;
  tmp = 0.0;
  tmp2 = 0.0;
  tmp3 = 0.0;
  Etmp = 0.0;
  for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
    const atoml = rlist[ll];
    if (atoml === atomi) continue;
    const ltype = mt[atoml];
    rjl[0] = x[3 * atomj] - x[3 * atoml];
    rjl[1] = x[3 * atomj + 1] - x[3 * atoml + 1];
    rjl[2] = x[3 * atomj + 2] - x[3 * atoml + 2];
    const rjlmag = Math.sqrt((rjl[0] * rjl[0]) + (rjl[1] * rjl[1]) + (rjl[2] * rjl[2]));
    const lamdaijl = 4.0 * kron(jtype, 1) * ((rho[ltype * 2 + 1] - rjlmag) - (rho[itype * 2 + 1] - rijmagMod));
    const wjl = sp(rjlmag, rmin[pidx(jtype, ltype)], rmax[pidx(jtype, ltype)], rdS);
    const Nlj = nC[atoml] - (wjl * kron(jtype, 0)) + nH[atoml] - (wjl * kron(jtype, 1));
    let cosijl = -1.0 * ((rij[0] * rjl[0]) + (rij[1] * rjl[1]) + (rij[2] * rjl[2])) / (rijmag * rjlmag);
    cosijl = Math.min(cosijl, 1.0);
    cosijl = Math.max(cosijl, -1.0);
    const g = gSpline(core, cosijl, NjiC + NjiH, jtype, rdgdc, rdgdN);
    dgdN = rdgdN.v;
    Etmp += (wjl * g * libmExp(lamdaijl));
    tmp3 += (wjl * dgdN * libmExp(lamdaijl));
    NconjtmpJ = NconjtmpJ + (kron(ltype, 0) * wjl * sp(Nlj, Nmin, Nmax, rdS));
  }
  dN2PJI[0] = 0.0;
  dN2PJI[1] = 0.0;
  const PjiS = pijEval(core, NjiC, NjiH, NijC, NijH, jtype, itype, dN2PJI);
  const pji = 1.0 / Math.sqrt(1.0 + Etmp + PjiS);
  const tmppji = -0.5 * (pji * pji * pji);
  const tmp3pji = tmp3;
  const Nijconj = 1.0 + (NconjtmpI * NconjtmpI) + (NconjtmpJ * NconjtmpJ);
  const piRC = piRCSpline(core, NijC + NijH, NjiC + NjiH, Nijconj, itype, jtype, dN3piRC);
  let Tij = 0.0;
  dN3Tij[0] = 0.0;
  dN3Tij[1] = 0.0;
  dN3Tij[2] = 0.0;
  if (itype === 0 && jtype === 0) Tij = TijSpline(core, NijC + NijH, NjiC + NjiH, Nijconj, dN3Tij);
  Etmp = 0.0;
  if (Math.abs(Tij) > TOL) {
    const atom2 = atomi, atom3 = atomj;
    r32[0] = x[3 * atom3] - x[3 * atom2];
    r32[1] = x[3 * atom3 + 1] - x[3 * atom2 + 1];
    r32[2] = x[3 * atom3 + 2] - x[3 * atom2 + 2];
    const r32mag = Math.sqrt((r32[0] * r32[0]) + (r32[1] * r32[1]) + (r32[2] * r32[2]));
    r23[0] = -r32[0];
    r23[1] = -r32[1];
    r23[2] = -r32[2];
    const r23mag = r32mag;
    for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
      const atomk = rlist[kk];
      const atom1 = atomk;
      const ktype = mt[atomk];
      if (atomk === atomj) continue;
      r21[0] = x[3 * atom2] - x[3 * atom1];
      r21[1] = x[3 * atom2 + 1] - x[3 * atom1 + 1];
      r21[2] = x[3 * atom2 + 2] - x[3 * atom1 + 2];
      const r21mag = Math.sqrt(r21[0] * r21[0] + r21[1] * r21[1] + r21[2] * r21[2]);
      let cos321 = -1.0 * ((r21[0] * r32[0]) + (r21[1] * r32[1]) + (r21[2] * r32[2])) / (r21mag * r32mag);
      cos321 = Math.min(cos321, 1.0);
      cos321 = Math.max(cos321, -1.0);
      const sin321 = Math.sqrt(1.0 - cos321 * cos321);
      if ((sin321 > TOL) && (r21mag > TOL)) {
        const w21 = sp(r21mag, rmin[pidx(itype, ktype)], rmaxp[pidx(itype, ktype)], rdw21);
        const tspjik = sp2(cos321, P.thmin, P.thmax, rdts);
        for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
          const atoml = rlist[ll];
          const atom4 = atoml;
          const ltype = mt[atoml];
          if (atoml === atomi || atoml === atomk) continue;
          r34[0] = x[3 * atom3] - x[3 * atom4];
          r34[1] = x[3 * atom3 + 1] - x[3 * atom4 + 1];
          r34[2] = x[3 * atom3 + 2] - x[3 * atom4 + 2];
          const r34mag = Math.sqrt(r34[0] * r34[0] + r34[1] * r34[1] + r34[2] * r34[2]);
          let cos234 = (r32[0] * r34[0] + r32[1] * r34[1] + r32[2] * r34[2]) / (r32mag * r34mag);
          cos234 = Math.min(cos234, 1.0);
          cos234 = Math.max(cos234, -1.0);
          const sin234 = Math.sqrt(1.0 - cos234 * cos234);
          if ((sin234 > TOL) && (r34mag > TOL)) {
            const w34 = sp(r34mag, rmin[pidx(jtype, ltype)], rmaxp[pidx(jtype, ltype)], rdw34);
            const tspijl = sp2(cos234, P.thmin, P.thmax, rdtl);
            cross321[0] = (r32[1] * r21[2]) - (r32[2] * r21[1]);
            cross321[1] = (r32[2] * r21[0]) - (r32[0] * r21[2]);
            cross321[2] = (r32[0] * r21[1]) - (r32[1] * r21[0]);
            cross234[0] = (r23[1] * r34[2]) - (r23[2] * r34[1]);
            cross234[1] = (r23[2] * r34[0]) - (r23[0] * r34[2]);
            cross234[2] = (r23[0] * r34[1]) - (r23[1] * r34[0]);
            const cwnum = (cross321[0] * cross234[0]) + (cross321[1] * cross234[1]) + (cross321[2] * cross234[2]);
            const cwnom = r21mag * r34mag * r23mag * r23mag * sin321 * sin234;
            const om1234 = cwnum / cwnom;
            Etmp += ((1.0 - om1234 * om1234) * w21 * w34) * (1.0 - tspjik) * (1.0 - tspijl);
          }
        }
      }
    }
  }
  const bij = (0.5 * (pij + pji)) + piRC + (Tij * Etmp);
  Stb = sp2(bij, P.bLJmin[pidx(itype, jtype)], P.bLJmax[pidx(itype, jtype)], rdStb);
  dStb = rdStb.v;
  VA = VA * dStb;
  if (dStb !== 0.0) {
    tmp = tmppij;
    dN2[0] = dN2PIJ[0];
    dN2[1] = dN2PIJ[1];
    tmp3 = tmp3pij;
    // --- i side of P_ij
    for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
      const atomk = rlist[kk];
      const ktype = mt[atomk];
      if (atomk === atomj) continue;
      rik[0] = x[3 * atomi] - x[3 * atomk];
      rik[1] = x[3 * atomi + 1] - x[3 * atomk + 1];
      rik[2] = x[3 * atomi + 2] - x[3 * atomk + 2];
      const rikmag = Math.sqrt(rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2]);
      const lamdajik = 4.0 * kron(itype, 1) * ((rho[ktype * 2 + 1] - rikmag) - (rho[jtype * 2 + 1] - rijmagMod));
      const wik = sp(rikmag, rmin[pidx(itype, ktype)], rmax[pidx(itype, ktype)], rwik);
      const dwik = rwik.v;
      let cosjik = (rij[0] * rik[0] + rij[1] * rik[1] + rij[2] * rik[2]) / (rijmag * rikmag);
      cosjik = Math.min(cosjik, 1.0);
      cosjik = Math.max(cosjik, -1.0);
      dcosjikdri[0] = ((rij[0] + rik[0]) / (rijmag * rikmag)) - (cosjik * ((rij[0] / (rijmag * rijmag)) + (rik[0] / (rikmag * rikmag))));
      dcosjikdri[1] = ((rij[1] + rik[1]) / (rijmag * rikmag)) - (cosjik * ((rij[1] / (rijmag * rijmag)) + (rik[1] / (rikmag * rikmag))));
      dcosjikdri[2] = ((rij[2] + rik[2]) / (rijmag * rikmag)) - (cosjik * ((rij[2] / (rijmag * rijmag)) + (rik[2] / (rikmag * rikmag))));
      dcosjikdrk[0] = (-rij[0] / (rijmag * rikmag)) + (cosjik * (rik[0] / (rikmag * rikmag)));
      dcosjikdrk[1] = (-rij[1] / (rijmag * rikmag)) + (cosjik * (rik[1] / (rikmag * rikmag)));
      dcosjikdrk[2] = (-rij[2] / (rijmag * rikmag)) + (cosjik * (rik[2] / (rikmag * rikmag)));
      dcosjikdrj[0] = (-rik[0] / (rijmag * rikmag)) + (cosjik * (rij[0] / (rijmag * rijmag)));
      dcosjikdrj[1] = (-rik[1] / (rijmag * rikmag)) + (cosjik * (rij[1] / (rijmag * rijmag)));
      dcosjikdrj[2] = (-rik[2] / (rijmag * rikmag)) + (cosjik * (rij[2] / (rijmag * rijmag)));
      const g = gSpline(core, cosjik, NijC + NijH, itype, rdgdc, rdgdN);
      dgdc = rdgdc.v;
      tmp2 = VA * .5 * (tmp * wik * dgdc * libmExp(lamdajik));
      fj[0] = -tmp2 * dcosjikdrj[0];
      fj[1] = -tmp2 * dcosjikdrj[1];
      fj[2] = -tmp2 * dcosjikdrj[2];
      fi[0] = -tmp2 * dcosjikdri[0];
      fi[1] = -tmp2 * dcosjikdri[1];
      fi[2] = -tmp2 * dcosjikdri[2];
      fk[0] = -tmp2 * dcosjikdrk[0];
      fk[1] = -tmp2 * dcosjikdrk[1];
      fk[2] = -tmp2 * dcosjikdrk[2];
      tmp2 = VA * .5 * (tmp * wik * g * libmExp(lamdajik) * 4.0 * kron(itype, 1));
      fi[0] += tmp2 * (rik[0] / rikmag);
      fi[1] += tmp2 * (rik[1] / rikmag);
      fi[2] += tmp2 * (rik[2] / rikmag);
      fk[0] -= tmp2 * (rik[0] / rikmag);
      fk[1] -= tmp2 * (rik[1] / rikmag);
      fk[2] -= tmp2 * (rik[2] / rikmag);
      tmp2 = VA * .5 * (tmp * dwik * g * libmExp(lamdajik)) / rikmag;
      fi[0] -= tmp2 * rik[0];
      fi[1] -= tmp2 * rik[1];
      fi[2] -= tmp2 * rik[2];
      fk[0] += tmp2 * rik[0];
      fk[1] += tmp2 * rik[1];
      fk[2] += tmp2 * rik[2];
      tmp2 = VA * .5 * (tmp * dN2[ktype] * dwik) / rikmag;
      fi[0] -= tmp2 * rik[0];
      fi[1] -= tmp2 * rik[1];
      fi[2] -= tmp2 * rik[2];
      fk[0] += tmp2 * rik[0];
      fk[1] += tmp2 * rik[1];
      fk[2] += tmp2 * rik[2];
      tmp2 = VA * .5 * (tmp * tmp3 * dwik) / rikmag;
      fi[0] -= tmp2 * rik[0];
      fi[1] -= tmp2 * rik[1];
      fi[2] -= tmp2 * rik[2];
      fk[0] += tmp2 * rik[0];
      fk[1] += tmp2 * rik[1];
      fk[2] += tmp2 * rik[2];
      addF(f, atomi, fi[0], fi[1], fi[2]);
      addF(f, atomj, fj[0], fj[1], fj[2]);
      addF(f, atomk, fk[0], fk[1], fk[2]);
      if (vatom) {
        rji[0] = -rij[0]; rji[1] = -rij[1]; rji[2] = -rij[2];
        rki[0] = -rik[0]; rki[1] = -rik[1]; rki[2] = -rik[2];
        vTally3(vatom, atomi, atomj, atomk, fj, fk, rji, rki);
      }
    }
    // --- j side of P_ji
    tmp = tmppji;
    tmp3 = tmp3pji;
    dN2[0] = dN2PJI[0];
    dN2[1] = dN2PJI[1];
    for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
      const atoml = rlist[ll];
      if (atoml === atomi) continue;
      const ltype = mt[atoml];
      rjl[0] = x[3 * atomj] - x[3 * atoml];
      rjl[1] = x[3 * atomj + 1] - x[3 * atoml + 1];
      rjl[2] = x[3 * atomj + 2] - x[3 * atoml + 2];
      const rjlmag = Math.sqrt((rjl[0] * rjl[0]) + (rjl[1] * rjl[1]) + (rjl[2] * rjl[2]));
      const lamdaijl = 4.0 * kron(jtype, 1) * ((rho[ltype * 2 + 1] - rjlmag) - (rho[itype * 2 + 1] - rijmagMod));
      const wjl = sp(rjlmag, rmin[pidx(jtype, ltype)], rmax[pidx(jtype, ltype)], rwjl);
      const dwjl = rwjl.v;
      let cosijl = (-1.0 * ((rij[0] * rjl[0]) + (rij[1] * rjl[1]) + (rij[2] * rjl[2]))) / (rijmag * rjlmag);
      cosijl = Math.min(cosijl, 1.0);
      cosijl = Math.max(cosijl, -1.0);
      dcosijldri[0] = (-rjl[0] / (rijmag * rjlmag)) - (cosijl * rij[0] / (rijmag * rijmag));
      dcosijldri[1] = (-rjl[1] / (rijmag * rjlmag)) - (cosijl * rij[1] / (rijmag * rijmag));
      dcosijldri[2] = (-rjl[2] / (rijmag * rjlmag)) - (cosijl * rij[2] / (rijmag * rijmag));
      dcosijldrj[0] = ((-rij[0] + rjl[0]) / (rijmag * rjlmag)) + (cosijl * ((rij[0] / (rijmag * rijmag)) - (rjl[0] / (rjlmag * rjlmag))));
      dcosijldrj[1] = ((-rij[1] + rjl[1]) / (rijmag * rjlmag)) + (cosijl * ((rij[1] / (rijmag * rijmag)) - (rjl[1] / (rjlmag * rjlmag))));
      dcosijldrj[2] = ((-rij[2] + rjl[2]) / (rijmag * rjlmag)) + (cosijl * ((rij[2] / (rijmag * rijmag)) - (rjl[2] / (rjlmag * rjlmag))));
      dcosijldrl[0] = (rij[0] / (rijmag * rjlmag)) + (cosijl * rjl[0] / (rjlmag * rjlmag));
      dcosijldrl[1] = (rij[1] / (rijmag * rjlmag)) + (cosijl * rjl[1] / (rjlmag * rjlmag));
      dcosijldrl[2] = (rij[2] / (rijmag * rjlmag)) + (cosijl * rjl[2] / (rjlmag * rjlmag));
      const g = gSpline(core, cosijl, NjiC + NjiH, jtype, rdgdc, rdgdN);
      dgdc = rdgdc.v;
      tmp2 = VA * .5 * (tmp * wjl * dgdc * libmExp(lamdaijl));
      fi[0] = -tmp2 * dcosijldri[0];
      fi[1] = -tmp2 * dcosijldri[1];
      fi[2] = -tmp2 * dcosijldri[2];
      fj[0] = -tmp2 * dcosijldrj[0];
      fj[1] = -tmp2 * dcosijldrj[1];
      fj[2] = -tmp2 * dcosijldrj[2];
      fl[0] = -tmp2 * dcosijldrl[0];
      fl[1] = -tmp2 * dcosijldrl[1];
      fl[2] = -tmp2 * dcosijldrl[2];
      tmp2 = VA * .5 * (tmp * wjl * g * libmExp(lamdaijl) * 4.0 * kron(jtype, 1));
      fj[0] += tmp2 * (rjl[0] / rjlmag);
      fj[1] += tmp2 * (rjl[1] / rjlmag);
      fj[2] += tmp2 * (rjl[2] / rjlmag);
      fl[0] -= tmp2 * (rjl[0] / rjlmag);
      fl[1] -= tmp2 * (rjl[1] / rjlmag);
      fl[2] -= tmp2 * (rjl[2] / rjlmag);
      tmp2 = VA * .5 * (tmp * dwjl * g * libmExp(lamdaijl)) / rjlmag;
      fj[0] -= tmp2 * rjl[0];
      fj[1] -= tmp2 * rjl[1];
      fj[2] -= tmp2 * rjl[2];
      fl[0] += tmp2 * rjl[0];
      fl[1] += tmp2 * rjl[1];
      fl[2] += tmp2 * rjl[2];
      tmp2 = VA * .5 * (tmp * dN2[ltype] * dwjl) / rjlmag;
      fj[0] -= tmp2 * rjl[0];
      fj[1] -= tmp2 * rjl[1];
      fj[2] -= tmp2 * rjl[2];
      fl[0] += tmp2 * rjl[0];
      fl[1] += tmp2 * rjl[1];
      fl[2] += tmp2 * rjl[2];
      tmp2 = VA * .5 * (tmp * tmp3 * dwjl) / rjlmag;
      fj[0] -= tmp2 * rjl[0];
      fj[1] -= tmp2 * rjl[1];
      fj[2] -= tmp2 * rjl[2];
      fl[0] += tmp2 * rjl[0];
      fl[1] += tmp2 * rjl[1];
      fl[2] += tmp2 * rjl[2];
      addF(f, atomi, fi[0], fi[1], fi[2]);
      addF(f, atomj, fj[0], fj[1], fj[2]);
      addF(f, atoml, fl[0], fl[1], fl[2]);
      if (vatom) {
        rlj[0] = -rjl[0]; rlj[1] = -rjl[1]; rlj[2] = -rjl[2];
        vTally3(vatom, atomi, atomj, atoml, fi, fl, rij, rlj);
      }
    }
    // bond-centric P cross forces (no-op for atom-centric P)
    bondorderPijCross(core, atomi, atomj, itype, jtype, VA, tmppij, tmppji, dN2PIJ, dN2PJI);
    dN3[0] = dN3piRC[0];
    dN3[1] = dN3piRC[1];
    dN3[2] = dN3piRC[2];
    // --- piRC forces, i side
    for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
      const atomk = rlist[kk];
      if (atomk === atomj) continue;
      const ktype = mt[atomk];
      rik[0] = x[3 * atomi] - x[3 * atomk];
      rik[1] = x[3 * atomi + 1] - x[3 * atomk + 1];
      rik[2] = x[3 * atomi + 2] - x[3 * atomk + 2];
      const rikmag = Math.sqrt((rik[0] * rik[0]) + (rik[1] * rik[1]) + (rik[2] * rik[2]));
      const wik = sp(rikmag, rmin[pidx(itype, ktype)], rmax[pidx(itype, ktype)], rwik);
      const dwik = rwik.v;
      const Nki = nC[atomk] - (wik * kron(itype, 0)) + nH[atomk] - (wik * kron(itype, 1));
      const SpN = sp(Nki, Nmin, Nmax, rdNki);
      const dNki = rdNki.v;
      tmp2 = VA * dN3[0] * dwik / rikmag;
      f[3 * atomi] -= tmp2 * rik[0];
      f[3 * atomi + 1] -= tmp2 * rik[1];
      f[3 * atomi + 2] -= tmp2 * rik[2];
      f[3 * atomk] += tmp2 * rik[0];
      f[3 * atomk + 1] += tmp2 * rik[1];
      f[3 * atomk + 2] += tmp2 * rik[2];
      if (vatom) vTally2(vatom, atomi, atomk, -tmp2, rik[0], rik[1], rik[2]);
      if (ktype !== 0) continue;
      tmp2 = VA * dN3[2] * (2.0 * NconjtmpI * dwik * SpN) / rikmag;
      f[3 * atomi] -= tmp2 * rik[0];
      f[3 * atomi + 1] -= tmp2 * rik[1];
      f[3 * atomi + 2] -= tmp2 * rik[2];
      f[3 * atomk] += tmp2 * rik[0];
      f[3 * atomk + 1] += tmp2 * rik[1];
      f[3 * atomk + 2] += tmp2 * rik[2];
      if (vatom) vTally2(vatom, atomi, atomk, -tmp2, rik[0], rik[1], rik[2]);
      if (Math.abs(dNki) > TOL) {
        for (let nn = rstart[atomk]; nn < rstart[atomk + 1]; nn++) {
          const atomn = rlist[nn];
          if (atomn === atomi) continue;
          const ntype = mt[atomn];
          rkn[0] = x[3 * atomk] - x[3 * atomn];
          rkn[1] = x[3 * atomk + 1] - x[3 * atomn + 1];
          rkn[2] = x[3 * atomk + 2] - x[3 * atomn + 2];
          const rknmag = Math.sqrt((rkn[0] * rkn[0]) + (rkn[1] * rkn[1]) + (rkn[2] * rkn[2]));
          sp(rknmag, rmin[pidx(ktype, ntype)], rmax[pidx(ktype, ntype)], rdwkn);
          const dwkn = rdwkn.v;
          tmp2 = VA * dN3[2] * (2.0 * NconjtmpI * wik * dNki * dwkn) / rknmag;
          f[3 * atomk] -= tmp2 * rkn[0];
          f[3 * atomk + 1] -= tmp2 * rkn[1];
          f[3 * atomk + 2] -= tmp2 * rkn[2];
          f[3 * atomn] += tmp2 * rkn[0];
          f[3 * atomn + 1] += tmp2 * rkn[1];
          f[3 * atomn + 2] += tmp2 * rkn[2];
          if (vatom) vTally2(vatom, atomk, atomn, -tmp2, rkn[0], rkn[1], rkn[2]);
        }
      }
    }
    // --- piRC forces, j side
    for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
      const atoml = rlist[ll];
      if (atoml === atomi) continue;
      const ltype = mt[atoml];
      rjl[0] = x[3 * atomj] - x[3 * atoml];
      rjl[1] = x[3 * atomj + 1] - x[3 * atoml + 1];
      rjl[2] = x[3 * atomj + 2] - x[3 * atoml + 2];
      const rjlmag = Math.sqrt((rjl[0] * rjl[0]) + (rjl[1] * rjl[1]) + (rjl[2] * rjl[2]));
      const wjl = sp(rjlmag, rmin[pidx(jtype, ltype)], rmax[pidx(jtype, ltype)], rwjl);
      const dwjl = rwjl.v;
      const Nlj = nC[atoml] - (wjl * kron(jtype, 0)) + nH[atoml] - (wjl * kron(jtype, 1));
      const SpN = sp(Nlj, Nmin, Nmax, rdNlj);
      const dNlj = rdNlj.v;
      tmp2 = VA * dN3[1] * dwjl / rjlmag;
      f[3 * atomj] -= tmp2 * rjl[0];
      f[3 * atomj + 1] -= tmp2 * rjl[1];
      f[3 * atomj + 2] -= tmp2 * rjl[2];
      f[3 * atoml] += tmp2 * rjl[0];
      f[3 * atoml + 1] += tmp2 * rjl[1];
      f[3 * atoml + 2] += tmp2 * rjl[2];
      if (vatom) vTally2(vatom, atomj, atoml, -tmp2, rjl[0], rjl[1], rjl[2]);
      if (ltype !== 0) continue;
      tmp2 = VA * dN3[2] * (2.0 * NconjtmpJ * dwjl * SpN) / rjlmag;
      f[3 * atomj] -= tmp2 * rjl[0];
      f[3 * atomj + 1] -= tmp2 * rjl[1];
      f[3 * atomj + 2] -= tmp2 * rjl[2];
      f[3 * atoml] += tmp2 * rjl[0];
      f[3 * atoml + 1] += tmp2 * rjl[1];
      f[3 * atoml + 2] += tmp2 * rjl[2];
      if (vatom) vTally2(vatom, atomj, atoml, -tmp2, rjl[0], rjl[1], rjl[2]);
      if (Math.abs(dNlj) > TOL) {
        for (let nn = rstart[atoml]; nn < rstart[atoml + 1]; nn++) {
          const atomn = rlist[nn];
          if (atomn === atomj) continue;
          const ntype = mt[atomn];
          rln[0] = x[3 * atoml] - x[3 * atomn];
          rln[1] = x[3 * atoml + 1] - x[3 * atomn + 1];
          rln[2] = x[3 * atoml + 2] - x[3 * atomn + 2];
          const rlnmag = Math.sqrt((rln[0] * rln[0]) + (rln[1] * rln[1]) + (rln[2] * rln[2]));
          sp(rlnmag, rmin[pidx(ltype, ntype)], rmax[pidx(ltype, ntype)], rdwln);
          const dwln = rdwln.v;
          tmp2 = VA * dN3[2] * (2.0 * NconjtmpJ * wjl * dNlj * dwln) / rlnmag;
          f[3 * atoml] -= tmp2 * rln[0];
          f[3 * atoml + 1] -= tmp2 * rln[1];
          f[3 * atoml + 2] -= tmp2 * rln[2];
          f[3 * atomn] += tmp2 * rln[0];
          f[3 * atomn + 1] += tmp2 * rln[1];
          f[3 * atomn + 2] += tmp2 * rln[2];
          if (vatom) vTally2(vatom, atoml, atomn, -tmp2, rln[0], rln[1], rln[2]);
        }
      }
    }
    // --- T_ij force: four-body paths, with the LJ-form cosines of the C++ text
    dN3[0] = dN3Tij[0];
    dN3[1] = dN3Tij[1];
    dN3[2] = dN3Tij[2];
    if (Math.abs(Tij) > TOL) {
      const atom2 = atomi, atom3 = atomj;
      r32[0] = x[3 * atom3] - x[3 * atom2];
      r32[1] = x[3 * atom3 + 1] - x[3 * atom2 + 1];
      r32[2] = x[3 * atom3 + 2] - x[3 * atom2 + 2];
      const r32mag = Math.sqrt((r32[0] * r32[0]) + (r32[1] * r32[1]) + (r32[2] * r32[2]));
      r23[0] = -r32[0];
      r23[1] = -r32[1];
      r23[2] = -r32[2];
      let r23mag = r32mag;
      for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
        const atomk = rlist[kk];
        const atom1 = atomk;
        const ktype = mt[atomk];
        if (atomk === atomj) continue;
        r21[0] = x[3 * atom2] - x[3 * atom1];
        r21[1] = x[3 * atom2 + 1] - x[3 * atom1 + 1];
        r21[2] = x[3 * atom2 + 2] - x[3 * atom1 + 2];
        const r21mag = Math.sqrt(r21[0] * r21[0] + r21[1] * r21[1] + r21[2] * r21[2]);
        const cos321 = ((r21[0] * rij[0]) + (r21[1] * rij[1]) + (r21[2] * rij[2])) / (r21mag * rijmag);
        const sin321 = Math.sqrt(1.0 - cos321 * cos321);
        if ((sin321 > TOL) && (r21mag > TOL)) {
          const sink2i = 1.0 / (sin321 * sin321);
          const rik2i = 1.0 / (r21mag * r21mag);
          let rr = (rijmag * rijmag) - (r21mag * r21mag);
          rjk[0] = r21[0] - r23[0];
          rjk[1] = r21[1] - r23[1];
          rjk[2] = r21[2] - r23[2];
          const rjk2 = (rjk[0] * rjk[0]) + (rjk[1] * rjk[1]) + (rjk[2] * rjk[2]);
          const rijrik = 2.0 * r23mag * r21mag;
          let rik2 = r21mag * r21mag;
          const dctik = (-rr + rjk2) / (rijrik * rik2);
          const dctij = (rr + rjk2) / (rijrik * r23mag * r23mag);
          const dctjk = -2.0 / rijrik;
          const w21 = sp(r21mag, rmin[pidx(itype, ktype)], rmaxp[pidx(itype, ktype)], rdw21);
          const dw21 = rdw21.v;
          rijmag = r32mag;
          const rikmag = r21mag;
          const rij2 = r32mag * r32mag;
          rik2 = r21mag * r21mag;
          const costmp = 0.5 * (rij2 + rik2 - rjk2) / rijmag / rikmag;
          const tspjik = sp2(costmp, P.thmin, P.thmax, rdts);
          const dtsjik = -rdts.v;
          for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
            const atoml = rlist[ll];
            const atom4 = atoml;
            const ltype = mt[atoml];
            if (atoml === atomi || atoml === atomk) continue;
            r34[0] = x[3 * atom3] - x[3 * atom4];
            r34[1] = x[3 * atom3 + 1] - x[3 * atom4 + 1];
            r34[2] = x[3 * atom3 + 2] - x[3 * atom4 + 2];
            const r34mag = Math.sqrt(r34[0] * r34[0] + r34[1] * r34[1] + r34[2] * r34[2]);
            const cos234 = (r32[0] * r34[0] + r32[1] * r34[1] + r32[2] * r34[2]) / (r32mag * r34mag);
            const sin234 = Math.sqrt(1.0 - cos234 * cos234);
            if ((sin234 > TOL) && (r34mag > TOL)) {
              const sinl2i = 1.0 / (sin234 * sin234);
              const rjl2i = 1.0 / (r34mag * r34mag);
              const w34 = sp(r34mag, rmin[pidx(jtype, ltype)], rmaxp[pidx(jtype, ltype)], rdw34);
              const dw34 = rdw34.v;
              rr = (r23mag * r23mag) - (r34mag * r34mag);
              ril[0] = r23[0] + r34[0];
              ril[1] = r23[1] + r34[1];
              ril[2] = r23[2] + r34[2];
              const ril2 = (ril[0] * ril[0]) + (ril[1] * ril[1]) + (ril[2] * ril[2]);
              const rijrjl = 2.0 * r23mag * r34mag;
              let rjl2 = r34mag * r34mag;
              const dctjl = (-rr + ril2) / (rijrjl * rjl2);
              const dctji = (rr + ril2) / (rijrjl * r23mag * r23mag);
              const dctil = -2.0 / rijrjl;
              const rjlmag = r34mag;
              rjl2 = r34mag * r34mag;
              const costmp2 = 0.5 * (rij2 + rjl2 - ril2) / rijmag / rjlmag;
              const tspijl = sp2(costmp2, P.thmin, P.thmax, rdtl);
              const dtsijl = -rdtl.v;
              const prefactor = VA * Tij;
              cross321[0] = (r32[1] * r21[2]) - (r32[2] * r21[1]);
              cross321[1] = (r32[2] * r21[0]) - (r32[0] * r21[2]);
              cross321[2] = (r32[0] * r21[1]) - (r32[1] * r21[0]);
              cross234[0] = (r23[1] * r34[2]) - (r23[2] * r34[1]);
              cross234[1] = (r23[2] * r34[0]) - (r23[0] * r34[2]);
              cross234[2] = (r23[0] * r34[1]) - (r23[1] * r34[0]);
              const cwnum = (cross321[0] * cross234[0]) + (cross321[1] * cross234[1]) + (cross321[2] * cross234[2]);
              const cwnom = r21mag * r34mag * r23mag * r23mag * sin321 * sin234;
              const om1234 = cwnum / cwnom;
              const cw = om1234;
              const dt1dik = rik2i - (dctik * sink2i * cos321);
              const dt1djk = (-dctjk * sink2i * cos321);
              const dt1djl = rjl2i - (dctjl * sinl2i * cos234);
              const dt1dil = (-dctil * sinl2i * cos234);
              const dt1dij = (2.0 / (r23mag * r23mag)) - (dctij * sink2i * cos321) - (dctji * sinl2i * cos234);
              dt2dik[0] = (-r23[2] * cross234[1]) + (r23[1] * cross234[2]);
              dt2dik[1] = (-r23[0] * cross234[2]) + (r23[2] * cross234[0]);
              dt2dik[2] = (-r23[1] * cross234[0]) + (r23[0] * cross234[1]);
              dt2djl[0] = (-r23[1] * cross321[2]) + (r23[2] * cross321[1]);
              dt2djl[1] = (-r23[2] * cross321[0]) + (r23[0] * cross321[2]);
              dt2djl[2] = (-r23[0] * cross321[1]) + (r23[1] * cross321[0]);
              dt2dij[0] = (r21[2] * cross234[1]) - (r34[2] * cross321[1]) - (r21[1] * cross234[2]) + (r34[1] * cross321[2]);
              dt2dij[1] = (r21[0] * cross234[2]) - (r34[0] * cross321[2]) - (r21[2] * cross234[0]) + (r34[2] * cross321[0]);
              dt2dij[2] = (r21[1] * cross234[0]) - (r34[1] * cross321[0]) - (r21[0] * cross234[1]) + (r34[0] * cross321[1]);
              const aa = (prefactor * 2.0 * cw / cwnom) * w21 * w34 * (1.0 - tspjik) * (1.0 - tspijl);
              const aaa2 = -prefactor * (1.0 - om1234 * om1234) * w21 * w34;
              const at2 = aa * cwnum;
              const fcijpc = (-dt1dij * at2) + (aaa2 * dtsjik * dctij * (1.0 - tspijl)) + (aaa2 * dtsijl * dctji * (1.0 - tspjik));
              const fcikpc = (-dt1dik * at2) + (aaa2 * dtsjik * dctik * (1.0 - tspijl));
              const fcjlpc = (-dt1djl * at2) + (aaa2 * dtsijl * dctjl * (1.0 - tspjik));
              const fcjkpc = (-dt1djk * at2) + (aaa2 * dtsjik * dctjk * (1.0 - tspijl));
              const fcilpc = (-dt1dil * at2) + (aaa2 * dtsijl * dctil * (1.0 - tspjik));
              for (let q = 0; q < 3; q++) {
                F23[q] = (fcijpc * r23[q]) + (aa * dt2dij[q]);
                F12[q] = (fcikpc * r21[q]) + (aa * dt2dik[q]);
                F34[q] = (fcjlpc * r34[q]) + (aa * dt2djl[q]);
                F31[q] = (fcjkpc * rjk[q]);
                F24[q] = (fcilpc * ril[q]);
              }
              for (let q = 0; q < 3; q++) {
                f1[q] = -F12[q] - F31[q];
                f2[q] = F23[q] + F12[q] + F24[q];
                f3[q] = -F23[q] + F34[q] + F31[q];
                f4[q] = -F34[q] - F24[q];
              }
              // coordination forces
              let t2 = VA * Tij * (1.0 - (om1234 * om1234)) * (1.0 - tspjik) * (1.0 - tspijl) * dw21 * w34 / r21mag;
              f2[0] -= t2 * r21[0];
              f2[1] -= t2 * r21[1];
              f2[2] -= t2 * r21[2];
              f1[0] += t2 * r21[0];
              f1[1] += t2 * r21[1];
              f1[2] += t2 * r21[2];
              t2 = VA * Tij * (1.0 - (om1234 * om1234)) * (1.0 - tspjik) * (1.0 - tspijl) * w21 * dw34 / r34mag;
              f3[0] -= t2 * r34[0];
              f3[1] -= t2 * r34[1];
              f3[2] -= t2 * r34[2];
              f4[0] += t2 * r34[0];
              f4[1] += t2 * r34[1];
              f4[2] += t2 * r34[2];
              addF(f, atom1, f1[0], f1[1], f1[2]);
              addF(f, atom2, f2[0], f2[1], f2[2]);
              addF(f, atom3, f3[0], f3[1], f3[2]);
              addF(f, atom4, f4[0], f4[1], f4[2]);
              if (vatom) {
                r13[0] = -rjk[0]; r13[1] = -rjk[1]; r13[2] = -rjk[2];
                r43[0] = -r34[0]; r43[1] = -r34[1]; r43[2] = -r34[2];
                vTally4(vatom, atom1, atom2, atom3, atom4, f1, f2, f4, r13, r23, r43);
              }
            }
          }
        }
      }
    }
    // --- T_ij forces with Etmp: i side, then j side (as in the C++ text)
    for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
      const atomk = rlist[kk];
      if (atomk === atomj) continue;
      const ktype = mt[atomk];
      rik[0] = x[3 * atomi] - x[3 * atomk];
      rik[1] = x[3 * atomi + 1] - x[3 * atomk + 1];
      rik[2] = x[3 * atomi + 2] - x[3 * atomk + 2];
      const rikmag = Math.sqrt((rik[0] * rik[0]) + (rik[1] * rik[1]) + (rik[2] * rik[2]));
      const wik = sp(rikmag, rmin[pidx(itype, ktype)], rmax[pidx(itype, ktype)], rwik);
      const dwik = rwik.v;
      const Nki = nC[atomk] - (wik * kron(itype, 0)) + nH[atomk] - (wik * kron(itype, 1));
      const SpN = sp(Nki, Nmin, Nmax, rdNki);
      const dNki = rdNki.v;
      tmp2 = VA * dN3[0] * dwik * Etmp / rikmag;
      f[3 * atomi] -= tmp2 * rik[0];
      f[3 * atomi + 1] -= tmp2 * rik[1];
      f[3 * atomi + 2] -= tmp2 * rik[2];
      f[3 * atomk] += tmp2 * rik[0];
      f[3 * atomk + 1] += tmp2 * rik[1];
      f[3 * atomk + 2] += tmp2 * rik[2];
      if (vatom) vTally2(vatom, atomi, atomk, -tmp2, rik[0], rik[1], rik[2]);
      if (ktype !== 0) continue;
      tmp2 = VA * dN3[2] * (2.0 * NconjtmpI * dwik * SpN) * Etmp / rikmag;
      f[3 * atomi] -= tmp2 * rik[0];
      f[3 * atomi + 1] -= tmp2 * rik[1];
      f[3 * atomi + 2] -= tmp2 * rik[2];
      f[3 * atomk] += tmp2 * rik[0];
      f[3 * atomk + 1] += tmp2 * rik[1];
      f[3 * atomk + 2] += tmp2 * rik[2];
      if (vatom) vTally2(vatom, atomi, atomk, -tmp2, rik[0], rik[1], rik[2]);
      if (Math.abs(dNki) > TOL) {
        for (let nn = rstart[atomk]; nn < rstart[atomk + 1]; nn++) {
          const atomn = rlist[nn];
          if (atomn === atomi) continue;
          const ntype = mt[atomn];
          rkn[0] = x[3 * atomk] - x[3 * atomn];
          rkn[1] = x[3 * atomk + 1] - x[3 * atomn + 1];
          rkn[2] = x[3 * atomk + 2] - x[3 * atomn + 2];
          const rknmag = Math.sqrt((rkn[0] * rkn[0]) + (rkn[1] * rkn[1]) + (rkn[2] * rkn[2]));
          sp(rknmag, rmin[pidx(ktype, ntype)], rmax[pidx(ktype, ntype)], rdwkn);
          const dwkn = rdwkn.v;
          tmp2 = VA * dN3[2] * (2.0 * NconjtmpI * wik * dNki * dwkn) * Etmp / rknmag;
          f[3 * atomk] -= tmp2 * rkn[0];
          f[3 * atomk + 1] -= tmp2 * rkn[1];
          f[3 * atomk + 2] -= tmp2 * rkn[2];
          f[3 * atomn] += tmp2 * rkn[0];
          f[3 * atomn + 1] += tmp2 * rkn[1];
          f[3 * atomn + 2] += tmp2 * rkn[2];
          if (vatom) vTally2(vatom, atomk, atomn, -tmp2, rkn[0], rkn[1], rkn[2]);
        }
      }
    }
    for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
      const atoml = rlist[ll];
      if (atoml === atomi) continue;
      const ltype = mt[atoml];
      rjl[0] = x[3 * atomj] - x[3 * atoml];
      rjl[1] = x[3 * atomj + 1] - x[3 * atoml + 1];
      rjl[2] = x[3 * atomj + 2] - x[3 * atoml + 2];
      const rjlmag = Math.sqrt((rjl[0] * rjl[0]) + (rjl[1] * rjl[1]) + (rjl[2] * rjl[2]));
      const wjl = sp(rjlmag, rmin[pidx(jtype, ltype)], rmax[pidx(jtype, ltype)], rwjl);
      const dwjl = rwjl.v;
      const Nlj = nC[atoml] - (wjl * kron(jtype, 0)) + nH[atoml] - (wjl * kron(jtype, 1));
      const SpN = sp(Nlj, Nmin, Nmax, rdNlj);
      const dNlj = rdNlj.v;
      tmp2 = VA * dN3[1] * dwjl * Etmp / rjlmag;
      f[3 * atomj] -= tmp2 * rjl[0];
      f[3 * atomj + 1] -= tmp2 * rjl[1];
      f[3 * atomj + 2] -= tmp2 * rjl[2];
      f[3 * atoml] += tmp2 * rjl[0];
      f[3 * atoml + 1] += tmp2 * rjl[1];
      f[3 * atoml + 2] += tmp2 * rjl[2];
      if (vatom) vTally2(vatom, atomj, atoml, -tmp2, rjl[0], rjl[1], rjl[2]);
      if (ltype !== 0) continue;
      tmp2 = VA * dN3[2] * (2.0 * NconjtmpJ * dwjl * SpN) * Etmp / rjlmag;
      f[3 * atomj] -= tmp2 * rjl[0];
      f[3 * atomj + 1] -= tmp2 * rjl[1];
      f[3 * atomj + 2] -= tmp2 * rjl[2];
      f[3 * atoml] += tmp2 * rjl[0];
      f[3 * atoml + 1] += tmp2 * rjl[1];
      f[3 * atoml + 2] += tmp2 * rjl[2];
      if (vatom) vTally2(vatom, atomj, atoml, -tmp2, rjl[0], rjl[1], rjl[2]);
      if (Math.abs(dNlj) > TOL) {
        for (let nn = rstart[atoml]; nn < rstart[atoml + 1]; nn++) {
          const atomn = rlist[nn];
          if (atomn === atomj) continue;
          const ntype = mt[atomn];
          rln[0] = x[3 * atoml] - x[3 * atomn];
          rln[1] = x[3 * atoml + 1] - x[3 * atomn + 1];
          rln[2] = x[3 * atoml + 2] - x[3 * atomn + 2];
          const rlnmag = Math.sqrt((rln[0] * rln[0]) + (rln[1] * rln[1]) + (rln[2] * rln[2]));
          sp(rlnmag, rmin[pidx(ltype, ntype)], rmax[pidx(ltype, ntype)], rdwln);
          const dwln = rdwln.v;
          tmp2 = VA * dN3[2] * (2.0 * NconjtmpJ * wjl * dNlj * dwln) * Etmp / rlnmag;
          f[3 * atoml] -= tmp2 * rln[0];
          f[3 * atoml + 1] -= tmp2 * rln[1];
          f[3 * atoml + 2] -= tmp2 * rln[2];
          f[3 * atomn] += tmp2 * rln[0];
          f[3 * atomn + 1] += tmp2 * rln[1];
          f[3 * atomn + 2] += tmp2 * rln[2];
          if (vatom) vTally2(vatom, atoml, atomn, -tmp2, rln[0], rln[1], rln[2]);
        }
      }
    }
  }
  return Stb;
};

/**
 * PairAIREBO::FLJ: the LJ term with the connectivity switch. Each owned pair (i, j) is visited once
 * (the same tag rule as FREBO); the 2-, 3- and 4-atom paths of the REBO neighbor lists give the
 * largest path weight `best`, and C_ij = 1 - best scales the LJ energy.
 */
export const flj = (core: LJCore, nlocal: number): void => {
  const { P, mt, tag, x, f, rstart, rlist, ljFirst, ljNum, ljList, ljp } = core;
  const pidx = (a: number, b: number) => a * 2 + b;
  const eatom = core.eatom, vatom = core.vatom;
  const dwij = newRef(), dwik = newRef(), dwkj = newRef(), dwkm = newRef(), dwmj = newRef();
  const delij = v3(), delik = v3(), deljk = v3(), delkm = v3(), deljm = v3();
  const delikS = v3(), deljkS = v3(), delkmS = v3(), deljmS = v3(), delimS = v3();
  const fa = v3(), fb = v3(), fc = v3(), fd = v3();
  const cutljsq = ljp.cutljsq;
  for (let i = 0; i < nlocal; i++) {
    const itag = tag[i];
    const itype = mt[i];
    if (itype < 0) continue;
    const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
    for (let jj = ljFirst[i], jj1 = jj + ljNum[i]; jj < jj1; jj++) {
      const j = ljList[jj] & NEIGHMASK;
      if (mt[j] < 0) continue;
      const jtag = tag[j];
      if (itag > jtag) {
        if ((itag + jtag) % 2 === 0) continue;
      } else if (itag < jtag) {
        if ((itag + jtag) % 2 === 1) continue;
      } else {
        if (x[3 * j + 2] < ztmp) continue;
        if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
        if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
      }
      const jtype = mt[j];
      const atomi = i, atomj = j;
      delij[0] = xtmp - x[3 * j];
      delij[1] = ytmp - x[3 * j + 1];
      delij[2] = ztmp - x[3 * j + 2];
      const rijsq = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
      // if outside of LJ cutoff, skip; outside of the 4-path cutoff, best = 0 and no path test
      let npath = 0, testpath = 0, done = 0;
      let best = 0.0;
      let atomk = -1, atomm = -1;
      if (rijsq >= cutljsq[pidx(itype, jtype)]) continue;
      const rij = Math.sqrt(rijsq);
      if (rij >= ljp.cut3rebo) {
        best = 0.0;
        testpath = 0;
      } else if (rij >= P.rcmax[pidx(itype, jtype)]) {
        best = 0.0;
        testpath = 1;
      } else {
        best = sp(rij, P.rcmin[pidx(itype, jtype)], P.rcmax[pidx(itype, jtype)], dwij);
        npath = 2;
        testpath = best < 1.0 ? 1 : 0;
      }
      // scalars of the best 3- and 4-paths
      let wikS = 0, dwikS = 0, rikS = 1, wkjS = 0, dwkjS = 0, rkjS = 1;
      let wkmS = 0, dwkmS = 0, rkmS = 1, wmjS = 0, dwmjS = 0, rmjS = 1;
      if (testpath) {
        // all 3-body paths I-K-J (I-K from the REBO neighbors of I), then the 4-body paths I-K-M-J
        for (let kk = rstart[i]; kk < rstart[i + 1] && done === 0; kk++) {
          const k = rlist[kk];
          if (k === j) continue;
          const ktype = mt[k];
          delik[0] = x[3 * i] - x[3 * k];
          delik[1] = x[3 * i + 1] - x[3 * k + 1];
          delik[2] = x[3 * i + 2] - x[3 * k + 2];
          let rsq = delik[0] * delik[0] + delik[1] * delik[1] + delik[2] * delik[2];
          let rik: number, wik: number, dwikv: number;
          if (rsq < P.rcmaxsq[pidx(itype, ktype)]) {
            rik = Math.sqrt(rsq);
            wik = sp(rik, P.rcmin[pidx(itype, ktype)], P.rcmax[pidx(itype, ktype)], dwik);
            dwikv = dwik.v;
          } else {
            dwikv = wik = 0.0;
            rik = 1.0;
          }
          if (wik > best) {
            deljk[0] = x[3 * j] - x[3 * k];
            deljk[1] = x[3 * j + 1] - x[3 * k + 1];
            deljk[2] = x[3 * j + 2] - x[3 * k + 2];
            rsq = deljk[0] * deljk[0] + deljk[1] * deljk[1] + deljk[2] * deljk[2];
            if (rsq < P.rcmaxsq[pidx(ktype, jtype)]) {
              const rkj = Math.sqrt(rsq);
              const wkj = sp(rkj, P.rcmin[pidx(ktype, jtype)], P.rcmax[pidx(ktype, jtype)], dwkj);
              if (wik * wkj > best) {
                best = wik * wkj;
                npath = 3;
                atomk = k;
                delikS.set(delik);
                rikS = rik;
                wikS = wik;
                dwikS = dwikv;
                deljkS.set(deljk);
                rkjS = rkj;
                wkjS = wkj;
                dwkjS = dwkj.v;
                if (best === 1.0) {
                  break;
                }
              }
            }
            // 4-body paths I-K-M-J: K-M from the REBO neighbors of K
            for (let mm = rstart[k]; mm < rstart[k + 1] && done === 0; mm++) {
              const m = rlist[mm];
              if (m === i || m === j) continue;
              const mtype = mt[m];
              delkm[0] = x[3 * k] - x[3 * m];
              delkm[1] = x[3 * k + 1] - x[3 * m + 1];
              delkm[2] = x[3 * k + 2] - x[3 * m + 2];
              rsq = delkm[0] * delkm[0] + delkm[1] * delkm[1] + delkm[2] * delkm[2];
              let rkm: number, wkm: number, dwkmv: number;
              if (rsq < P.rcmaxsq[pidx(ktype, mtype)]) {
                rkm = Math.sqrt(rsq);
                wkm = sp(rkm, P.rcmin[pidx(ktype, mtype)], P.rcmax[pidx(ktype, mtype)], dwkm);
                dwkmv = dwkm.v;
              } else {
                dwkmv = wkm = 0.0;
                rkm = 1.0;
              }
              if (wik * wkm > best) {
                deljm[0] = x[3 * j] - x[3 * m];
                deljm[1] = x[3 * j + 1] - x[3 * m + 1];
                deljm[2] = x[3 * j + 2] - x[3 * m + 2];
                rsq = deljm[0] * deljm[0] + deljm[1] * deljm[1] + deljm[2] * deljm[2];
                if (rsq < P.rcmaxsq[pidx(mtype, jtype)]) {
                  const rmj = Math.sqrt(rsq);
                  const wmj = sp(rmj, P.rcmin[pidx(mtype, jtype)], P.rcmax[pidx(mtype, jtype)], dwmj);
                  if (wik * wkm * wmj > best) {
                    best = wik * wkm * wmj;
                    npath = 4;
                    atomk = k;
                    delikS.set(delik);
                    rikS = rik;
                    wikS = wik;
                    dwikS = dwikv;
                    atomm = m;
                    delkmS.set(delkm);
                    rkmS = rkm;
                    wkmS = wkm;
                    dwkmS = dwkmv;
                    deljmS.set(deljm);
                    rmjS = rmj;
                    wmjS = wmj;
                    dwmjS = dwmj.v;
                    if (best === 1.0) {
                      done = 1;
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }
      const cij = 1.0 - best;
      if (cij === 0.0) continue;
      let rljmin = P.sigma[pidx(itype, jtype)];
      let rljmax = ljp.sigcut * rljmin;
      rljmin = ljp.sigmin * rljmin;
      let slw: number, dslw: number;
      if (rij > rljmax) {
        slw = 0.0;
        dslw = 0.0;
      } else if (rij > rljmin) {
        const drij = rij - rljmin;
        const swidth = rljmax - rljmin;
        const tee = drij / swidth;
        const tee2 = tee * tee;
        slw = 1.0 - tee2 * (3.0 - 2.0 * tee);
        dslw = -6.0 * tee * (1.0 - tee) / swidth;
      } else {
        slw = 1.0;
        dslw = 0.0;
      }
      let vdw: number, dvdw: number;
      if (ljp.morseflag) {
        const exr = libmExp(-rij * ljp.lj4[pidx(itype, jtype)]);
        vdw = ljp.lj1[pidx(itype, jtype)] * exr * (ljp.lj2[pidx(itype, jtype)] * exr - 2);
        dvdw = ljp.lj3[pidx(itype, jtype)] * exr * (1 - ljp.lj2[pidx(itype, jtype)] * exr);
      } else {
        const r2inv = 1.0 / rijsq;
        const r6inv = r2inv * r2inv * r2inv;
        vdw = r6inv * (ljp.lj3[pidx(itype, jtype)] * r6inv - ljp.lj4[pidx(itype, jtype)]);
        dvdw = -r6inv * (ljp.lj1[pidx(itype, jtype)] * r6inv - ljp.lj2[pidx(itype, jtype)]) / rij;
      }
      const VLJ = vdw * slw;
      const dVLJ = dvdw * slw + vdw * dslw;
      const Str = sp2(rij, P.rcLJmin[pidx(itype, jtype)], P.rcLJmax[pidx(itype, jtype)], rdStr);
      const dStr = rdStr.v;
      const VA = Str * cij * VLJ;
      let Stb: number;
      if (Str > 0.0) {
        Stb = bondorderLJ(core, i, j, P.rcmin[pidx(itype, jtype)], VA, delij, rij);
      } else Stb = 0.0;
      let fpair = -(dStr * (Stb * cij * VLJ - cij * VLJ) + dVLJ * (Str * Stb * cij + cij - Str * cij)) / rij;
      f[3 * i] += delij[0] * fpair;
      f[3 * i + 1] += delij[1] * fpair;
      f[3 * i + 2] += delij[2] * fpair;
      f[3 * j] -= delij[0] * fpair;
      f[3 * j + 1] -= delij[1] * fpair;
      f[3 * j + 2] -= delij[2] * fpair;
      const evdwl = VA * Stb + (1.0 - Str) * cij * VLJ;
      core.acc.evdwl += evdwl;
      if (eatom) {
        const half = 0.5 * evdwl;
        eatom[i] += half;
        eatom[j] += half;
      }
      if (vatom) vTally2(vatom, i, j, fpair, delij[0], delij[1], delij[2]);
      if (cij < 1.0) {
        const dC = Str * Stb * VLJ + (1.0 - Str) * VLJ;
        if (npath === 2) {
          fpair = dC * dwij.v / rij;
          f[3 * atomi] += delij[0] * fpair;
          f[3 * atomi + 1] += delij[1] * fpair;
          f[3 * atomi + 2] += delij[2] * fpair;
          f[3 * atomj] -= delij[0] * fpair;
          f[3 * atomj + 1] -= delij[1] * fpair;
          f[3 * atomj + 2] -= delij[2] * fpair;
          if (vatom) vTally2(vatom, atomi, atomj, fpair, delij[0], delij[1], delij[2]);
        } else if (npath === 3) {
          const fpair1 = dC * dwikS * wkjS / rikS;
          fa[0] = delikS[0] * fpair1;
          fa[1] = delikS[1] * fpair1;
          fa[2] = delikS[2] * fpair1;
          const fpair2 = dC * wikS * dwkjS / rkjS;
          fb[0] = deljkS[0] * fpair2;
          fb[1] = deljkS[1] * fpair2;
          fb[2] = deljkS[2] * fpair2;
          addF(f, atomi, fa[0], fa[1], fa[2]);
          addF(f, atomj, fb[0], fb[1], fb[2]);
          addF(f, atomk, -(fa[0] + fb[0]), -(fa[1] + fb[1]), -(fa[2] + fb[2]));
          if (vatom) vTally3(vatom, atomi, atomj, atomk, fa, fb, delikS, deljkS);
        } else if (npath === 4) {
          const fpair1 = dC * dwikS * wkmS * wmjS / rikS;
          fa[0] = delikS[0] * fpair1;
          fa[1] = delikS[1] * fpair1;
          fa[2] = delikS[2] * fpair1;
          const fpair2 = dC * wikS * dwkmS * wmjS / rkmS;
          fc[0] = delkmS[0] * fpair2 - fa[0];
          fc[1] = delkmS[1] * fpair2 - fa[1];
          fc[2] = delkmS[2] * fpair2 - fa[2];
          const fpair3 = dC * wikS * wkmS * dwmjS / rmjS;
          fb[0] = deljmS[0] * fpair3;
          fb[1] = deljmS[1] * fpair3;
          fb[2] = deljmS[2] * fpair3;
          fd[0] = -delkmS[0] * fpair2 - fb[0];
          fd[1] = -delkmS[1] * fpair2 - fb[1];
          fd[2] = -delkmS[2] * fpair2 - fb[2];
          addF(f, atomi, fa[0], fa[1], fa[2]);
          addF(f, atomj, fb[0], fb[1], fb[2]);
          addF(f, atomk, fc[0], fc[1], fc[2]);
          addF(f, atomm, fd[0], fd[1], fd[2]);
          if (vatom) {
            delimS[0] = delikS[0] + delkmS[0];
            delimS[1] = delikS[1] + delkmS[1];
            delimS[2] = delikS[2] + delkmS[2];
            vTally4(vatom, atomi, atomj, atomk, atomm, fa, fb, fc, delimS, deljmS, delkmS);
          }
        }
      }
    }
  }
};
