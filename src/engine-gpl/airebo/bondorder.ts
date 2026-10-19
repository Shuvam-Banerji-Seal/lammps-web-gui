/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_airebo.cpp (PairAIREBO::bondorder, the REBO bond-order
 * function b_ij with its P_ij, P_ji, pi^RC and T_ij terms and their forces), at commit
 * b332f28. Authors credited in the C++ file headers: Ase Henry (MIT); bugfixes and
 * optimizations: Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer (Temple U),
 * Markus Hoehnerbach (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M:
 * Thomas C. O'Connor (JHU) 2014. Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { sp, sp2, type Ref } from './splines';
import { TOL, kron, gSpline, piRCSpline, TijSpline, type ReboCore, newRef } from './rebo';
import { bondorderPijCross, pijEval } from './bc';
import { exp as libmExp } from '../libm/exp';
import { vTally2, vTally3, vTally4 } from './tally';

const v3 = (): Float64Array => new Float64Array(3);

// scratch vectors of the C++ locals (the function is not re-entered)
const rik = v3(), rjl = v3(), rkn = v3();
const rln = v3(), r32 = v3(), r23 = v3(), r21 = v3(), r34 = v3(), rjk = v3(), ril = v3();
const rji = v3(), rki = v3(), rlj = v3(), r13 = v3(), r43 = v3();
const dcosjikdri = v3(), dcosijldri = v3(), dcosjikdrk = v3(), dcosjikdrj = v3();
const dcosijldrj = v3(), dcosijldrl = v3();
const fi = v3(), fj = v3(), fk = v3(), fl = v3();
const cross321 = v3(), cross234 = v3();
const dt2dik = v3(), dt2djl = v3(), dt2dij = v3();
const F23 = v3(), F12 = v3(), F34 = v3(), F31 = v3(), F24 = v3();
const f1 = v3(), f2 = v3(), f3 = v3(), f4 = v3();
const dN2 = new Float64Array(2);
// bond-averaged P derivatives of pij and pji, kept for the AIREBO-BC cross force (bc.ts)
const dN2PIJbc = new Float64Array(2), dN2PJIbc = new Float64Array(2);
const dN3 = new Float64Array(3);
const rdgdc = newRef(), rdgdN = newRef();
const rS = newRef(), rwik = newRef(), rwjl = newRef(), rdNki = newRef(), rdNlj = newRef();
const rwkn = newRef(), rwln = newRef(), rdw21 = newRef(), rdw34 = newRef();
const rdcut = newRef(), rdts = newRef(), rdtl = newRef();

/** Adds the force vector s * src to f at atom a (the C++ f[a][0..2] += ...). */
const addF = (f: Float64Array, a: number, x: number, y: number, z: number): void => {
  f[3 * a] += x;
  f[3 * a + 1] += y;
  f[3 * a + 2] += z;
};

/**
 * PairAIREBO::bondorder(i, j, rij, rijmag, VA, f): the bond-order b_ij (the value multiplying
 * the attractive term VA) for the bond i-j, with the forces of its contributions added to f.
 * rij is the vector from j to i (x_i - x_j), as passed by FREBO.
 */
export const bondorder = (
  core: ReboCore, i: number, j: number, rij: Float64Array, rijmag: number, VA: number,
): number => {
  const { P, nC, nH, x, f, mt, rstart, rlist } = core;
  const vatom = core.vatom;
  const atomi = i, atomj = j;
  const itype = mt[atomi], jtype = mt[atomj];
  const rmin = P.rcmin, rmax = P.rcmax, rmaxp = P.rcmaxp;
  const rho = P.rho;
  const Nmin = P.Nmin, Nmax = P.Nmax;
  const pairIdx = (a: number, b: number) => a * 2 + b;

  const wij = sp(rijmag, rmin[pairIdx(itype, jtype)], rmax[pairIdx(itype, jtype)], core.dwij);
  const NijC = nC[i] - (wij * kron(jtype, 0));
  const NijH = nH[i] - (wij * kron(jtype, 1));
  const NjiC = nC[j] - (wij * kron(itype, 0));
  const NjiH = nH[j] - (wij * kron(itype, 1));
  let bij = 0.0;
  let tmp = 0.0;
  let tmp2 = 0.0;
  let tmp3 = 0.0;
  let NconjtmpI = 0.0;
  let NconjtmpJ = 0.0;
  let Etmp = 0.0;
  let dgdc = 0.0;
  let dgdN = 0.0;

  // --- the i side: sum over the REBO neighbors k of i (k != j)
  for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
    const atomk = rlist[kk];
    if (atomk === atomj) continue;
    const ktype = mt[atomk];
    rik[0] = x[3 * atomi] - x[3 * atomk];
    rik[1] = x[3 * atomi + 1] - x[3 * atomk + 1];
    rik[2] = x[3 * atomi + 2] - x[3 * atomk + 2];
    const rikmag = Math.sqrt((rik[0] * rik[0]) + (rik[1] * rik[1]) + (rik[2] * rik[2]));
    const lamdajik = 4.0 * kron(itype, 1) * ((rho[ktype * 2 + 1] - rikmag) - (rho[jtype * 2 + 1] - rijmag));
    const wik = sp(rikmag, rmin[pairIdx(itype, ktype)], rmax[pairIdx(itype, ktype)], rS);
    const Nki = nC[atomk] - (wik * kron(itype, 0)) + nH[atomk] - (wik * kron(itype, 1));
    let cosjik = ((rij[0] * rik[0]) + (rij[1] * rik[1]) + (rij[2] * rik[2])) / (rijmag * rikmag);
    cosjik = Math.min(cosjik, 1.0);
    cosjik = Math.max(cosjik, -1.0);
    // evaluate splines g and derivatives dg
    const g = gSpline(core, cosjik, NijC + NijH, itype, rdgdc, rdgdN);
    dgdN = rdgdN.v;
    Etmp = Etmp + (wik * g * libmExp(lamdajik));
    tmp3 = tmp3 + (wik * dgdN * libmExp(lamdajik));
    NconjtmpI = NconjtmpI + (kron(ktype, 0) * wik * sp(Nki, Nmin, Nmax, rS));
  }

  const PijS = pijEval(core, NijC, NijH, NjiC, NjiH, itype, jtype, dN2);
  const pij = 1.0 / Math.sqrt(1.0 + Etmp + PijS);
  tmp = -0.5 * (pij * pij * pij);
  dN2PIJbc[0] = dN2[0];
  dN2PIJbc[1] = dN2[1];

  // pij forces
  for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
    const atomk = rlist[kk];
    if (atomk === atomj) continue;
    const ktype = mt[atomk];
    rik[0] = x[3 * atomi] - x[3 * atomk];
    rik[1] = x[3 * atomi + 1] - x[3 * atomk + 1];
    rik[2] = x[3 * atomi + 2] - x[3 * atomk + 2];
    const rikmag = Math.sqrt((rik[0] * rik[0]) + (rik[1] * rik[1]) + (rik[2] * rik[2]));
    const lamdajik = 4.0 * kron(itype, 1) * ((rho[ktype * 2 + 1] - rikmag) - (rho[jtype * 2 + 1] - rijmag));
    const wik = sp(rikmag, rmin[pairIdx(itype, ktype)], rmax[pairIdx(itype, ktype)], rwik);
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
    fj[0] -= tmp2 * (-rij[0] / rijmag);
    fj[1] -= tmp2 * (-rij[1] / rijmag);
    fj[2] -= tmp2 * (-rij[2] / rijmag);
    fi[0] -= tmp2 * ((-rik[0] / rikmag) + (rij[0] / rijmag));
    fi[1] -= tmp2 * ((-rik[1] / rikmag) + (rij[1] / rijmag));
    fi[2] -= tmp2 * ((-rik[2] / rikmag) + (rij[2] / rijmag));
    fk[0] -= tmp2 * (rik[0] / rikmag);
    fk[1] -= tmp2 * (rik[1] / rikmag);
    fk[2] -= tmp2 * (rik[2] / rikmag);
    // coordination forces: dwik
    tmp2 = VA * .5 * (tmp * dwik * g * libmExp(lamdajik)) / rikmag;
    fi[0] -= tmp2 * rik[0];
    fi[1] -= tmp2 * rik[1];
    fi[2] -= tmp2 * rik[2];
    fk[0] += tmp2 * rik[0];
    fk[1] += tmp2 * rik[1];
    fk[2] += tmp2 * rik[2];
    // PIJ forces
    tmp2 = VA * .5 * (tmp * dN2[ktype] * dwik) / rikmag;
    fi[0] -= tmp2 * rik[0];
    fi[1] -= tmp2 * rik[1];
    fi[2] -= tmp2 * rik[2];
    fk[0] += tmp2 * rik[0];
    fk[1] += tmp2 * rik[1];
    fk[2] += tmp2 * rik[2];
    // dgdN forces
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

  // --- the j side: sum over the REBO neighbors l of j (l != i)
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
    const lamdaijl = 4.0 * kron(jtype, 1) * ((rho[ltype * 2 + 1] - rjlmag) - (rho[itype * 2 + 1] - rijmag));
    const wjl = sp(rjlmag, rmin[pairIdx(jtype, ltype)], rmax[pairIdx(jtype, ltype)], rS);
    const Nlj = nC[atoml] - (wjl * kron(jtype, 0)) + nH[atoml] - (wjl * kron(jtype, 1));
    let cosijl = -1.0 * ((rij[0] * rjl[0]) + (rij[1] * rjl[1]) + (rij[2] * rjl[2])) / (rijmag * rjlmag);
    cosijl = Math.min(cosijl, 1.0);
    cosijl = Math.max(cosijl, -1.0);
    // evaluate splines g and derivatives dg
    const g = gSpline(core, cosijl, NjiC + NjiH, jtype, rdgdc, rdgdN);
    dgdN = rdgdN.v;
    Etmp = Etmp + (wjl * g * libmExp(lamdaijl));
    tmp3 = tmp3 + (wjl * dgdN * libmExp(lamdaijl));
    NconjtmpJ = NconjtmpJ + (kron(ltype, 0) * wjl * sp(Nlj, Nmin, Nmax, rS));
  }
  const PjiS = pijEval(core, NjiC, NjiH, NijC, NijH, jtype, itype, dN2);
  const pji = 1.0 / Math.sqrt(1.0 + Etmp + PjiS);
  tmp = -0.5 * (pji * pji * pji);
  dN2PJIbc[0] = dN2[0];
  dN2PJIbc[1] = dN2[1];

  for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
    const atoml = rlist[ll];
    if (atoml === atomi) continue;
    const ltype = mt[atoml];
    rjl[0] = x[3 * atomj] - x[3 * atoml];
    rjl[1] = x[3 * atomj + 1] - x[3 * atoml + 1];
    rjl[2] = x[3 * atomj + 2] - x[3 * atoml + 2];
    const rjlmag = Math.sqrt((rjl[0] * rjl[0]) + (rjl[1] * rjl[1]) + (rjl[2] * rjl[2]));
    const lamdaijl = 4.0 * kron(jtype, 1) * ((rho[ltype * 2 + 1] - rjlmag) - (rho[itype * 2 + 1] - rijmag));
    const wjl = sp(rjlmag, rmin[pairIdx(jtype, ltype)], rmax[pairIdx(jtype, ltype)], rwjl);
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
    // evaluate splines g and derivatives dg
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
    fi[0] -= tmp2 * (rij[0] / rijmag);
    fi[1] -= tmp2 * (rij[1] / rijmag);
    fi[2] -= tmp2 * (rij[2] / rijmag);
    fj[0] -= tmp2 * ((-rjl[0] / rjlmag) - (rij[0] / rijmag));
    fj[1] -= tmp2 * ((-rjl[1] / rjlmag) - (rij[1] / rijmag));
    fj[2] -= tmp2 * ((-rjl[2] / rjlmag) - (rij[2] / rijmag));
    fl[0] -= tmp2 * (rjl[0] / rjlmag);
    fl[1] -= tmp2 * (rjl[1] / rjlmag);
    fl[2] -= tmp2 * (rjl[2] / rjlmag);
    // coordination forces: dwjl
    tmp2 = VA * .5 * (tmp * dwjl * g * libmExp(lamdaijl)) / rjlmag;
    fj[0] -= tmp2 * rjl[0];
    fj[1] -= tmp2 * rjl[1];
    fj[2] -= tmp2 * rjl[2];
    fl[0] += tmp2 * rjl[0];
    fl[1] += tmp2 * rjl[1];
    fl[2] += tmp2 * rjl[2];
    // PIJ forces
    tmp2 = VA * .5 * (tmp * dN2[ltype] * dwjl) / rjlmag;
    fj[0] -= tmp2 * rjl[0];
    fj[1] -= tmp2 * rjl[1];
    fj[2] -= tmp2 * rjl[2];
    fl[0] += tmp2 * rjl[0];
    fl[1] += tmp2 * rjl[1];
    fl[2] += tmp2 * rjl[2];
    // dgdN forces
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
  bondorderPijCross(core, atomi, atomj, itype, jtype, VA, -0.5 * (pij * pij * pij), -0.5 * (pji * pji * pji),
    dN2PIJbc, dN2PJIbc);

  // evaluate Nij conj
  const Nijconj = 1.0 + (NconjtmpI * NconjtmpI) + (NconjtmpJ * NconjtmpJ);
  const piRC = piRCSpline(core, NijC + NijH, NjiC + NjiH, Nijconj, itype, jtype, dN3);

  // piRC forces: i side
  for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
    const atomk = rlist[kk];
    if (atomk === atomj) continue;
    const ktype = mt[atomk];
    rik[0] = x[3 * atomi] - x[3 * atomk];
    rik[1] = x[3 * atomi + 1] - x[3 * atomk + 1];
    rik[2] = x[3 * atomi + 2] - x[3 * atomk + 2];
    const rikmag = Math.sqrt((rik[0] * rik[0]) + (rik[1] * rik[1]) + (rik[2] * rik[2]));
    const wik = sp(rikmag, rmin[pairIdx(itype, ktype)], rmax[pairIdx(itype, ktype)], rwik);
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
    // due to kronecker(ktype, 0) term in contribution to NconjtmpI and later Nijconj
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
        sp(rknmag, rmin[pairIdx(ktype, ntype)], rmax[pairIdx(ktype, ntype)], rwkn);
        const dwkn = rwkn.v;
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

  // piRC forces: j side
  for (let ll = rstart[atomj]; ll < rstart[atomj + 1]; ll++) {
    const atoml = rlist[ll];
    if (atoml === atomi) continue;
    const ltype = mt[atoml];
    rjl[0] = x[3 * atomj] - x[3 * atoml];
    rjl[1] = x[3 * atomj + 1] - x[3 * atoml + 1];
    rjl[2] = x[3 * atomj + 2] - x[3 * atoml + 2];
    const rjlmag = Math.sqrt((rjl[0] * rjl[0]) + (rjl[1] * rjl[1]) + (rjl[2] * rjl[2]));
    const wjl = sp(rjlmag, rmin[pairIdx(jtype, ltype)], rmax[pairIdx(jtype, ltype)], rwjl);
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
    // due to kronecker(ltype, 0) term in contribution to NconjtmpJ and later Nijconj
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
        sp(rlnmag, rmin[pairIdx(ltype, ntype)], rmax[pairIdx(ltype, ntype)], rwln);
        const dwln = rwln.v;
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

  // T_ij term (C-C bonds only)
  dN3[0] = 0.0;
  dN3[1] = 0.0;
  dN3[2] = 0.0;
  let Tij = 0.0;
  if (itype === 0 && jtype === 0) Tij = TijSpline(core, NijC + NijH, NjiC + NjiH, Nijconj, dN3);
  Etmp = 0.0;
  if (Math.abs(Tij) > TOL) {
    const atom2 = atomi;
    const atom3 = atomj;
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
      sp2(cos321, P.thmin, P.thmax, rdcut);
      const sin321 = Math.sqrt(1.0 - cos321 * cos321);
      if ((sin321 > TOL) && (r21mag > TOL)) {
        const sink2i = 1.0 / (sin321 * sin321);
        const rik2i = 1.0 / (r21mag * r21mag);
        let rr = (r23mag * r23mag) - (r21mag * r21mag);
        rjk[0] = r21[0] - r23[0];
        rjk[1] = r21[1] - r23[1];
        rjk[2] = r21[2] - r23[2];
        const rjk2 = (rjk[0] * rjk[0]) + (rjk[1] * rjk[1]) + (rjk[2] * rjk[2]);
        const rijrik = 2.0 * r23mag * r21mag;
        let rik2 = r21mag * r21mag;
        const dctik = (-rr + rjk2) / (rijrik * rik2);
        const dctij = (rr + rjk2) / (rijrik * r23mag * r23mag);
        const dctjk = -2.0 / rijrik;
        const w21 = sp(r21mag, rmin[pairIdx(itype, ktype)], rmaxp[pairIdx(itype, ktype)], rdw21);
        const dw21 = rdw21.v;
        const rijmag2 = r32mag;
        const rikmag2 = r21mag;
        const rij2 = r32mag * r32mag;
        rik2 = r21mag * r21mag;
        let costmp = 0.5 * (rij2 + rik2 - rjk2) / rijmag2 / rikmag2;
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
          const r34mag = Math.sqrt((r34[0] * r34[0]) + (r34[1] * r34[1]) + (r34[2] * r34[2]));
          let cos234 = (r32[0] * r34[0] + r32[1] * r34[1] + r32[2] * r34[2]) / (r32mag * r34mag);
          cos234 = Math.min(cos234, 1.0);
          cos234 = Math.max(cos234, -1.0);
          const sin234 = Math.sqrt(1.0 - cos234 * cos234);
          if ((sin234 > TOL) && (r34mag > TOL)) {
            const sinl2i = 1.0 / (sin234 * sin234);
            const rjl2i = 1.0 / (r34mag * r34mag);
            const w34 = sp(r34mag, rmin[pairIdx(jtype, ltype)], rmaxp[pairIdx(jtype, ltype)], rdw34);
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
            const rjlmag2 = r34mag;
            rjl2 = r34mag * r34mag;
            costmp = 0.5 * (rij2 + rjl2 - ril2) / rijmag2 / rjlmag2;
            const tspijl = sp2(costmp, P.thmin, P.thmax, rdtl);
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
            Etmp += ((1.0 - om1234 * om1234) * w21 * w34) * (1.0 - tspjik) * (1.0 - tspijl);
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
            tmp2 = VA * Tij * (1.0 - (om1234 * om1234)) * (1.0 - tspjik) * (1.0 - tspijl) * dw21 * w34 / r21mag;
            f2[0] -= tmp2 * r21[0];
            f2[1] -= tmp2 * r21[1];
            f2[2] -= tmp2 * r21[2];
            f1[0] += tmp2 * r21[0];
            f1[1] += tmp2 * r21[1];
            f1[2] += tmp2 * r21[2];
            tmp2 = VA * Tij * (1.0 - (om1234 * om1234)) * (1.0 - tspjik) * (1.0 - tspijl) * w21 * dw34 / r34mag;
            f3[0] -= tmp2 * r34[0];
            f3[1] -= tmp2 * r34[1];
            f3[2] -= tmp2 * r34[2];
            f4[0] += tmp2 * r34[0];
            f4[1] += tmp2 * r34[1];
            f4[2] += tmp2 * r34[2];
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

    // Tij forces now that we have Etmp: i side
    for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
      const atomk = rlist[kk];
      if (atomk === atomj) continue;
      const ktype = mt[atomk];
      rik[0] = x[3 * atomi] - x[3 * atomk];
      rik[1] = x[3 * atomi + 1] - x[3 * atomk + 1];
      rik[2] = x[3 * atomi + 2] - x[3 * atomk + 2];
      const rikmag = Math.sqrt((rik[0] * rik[0]) + (rik[1] * rik[1]) + (rik[2] * rik[2]));
      const wik = sp(rikmag, rmin[pairIdx(itype, ktype)], rmax[pairIdx(itype, ktype)], rwik);
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
          const ntype = mt[atomn];
          if (atomn === atomi) continue;
          rkn[0] = x[3 * atomk] - x[3 * atomn];
          rkn[1] = x[3 * atomk + 1] - x[3 * atomn + 1];
          rkn[2] = x[3 * atomk + 2] - x[3 * atomn + 2];
          const rknmag = Math.sqrt((rkn[0] * rkn[0]) + (rkn[1] * rkn[1]) + (rkn[2] * rkn[2]));
          sp(rknmag, rmin[pairIdx(ktype, ntype)], rmax[pairIdx(ktype, ntype)], rwkn);
          const dwkn = rwkn.v;
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

    // Tij forces: j side
    for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
      const atoml = rlist[ll];
      if (atoml === atomi) continue;
      const ltype = mt[atoml];
      rjl[0] = x[3 * atomj] - x[3 * atoml];
      rjl[1] = x[3 * atomj + 1] - x[3 * atoml + 1];
      rjl[2] = x[3 * atomj + 2] - x[3 * atoml + 2];
      const rjlmag = Math.sqrt((rjl[0] * rjl[0]) + (rjl[1] * rjl[1]) + (rjl[2] * rjl[2]));
      const wjl = sp(rjlmag, rmin[pairIdx(jtype, ltype)], rmax[pairIdx(jtype, ltype)], rwjl);
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
          const ntype = mt[atomn];
          if (atomn === atomj) continue;
          rln[0] = x[3 * atoml] - x[3 * atomn];
          rln[1] = x[3 * atoml + 1] - x[3 * atomn + 1];
          rln[2] = x[3 * atoml + 2] - x[3 * atomn + 2];
          const rlnmag = Math.sqrt((rln[0] * rln[0]) + (rln[1] * rln[1]) + (rln[2] * rln[2]));
          sp(rlnmag, rmin[pairIdx(ltype, ntype)], rmax[pairIdx(ltype, ntype)], rwln);
          const dwln = rwln.v;
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
  bij = (0.5 * (pij + pji)) + piRC + (Tij * Etmp);
  return bij;
};
