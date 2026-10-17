/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/reaxff_bonds.cpp (Bonds), src/REAXFF/reaxff_multi_body.cpp (Atom_Energy),
 * src/REAXFF/reaxff_valence_angles.cpp (Calculate_Theta, Calculate_dCos_Theta, Valence_Angles),
 * src/REAXFF/reaxff_torsion_angles.cpp (Calculate_Omega, Torsion_Angles), src/REAXFF/
 * reaxff_hydrogen_bonds.cpp (Hydrogen_Bonds), at commit b332f28. Authors credited in the C++
 * file headers: Hasan Metin Aktulga (Purdue University, now Lawrence Berkeley National
 * Laboratory); heavily modified and adapted for LAMMPS by the LAMMPS developers. Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Energies are accumulated into sys.en; forces into sys.ws.f (the LAMMPS sign: the pair style
 * applies -f). Per-atom energy and virial tallies are not made (compute pe/atom and stress/atom
 * are refused by the pair style).
 */
import { DEG2RAD, MIN_SINE, type RxSystem, type ThreeBodyList } from './core';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

const SQR = (x: number) => x * x;
const CUBE = (x: number) => x * x * x;

/** rvec helpers on flat arrays (a: array, o: offset). */
const dot3 = (a: Float64Array | number[], ao: number, b: Float64Array | number[], bo: number): number =>
  a[ao] * b[bo] + a[ao + 1] * b[bo + 1] + a[ao + 2] * b[bo + 2];

/** The 3-body list (thb_intrs) arrays are grown on demand. */
const newThreeBodyList = (nBondSlots: number, cap: number): ThreeBodyList => {
  const c = Math.max(cap, 1);
  return {
    start: new Int32Array(Math.max(nBondSlots, 1)), end: new Int32Array(Math.max(nBondSlots, 1)),
    thb: new Int32Array(c), pthb: new Int32Array(c), theta: new Float64Array(c),
    dcos_di: new Float64Array(3 * c), dcos_dj: new Float64Array(3 * c), dcos_dk: new Float64Array(3 * c),
    count: 0,
  };
};

/** Calculate_Theta: the angle between two bond vectors (vectors point from the central atom). */
export const calculateTheta = (dji: Float64Array, oji: number, dji_len: number, djk: Float64Array, ojk: number, djk_len: number): { theta: number; cos: number } => {
  let cos = dot3(dji, oji, djk, ojk) / (dji_len * djk_len);
  if (cos > 1.0) cos = 1.0;
  if (cos < -1.0) cos = -1.0;
  return { theta: Math.acos(cos), cos };
};

/**
 * Calculate_dCos_Theta: derivatives of cos(theta) with respect to the three atoms; written to
 * out[0..2] (di), out[3..5] (dj), out[6..8] (dk) of the given array at offset o.
 */
export const calculateDCosTheta = (
  dji: Float64Array, oji: number, dji_len: number, djk: Float64Array, ojk: number, djk_len: number,
  out: Float64Array, di: number, dj: number, dk: number,
): void => {
  const sqrDji = SQR(dji_len);
  const sqrDjk = SQR(djk_len);
  const invDists = 1.0 / (dji_len * djk_len);
  const invDists3 = CUBE(invDists);
  const dotDvecs = dot3(dji, oji, djk, ojk);
  const Cdot_inv3 = dotDvecs * invDists3;
  for (let t = 0; t < 3; t++) {
    out[di + t] = djk[ojk + t] * invDists - Cdot_inv3 * sqrDjk * dji[oji + t];
    out[dj + t] = -(djk[ojk + t] + dji[oji + t]) * invDists + Cdot_inv3 * (sqrDjk * dji[oji + t] + sqrDji * djk[ojk + t]);
    out[dk + t] = dji[oji + t] * invDists - Cdot_inv3 * sqrDji * djk[ojk + t];
  }
};

/* ------------------------------------------------------------------------------------------------ */

/** Bonds: the bond energy and its bond order derivatives, plus the stabilisation of triple bonds. */
export const bonds = (sys: RxSystem): void => {
  const { param, n: natoms, bonds: b, ws, en } = sys;
  const gp = param.gpL;
  const gp3 = gp[3], gp4 = gp[4], gp7 = gp[7], gp10 = gp[10];
  const gp37 = Math.trunc(gp[37]);
  const orig = sys.orig, x = sys.x, type = sys.type;
  for (let i = 0; i < natoms; i++) {
    const startI = b.start[i];
    const endI = b.end[i];
    for (let pj = startI; pj < endI; pj++) {
      const j = b.nbr[pj];
      if (orig[i] > orig[j]) continue;
      if (orig[i] === orig[j]) {
        if (x[3 * j + 2] < x[3 * i + 2]) continue;
        if (x[3 * j + 2] === x[3 * i + 2] && x[3 * j + 1] < x[3 * i + 1]) continue;
        if (x[3 * j + 2] === x[3 * i + 2] && x[3 * j + 1] === x[3 * i + 1] && x[3 * j] < x[3 * i]) continue;
      }
      const typeI = type[i], typeJ = type[j];
      if (typeI < 0 || typeJ < 0) continue;
      const sbpI = param.sbp[typeI];
      const sbpJ = param.sbp[typeJ];
      const twbp = param.tbp[typeI * param.numAtomTypes + typeJ];
      // the calculation of the constants
      const powBOs_be2 = b.BO_s[pj] === 0.0 ? 0.0 : libmPow(b.BO_s[pj], twbp.p_be2);
      const exp_be12 = libmExp(twbp.p_be1 * (1.0 - powBOs_be2));
      const CEbo = -twbp.De_s * exp_be12 * (1.0 - twbp.p_be1 * twbp.p_be2 * powBOs_be2);
      // the bond energy
      const ebond = -twbp.De_s * b.BO_s[pj] * exp_be12 - twbp.De_p * b.BO_pi[pj] - twbp.De_pp * b.BO_pi2[pj];
      en.e_bond += ebond;
      // derivatives of the bond orders
      b.Cdbo[pj] += CEbo;
      b.Cdbopi[pj] -= CEbo + twbp.De_p;
      b.Cdbopi2[pj] -= CEbo + twbp.De_pp;
      // stabilisation terminal triple bond
      if (b.BO[pj] >= 1.0) {
        if (gp37 === 2 || (sbpI.mass === 12.0 && sbpJ.mass === 15.999) || (sbpJ.mass === 12.0 && sbpI.mass === 15.999)) {
          const exphu = libmExp(-gp7 * SQR(b.BO[pj] - 2.5));
          const exphua1 = libmExp(-gp3 * (ws.total_bond_order[i] - b.BO[pj]));
          const exphub1 = libmExp(-gp3 * (ws.total_bond_order[j] - b.BO[pj]));
          const exphuov = libmExp(gp4 * (ws.Delta[i] + ws.Delta[j]));
          const hulpov = 1.0 / (1.0 + 25.0 * exphuov);
          const estriph = gp10 * exphu * hulpov * (exphua1 + exphub1);
          en.e_bond += estriph;
          const decobdbo = gp10 * exphu * hulpov * (exphua1 + exphub1) * (gp3 - 2.0 * gp7 * (b.BO[pj] - 2.5));
          const decobdboua = -gp10 * exphu * hulpov * (gp3 * exphua1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));
          const decobdboub = -gp10 * exphu * hulpov * (gp3 * exphub1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));
          b.Cdbo[pj] += decobdbo;
          ws.CdDelta[i] += decobdboua;
          ws.CdDelta[j] += decobdboub;
        }
      }
    }
  }
};

/** Atom_Energy: lone pair, C2 correction, over- and under-coordination energies and their derivatives. */
export const atomEnergy = (sys: RxSystem): void => {
  const { param, ws, bonds: b, en, ctrl, n } = sys;
  const gp = param.gpL;
  const p_lp3 = gp[5];
  const p_ovun3 = gp[32];
  const p_ovun4 = gp[31];
  const p_ovun6 = gp[6];
  const p_ovun7 = gp[8];
  const p_ovun8 = gp[9];
  const type = sys.type;
  for (let i = 0; i < n; i++) {
    const typeI = type[i];
    if (typeI < 0) continue;
    const sbpI = param.sbp[typeI];
    // lone-pair energy
    const p_lp2 = sbpI.p_lp2;
    const expvd2 = libmExp(-75 * ws.Delta_lp[i]);
    const inv_expvd2 = 1.0 / (1.0 + expvd2);
    let numbonds = 0;
    let e_lp = 0.0;
    for (let pj = b.start[i]; pj < b.end[i]; pj++) numbonds++;
    if (numbonds > 0 || ctrl.enobondsflag) {
      e_lp = p_lp2 * ws.Delta_lp[i] * inv_expvd2;
      en.e_lp += e_lp;
    }
    const dElp = p_lp2 * inv_expvd2 + 75 * p_lp2 * ws.Delta_lp[i] * expvd2 * SQR(inv_expvd2);
    const CElp = dElp * ws.dDelta_lp[i];
    if (numbonds > 0 || ctrl.enobondsflag) ws.CdDelta[i] += CElp;
    // correction for C2
    if (p_lp3 > 0.001 && sbpI.name === 'C') {
      for (let pj = b.start[i]; pj < b.end[i]; pj++) {
        const j = b.nbr[pj];
        const typeJ = type[j];
        if (typeJ < 0) continue;
        if (param.sbp[typeJ].name === 'C') {
          const Di = ws.Delta[i];
          const vov3 = b.BO[pj] - Di - 0.04 * libmPow(Di, 4.0);
          if (vov3 > 3.0) {
            const e_lph = p_lp3 * SQR(vov3 - 3.0);
            en.e_lp += e_lph;
            const deahu2dbo = 2.0 * p_lp3 * (vov3 - 3.0);
            const deahu2dsbo = 2.0 * p_lp3 * (vov3 - 3.0) * (-1.0 - 0.16 * libmPow(Di, 3.0));
            b.Cdbo[pj] += deahu2dbo;
            ws.CdDelta[i] += deahu2dsbo;
          }
        }
      }
    }
  }
  for (let i = 0; i < n; i++) {
    const typeI = type[i];
    if (typeI < 0) continue;
    const sbpI = param.sbp[typeI];
    // over-coordination energy (only for 1st-row elements)
    const dfvl = sbpI.mass > 21.0 ? 0.0 : 1.0;
    const p_ovun2 = sbpI.p_ovun2;
    let sum_ovun1 = 0, sum_ovun2 = 0;
    for (let pj = b.start[i]; pj < b.end[i]; pj++) {
      const j = b.nbr[pj];
      const typeJ = type[j];
      if (typeJ < 0) continue;
      const twbp = param.tbp[typeI * param.numAtomTypes + typeJ];
      sum_ovun1 += twbp.p_ovun1 * twbp.De_s * b.BO[pj];
      sum_ovun2 += (ws.Delta[j] - dfvl * ws.Delta_lp_temp[j]) * (b.BO_pi[pj] + b.BO_pi2[pj]);
    }
    const exp_ovun1 = p_ovun3 * libmExp(p_ovun4 * sum_ovun2);
    const inv_exp_ovun1 = 1.0 / (1 + exp_ovun1);
    const Delta_lpcorr = ws.Delta[i] - (dfvl * ws.Delta_lp_temp[i]) * inv_exp_ovun1;
    const exp_ovun2 = libmExp(p_ovun2 * Delta_lpcorr);
    const inv_exp_ovun2 = 1.0 / (1.0 + exp_ovun2);
    const DlpVi = 1.0 / (Delta_lpcorr + sbpI.valency + 1e-8);
    const CEover1 = Delta_lpcorr * DlpVi * inv_exp_ovun2;
    const e_ov = sum_ovun1 * CEover1;
    en.e_ov += e_ov;
    const CEover2 = sum_ovun1 * DlpVi * inv_exp_ovun2 * (1.0 - Delta_lpcorr * (DlpVi + p_ovun2 * exp_ovun2 * inv_exp_ovun2));
    const CEover3 = CEover2 * (1.0 - dfvl * ws.dDelta_lp[i] * inv_exp_ovun1);
    const CEover4 = CEover2 * (dfvl * ws.Delta_lp_temp[i]) * p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1);
    // under-coordination potential
    const p_ovun5 = sbpI.p_ovun5;
    const exp_ovun2n = 1.0 / exp_ovun2;
    const exp_ovun6 = libmExp(p_ovun6 * Delta_lpcorr);
    const exp_ovun8 = p_ovun7 * libmExp(p_ovun8 * sum_ovun2);
    const inv_exp_ovun2n = 1.0 / (1.0 + exp_ovun2n);
    const inv_exp_ovun8 = 1.0 / (1.0 + exp_ovun8);
    let numbonds = 0;
    let e_un = 0.0;
    for (let pj = b.start[i]; pj < b.end[i]; pj++) numbonds++;
    if (numbonds > 0 || ctrl.enobondsflag) {
      e_un = -p_ovun5 * (1.0 - exp_ovun6) * inv_exp_ovun2n * inv_exp_ovun8;
      en.e_un += e_un;
    }
    const CEunder1 = inv_exp_ovun2n * (p_ovun5 * p_ovun6 * exp_ovun6 * inv_exp_ovun8 + p_ovun2 * e_un * exp_ovun2n);
    const CEunder2 = -e_un * p_ovun8 * exp_ovun8 * inv_exp_ovun8;
    const CEunder3 = CEunder1 * (1.0 - dfvl * ws.dDelta_lp[i] * inv_exp_ovun1);
    const CEunder4 = CEunder1 * (dfvl * ws.Delta_lp_temp[i]) * p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1) + CEunder2;
    // forces
    ws.CdDelta[i] += CEover3;
    if (numbonds > 0 || ctrl.enobondsflag) ws.CdDelta[i] += CEunder3;
    for (let pj = b.start[i]; pj < b.end[i]; pj++) {
      const j = b.nbr[pj];
      const twbp = param.tbp[typeI * param.numAtomTypes + type[j]];
      b.Cdbo[pj] += CEover1 * twbp.p_ovun1 * twbp.De_s;
      ws.CdDelta[j] += CEover4 * (1.0 - dfvl * ws.dDelta_lp[j]) * (b.BO_pi[pj] + b.BO_pi2[pj]);
      b.Cdbopi[pj] += CEover4 * (ws.Delta[j] - dfvl * ws.Delta_lp_temp[j]);
      b.Cdbopi2[pj] += CEover4 * (ws.Delta[j] - dfvl * ws.Delta_lp_temp[j]);
      ws.CdDelta[j] += CEunder4 * (1.0 - dfvl * ws.dDelta_lp[j]) * (b.BO_pi[pj] + b.BO_pi2[pj]);
      b.Cdbopi[pj] += CEunder4 * (ws.Delta[j] - dfvl * ws.Delta_lp_temp[j]);
      b.Cdbopi2[pj] += CEunder4 * (ws.Delta[j] - dfvl * ws.Delta_lp_temp[j]);
    }
  }
};

/* ------------------------------------------------------------------------------------------------ */

/** Valence_Angles: the 3-body list and the valence, penalty and coalition energies. */
export const valenceAngles = (sys: RxSystem): void => {
  const { param, ws, bonds: b, en, ctrl, N, n } = sys;
  const gp = param.gpL;
  const p_val6 = gp[14];
  const p_val8 = gp[33];
  const p_val9 = gp[16];
  const p_val10 = gp[17];
  const type = sys.type;
  const nBondSlots = b.nbr.length;
  // capacity: for an atom j with nb bonds at most nb*(nb-1) three-body entries (see the bound in the notes)
  let cap = 1;
  for (let j = 0; j < N; j++) {
    const nb = b.end[j] - b.start[j];
    cap += nb * nb;
  }
  const thb = newThreeBodyList(nBondSlots, cap);
  sys.thb = thb;
  let numThb = 0;
  const thbPush = (): number => numThb++;
  for (let j = 0; j < N; j++) {
    const typeJ = type[j];
    if (typeJ < 0) continue;
    const startJ = b.start[j];
    const endJ = b.end[j];
    const p_val3 = param.sbp[typeJ].p_val3;
    const p_val5 = param.sbp[typeJ].p_val5;
    let SBOp = 0, prod_SBO = 1;
    for (let t = startJ; t < endJ; t++) {
      SBOp += b.BO_pi[t] + b.BO_pi2[t];
      let temp = SQR(b.BO[t]);
      temp *= temp;
      temp *= temp;
      prod_SBO *= libmExp(-temp);
    }
    let vlpadj: number, dSBO2: number;
    if (ws.vlpex[j] >= 0) {
      vlpadj = 0;
      dSBO2 = prod_SBO - 1;
    } else {
      vlpadj = ws.nlp[j];
      dSBO2 = (prod_SBO - 1) * (1 - p_val8 * ws.dDelta_lp[j]);
    }
    const SBO = SBOp + (1 - prod_SBO) * (-ws.Delta_boc[j] - p_val8 * vlpadj);
    const dSBO1 = -8 * prod_SBO * (ws.Delta_boc[j] + p_val8 * vlpadj);
    let SBO2: number, CSBO2: number;
    if (SBO <= 0) { SBO2 = 0; CSBO2 = 0; }
    else if (SBO > 0 && SBO <= 1) {
      SBO2 = libmPow(SBO, p_val9);
      CSBO2 = p_val9 * libmPow(SBO, p_val9 - 1);
    } else if (SBO > 1 && SBO < 2) {
      SBO2 = 2 - libmPow(2 - SBO, p_val9);
      CSBO2 = p_val9 * libmPow(2 - SBO, p_val9 - 1);
    } else { SBO2 = 2; CSBO2 = 0; }
    const expval6 = libmExp(p_val6 * ws.Delta_boc[j]);
    for (let pi = startJ; pi < endJ; pi++) {
      thb.start[pi] = numThb;
      const BOij = b.BO[pi];
      const BOA_ij = BOij - ctrl.thbCut;
      if (BOA_ij > 0.0 && (j < n || b.nbr[pi] < n)) {
        const i = b.nbr[pi];
        const typeI = type[i];
        for (let pk = startJ; pk < pi; pk++) {
          const startPk = thb.start[pk];
          const endPk = thb.end[pk];
          for (let t = startPk; t < endPk; t++) {
            if (thb.thb[t] === i) {
              const q = thbPush();
              thb.thb[q] = b.nbr[pk];
              thb.pthb[q] = pk;
              thb.theta[q] = thb.theta[t];
              for (let c = 0; c < 3; c++) {
                thb.dcos_di[3 * q + c] = thb.dcos_dk[3 * t + c];
                thb.dcos_dj[3 * q + c] = thb.dcos_dj[3 * t + c];
                thb.dcos_dk[3 * q + c] = thb.dcos_di[3 * t + c];
              }
              break;
            }
          }
        }
        for (let pk = pi + 1; pk < endJ; pk++) {
          const BOjk = b.BO[pk];
          const BOA_jk = BOjk - ctrl.thbCut;
          const k = b.nbr[pk];
          const typeK = type[k];
          const q = thbPush();
          const { theta } = calculateTheta(b.dvec, 3 * pi, b.d[pi], b.dvec, 3 * pk, b.d[pk]);
          // derivatives of cos(theta): written into the entry q's arrays
          const tmp = new Float64Array(9);
          calculateDCosTheta(b.dvec, 3 * pi, b.d[pi], b.dvec, 3 * pk, b.d[pk], tmp, 0, 3, 6);
          for (let c = 0; c < 3; c++) {
            thb.dcos_di[3 * q + c] = tmp[c];
            thb.dcos_dj[3 * q + c] = tmp[3 + c];
            thb.dcos_dk[3 * q + c] = tmp[6 + c];
          }
          thb.thb[q] = k;
          thb.pthb[q] = pk;
          thb.theta[q] = theta;
          let sinTheta = libmSin(theta);
          if (sinTheta < 1.0e-5) sinTheta = 1.0e-5;
          if (j < n && BOA_jk > 0.0 && BOij > ctrl.thbCut && BOjk > ctrl.thbCut && BOij * BOjk > ctrl.thbCutsq) {
            const thbh = param.thbp[(typeI * param.numAtomTypes + typeJ) * param.numAtomTypes + typeK];
            for (let cnt = 0; cnt < thbh.cnt; cnt++) {
              const prm = thbh.prm[cnt];
              if (Math.abs(prm.p_val1) > 0.001) {
                // ANGLE ENERGY
                const p_val1 = prm.p_val1;
                const p_val2 = prm.p_val2;
                const p_val4 = prm.p_val4;
                const p_val7 = prm.p_val7;
                const theta_00 = prm.theta_00;
                const exp3ij = libmExp(-p_val3 * libmPow(BOA_ij, p_val4));
                const f7_ij = 1.0 - exp3ij;
                const Cf7ij = p_val3 * p_val4 * libmPow(BOA_ij, p_val4 - 1.0) * exp3ij;
                const exp3jk = libmExp(-p_val3 * libmPow(BOA_jk, p_val4));
                const f7_jk = 1.0 - exp3jk;
                const Cf7jk = p_val3 * p_val4 * libmPow(BOA_jk, p_val4 - 1.0) * exp3jk;
                const expval7 = libmExp(-p_val7 * ws.Delta_boc[j]);
                const trm8 = 1.0 + expval6 + expval7;
                const f8_Dj = p_val5 - ((p_val5 - 1.0) * (2.0 + expval6) / trm8);
                const Cf8j = ((1.0 - p_val5) / SQR(trm8)) *
                  (p_val6 * expval6 * trm8 - (2.0 + expval6) * (p_val6 * expval6 - p_val7 * expval7));
                let theta_0 = 180.0 - theta_00 * (1.0 - libmExp(-p_val10 * (2.0 - SBO2)));
                theta_0 = DEG2RAD(theta_0);
                const expval2theta = libmExp(-p_val2 * SQR(theta_0 - theta));
                const expval12theta = p_val1 >= 0 ? p_val1 * (1.0 - expval2theta) : p_val1 * -expval2theta;
                const CEval1 = Cf7ij * f7_jk * f8_Dj * expval12theta;
                const CEval2 = Cf7jk * f7_ij * f8_Dj * expval12theta;
                const CEval3 = Cf8j * f7_ij * f7_jk * expval12theta;
                const CEval4 = -2.0 * p_val1 * p_val2 * f7_ij * f7_jk * f8_Dj * expval2theta * (theta_0 - theta);
                const Ctheta_0 = p_val10 * DEG2RAD(theta_00) * libmExp(-p_val10 * (2.0 - SBO2));
                const CEval5 = -CEval4 * Ctheta_0 * CSBO2;
                const CEval6 = CEval5 * dSBO1;
                const CEval7 = CEval5 * dSBO2;
                const CEval8 = -CEval4 / sinTheta;
                const e_ang = f7_ij * f7_jk * f8_Dj * expval12theta;
                en.e_ang += e_ang;
                // PENALTY ENERGY
                const p_pen1 = prm.p_pen1;
                const p_pen2 = gp[19];
                const p_pen3 = gp[20];
                const p_pen4 = gp[21];
                const exp_pen2ij = libmExp(-p_pen2 * SQR(BOA_ij - 2.0));
                const exp_pen2jk = libmExp(-p_pen2 * SQR(BOA_jk - 2.0));
                const exp_pen3 = libmExp(-p_pen3 * ws.Delta[j]);
                const exp_pen4 = libmExp(p_pen4 * ws.Delta[j]);
                const trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                const f9_Dj = (2.0 + exp_pen3) / trm_pen34;
                const Cf9j = (-p_pen3 * exp_pen3 * trm_pen34 - (2.0 + exp_pen3) * (-p_pen3 * exp_pen3 + p_pen4 * exp_pen4)) / SQR(trm_pen34);
                const e_pen = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                en.e_pen += e_pen;
                const CEpen1 = e_pen * Cf9j / f9_Dj;
                const temp = -2.0 * p_pen2 * e_pen;
                const CEpen2 = temp * (BOA_ij - 2.0);
                const CEpen3 = temp * (BOA_jk - 2.0);
                // COALITION ENERGY
                const p_coa1 = prm.p_coa1;
                const p_coa2 = gp[2];
                const p_coa3 = gp[38];
                const p_coa4 = gp[30];
                const exp_coa2 = libmExp(p_coa2 * ws.Delta_val[j]);
                const e_coa = p_coa1 / (1.0 + exp_coa2) *
                  libmExp(-p_coa3 * SQR(ws.total_bond_order[i] - BOA_ij)) *
                  libmExp(-p_coa3 * SQR(ws.total_bond_order[k] - BOA_jk)) *
                  libmExp(-p_coa4 * SQR(BOA_ij - 1.5)) *
                  libmExp(-p_coa4 * SQR(BOA_jk - 1.5));
                en.e_coa += e_coa;
                const CEcoa1 = -2 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                const CEcoa2 = -2 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                const CEcoa3 = -p_coa2 * exp_coa2 * e_coa / (1 + exp_coa2);
                const CEcoa4 = -2 * p_coa3 * (ws.total_bond_order[i] - BOA_ij) * e_coa;
                const CEcoa5 = -2 * p_coa3 * (ws.total_bond_order[k] - BOA_jk) * e_coa;
                // FORCES
                b.Cdbo[pi] += CEval1 + CEpen2 + (CEcoa1 - CEcoa4);
                b.Cdbo[pk] += CEval2 + CEpen3 + (CEcoa2 - CEcoa5);
                ws.CdDelta[j] += (CEval3 + CEval7) + CEpen1 + CEcoa3;
                ws.CdDelta[i] += CEcoa4;
                ws.CdDelta[k] += CEcoa5;
                for (let t = startJ; t < endJ; t++) {
                  const BOjt = b.BO[t];
                  const temp2 = CUBE(BOjt);
                  const pBOjt7 = temp2 * temp2 * BOjt;
                  b.Cdbo[t] += CEval6 * pBOjt7;
                  b.Cdbopi[t] += CEval5;
                  b.Cdbopi2[t] += CEval5;
                }
                for (let c = 0; c < 3; c++) {
                  ws.f[3 * i + c] += CEval8 * thb.dcos_di[3 * q + c];
                  ws.f[3 * j + c] += CEval8 * thb.dcos_dj[3 * q + c];
                  ws.f[3 * k + c] += CEval8 * thb.dcos_dk[3 * q + c];
                }
              }
            }
          }
        }
      }
      thb.end[pi] = numThb;
    }
  }
  thb.count = numThb;
};


/* ------------------------------------------------------------------------------------------------ */

/** The dcos_theta vectors of one three-body entry (p_ijk->dcos_di / dcos_dj / dcos_dk at index q). */
interface DCos { di: Float64Array; dj: Float64Array; dk: Float64Array; q: number }

/**
 * Calculate_Omega: the torsion angle omega of the chain i-j-k-l and its derivatives with respect to
 * the four atoms, written to out[0..2] (i), out[3..5] (j), out[6..8] (k), out[9..11] (l).
 */
const calculateOmega = (
  dij: Float64Array, oij: number, r_ij: number,
  djk: Float64Array, ojk: number, r_jk: number,
  dkl: Float64Array, okl: number, r_kl: number,
  dli: Float64Array, r_li: number,
  thetaIJK: number, thetaJKL: number,
  pijk: DCos, pjkl: DCos,
  out: Float64Array,
): number => {
  let sin_ijk = libmSin(thetaIJK);
  const cos_ijk = libmCos(thetaIJK);
  let sin_jkl = libmSin(thetaJKL);
  const cos_jkl = libmCos(thetaJKL);
  if (sin_ijk >= 0 && sin_ijk <= MIN_SINE) sin_ijk = MIN_SINE;
  else if (sin_ijk <= 0 && sin_ijk >= -MIN_SINE) sin_ijk = -MIN_SINE;
  if (sin_jkl >= 0 && sin_jkl <= MIN_SINE) sin_jkl = MIN_SINE;
  else if (sin_jkl <= 0 && sin_jkl >= -MIN_SINE) sin_jkl = -MIN_SINE;
  // omega
  const unnorm_cos_omega = -dot3(dij, oij, djk, ojk) * dot3(djk, ojk, dkl, okl) + SQR(r_jk) * dot3(dij, oij, dkl, okl);
  // cross_jk_kl = dvec_jk x dvec_kl
  const cx = djk[ojk + 1] * dkl[okl + 2] - djk[ojk + 2] * dkl[okl + 1];
  const cy = djk[ojk + 2] * dkl[okl] - djk[ojk] * dkl[okl + 2];
  const cz = djk[ojk] * dkl[okl + 1] - djk[ojk + 1] * dkl[okl];
  const unnorm_sin_omega = -r_jk * (dij[oij] * cx + dij[oij + 1] * cy + dij[oij + 2] * cz);
  const omega = Math.atan2(unnorm_sin_omega, unnorm_cos_omega);
  const htra = r_ij + cos_ijk * (r_kl * cos_jkl - r_jk);
  const htrb = r_jk - r_ij * cos_ijk - r_kl * cos_jkl;
  const htrc = r_kl + cos_jkl * (r_ij * cos_ijk - r_jk);
  const hthd = r_ij * sin_ijk * (r_jk - r_kl * cos_jkl);
  const hthe = r_kl * sin_jkl * (r_jk - r_ij * cos_ijk);
  const hnra = r_kl * sin_ijk * sin_jkl;
  const hnrc = r_ij * sin_ijk * sin_jkl;
  const hnhd = r_ij * r_kl * cos_ijk * sin_jkl;
  const hnhe = r_ij * r_kl * sin_ijk * cos_jkl;
  const tel = SQR(r_ij) + SQR(r_jk) + SQR(r_kl) - SQR(r_li) -
    2.0 * (r_ij * r_jk * cos_ijk - r_ij * r_kl * cos_ijk * cos_jkl + r_jk * r_kl * cos_jkl);
  const poem = 2.0 * r_ij * r_kl * sin_ijk * sin_jkl;
  let arg = tel / poem;
  if (arg > 1.0) arg = 1.0;
  if (arg < -1.0) arg = -1.0;
  const a1 = (htra - arg * hnra) / r_ij;
  const a2 = (hthd - arg * hnhd) / sin_ijk;
  const a3 = (hthe - arg * hnhe) / sin_jkl;
  const b1 = (htrc - arg * hnrc) / r_kl;
  const s = 2.0 / poem;
  for (let t = 0; t < 3; t++) {
    // dcos_omega_di
    out[t] = (a1 * dij[oij + t] - dli[t] - a2 * pijk.dk[pijk.q + t]) * s;
    // dcos_omega_dj
    out[3 + t] = (-a1 * dij[oij + t] - (htrb / r_jk) * djk[ojk + t] - a2 * pijk.dj[pijk.q + t] -
      a3 * pjkl.di[pjkl.q + t]) * s;
    // dcos_omega_dk
    out[6 + t] = (-b1 * dkl[okl + t] + (htrb / r_jk) * djk[ojk + t] - a2 * pijk.di[pijk.q + t] -
      a3 * pjkl.dj[pjkl.q + t]) * s;
    // dcos_omega_dl
    out[9 + t] = (b1 * dkl[okl + t] + dli[t] - a3 * pjkl.dk[pjkl.q + t]) * s;
  }
  return omega;
};

/** Torsion_Angles: the torsion energy (V1, V2, V3 terms) and the 4-body conjugation energy. */
export const torsionAngles = (sys: RxSystem): void => {
  const { param, ws, bonds: b, en, ctrl, n } = sys;
  const gp = param.gpL;
  const p_tor2 = gp[23];
  const p_tor3 = gp[24];
  const p_tor4 = gp[25];
  const p_cot2 = gp[27];
  const thb = sys.thb;
  const type = sys.type;
  const orig = sys.orig;
  const x = sys.x;
  const thbCut = ctrl.thbCut;
  const dvecLI = new Float64Array(3);
  const dcosOmega = new Float64Array(12);
  for (let j = 0; j < n; j++) {
    const typeJ = type[j];
    const Delta_j = ws.Delta_boc[j];
    const startJ = b.start[j];
    const endJ = b.end[j];
    for (let pk = startJ; pk < endJ; pk++) {
      const k = b.nbr[pk];
      const BOA_jk = b.BO[pk] - thbCut;
      if (orig[j] > orig[k]) continue;
      if (orig[j] === orig[k]) {
        if (x[3 * k + 2] < x[3 * j + 2]) continue;
        if (x[3 * k + 2] === x[3 * j + 2] && x[3 * k + 1] < x[3 * j + 1]) continue;
        if (x[3 * k + 2] === x[3 * j + 2] && x[3 * k + 1] === x[3 * j + 1] && x[3 * k] < x[3 * j]) continue;
      }
      if (!(b.BO[pk] > thbCut && thb.end[pk] - thb.start[pk] > 0)) continue;
      const pj = b.sym[pk];
      if (!(thb.end[pj] - thb.start[pj] > 0)) continue;
      const typeK = type[k];
      const Delta_k = ws.Delta_boc[k];
      const r_jk = b.d[pk];
      const startPk = thb.start[pk], endPk = thb.end[pk];
      const startPj = thb.start[pj], endPj = thb.end[pj];
      const exp_tor2_jk = libmExp(-p_tor2 * BOA_jk);
      const exp_cot2_jk = libmExp(-p_cot2 * SQR(BOA_jk - 1.5));
      const exp_tor3_DjDk = libmExp(-p_tor3 * (Delta_j + Delta_k));
      const exp_tor4_DjDk = libmExp(p_tor4 * (Delta_j + Delta_k));
      const exp_tor34_inv = 1.0 / (1.0 + exp_tor3_DjDk + exp_tor4_DjDk);
      const f11_DjDk = (2.0 + exp_tor3_DjDk) * exp_tor34_inv;
      for (let pi = startPk; pi < endPk; pi++) {
        const pbondIJ = thb.pthb[pi];
        const BOij = b.BO[pbondIJ];
        if (!(BOij > thbCut)) continue;
        const i = thb.thb[pi];
        const typeI = type[i];
        const r_ij = b.d[pbondIJ];
        const BOA_ij = BOij - thbCut;
        const theta_ijk = thb.theta[pi];
        const sin_ijk = libmSin(theta_ijk);
        const cos_ijk = libmCos(theta_ijk);
        const exp_tor2_ij = libmExp(-p_tor2 * BOA_ij);
        const exp_cot2_ij = libmExp(-p_cot2 * SQR(BOA_ij - 1.5));
        let tan_ijk_i: number;
        if (sin_ijk >= 0 && sin_ijk <= MIN_SINE) tan_ijk_i = cos_ijk / MIN_SINE;
        else if (sin_ijk <= 0 && sin_ijk >= -MIN_SINE) tan_ijk_i = cos_ijk / -MIN_SINE;
        else tan_ijk_i = cos_ijk / sin_ijk;
        for (let pl = startPj; pl < endPj; pl++) {
          const l = thb.thb[pl];
          const plk = thb.pthb[pl];
          const BOkl = b.BO[plk];
          const typeL = type[l];
          const fbh = param.fbp[((typeI * param.numAtomTypes + typeJ) * param.numAtomTypes + typeK) * param.numAtomTypes + typeL];
          if (!(i !== l && fbh.cnt && BOkl > thbCut && BOij * b.BO[pk] * BOkl > thbCut)) continue;
          const fbp = fbh.prm[0];
          const r_kl = b.d[plk];
          const BOA_kl = BOkl - thbCut;
          const theta_jkl = thb.theta[pl];
          const sin_jkl = libmSin(theta_jkl);
          const cos_jkl = libmCos(theta_jkl);
          let tan_jkl_i: number;
          if (sin_jkl >= 0 && sin_jkl <= MIN_SINE) tan_jkl_i = cos_jkl / MIN_SINE;
          else if (sin_jkl <= 0 && sin_jkl >= -MIN_SINE) tan_jkl_i = cos_jkl / -MIN_SINE;
          else tan_jkl_i = cos_jkl / sin_jkl;
          for (let c = 0; c < 3; c++) dvecLI[c] = x[3 * i + c] - x[3 * l + c];
          const r_li = Math.sqrt(dot3(dvecLI, 0, dvecLI, 0));
          // omega and its derivative
          const omega = calculateOmega(
            b.dvec, 3 * pbondIJ, r_ij, b.dvec, 3 * pk, r_jk, b.dvec, 3 * plk, r_kl, dvecLI, r_li,
            theta_ijk, theta_jkl,
            { di: thb.dcos_di, dj: thb.dcos_dj, dk: thb.dcos_dk, q: 3 * pi },
            { di: thb.dcos_di, dj: thb.dcos_dj, dk: thb.dcos_dk, q: 3 * pl },
            dcosOmega,
          );
          const cos_omega = libmCos(omega);
          const cos2omega = libmCos(2.0 * omega);
          const cos3omega = libmCos(3.0 * omega);
          // torsion energy
          const exp_tor1 = libmExp(fbp.p_tor1 * SQR(2.0 - b.BO_pi[pk] - f11_DjDk));
          const exp_tor2_kl = libmExp(-p_tor2 * BOA_kl);
          const exp_cot2_kl = libmExp(-p_cot2 * SQR(BOA_kl - 1.5));
          const fn10 = (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * (1.0 - exp_tor2_kl);
          const CV = 0.5 * (fbp.V1 * (1.0 + cos_omega) + fbp.V2 * exp_tor1 * (1.0 - cos2omega) + fbp.V3 * (1.0 + cos3omega));
          const e_tor = fn10 * sin_ijk * sin_jkl * CV;
          en.e_tor += e_tor;
          const dfn11 = (-p_tor3 * exp_tor3_DjDk + (p_tor3 * exp_tor3_DjDk - p_tor4 * exp_tor4_DjDk) *
            (2.0 + exp_tor3_DjDk) * exp_tor34_inv) * exp_tor34_inv;
          const CEtors1 = sin_ijk * sin_jkl * CV;
          const CEtors2 = -fn10 * 2.0 * fbp.p_tor1 * fbp.V2 * exp_tor1 * (2.0 - b.BO_pi[pk] - f11_DjDk) *
            (1.0 - SQR(cos_omega)) * sin_ijk * sin_jkl;
          const CEtors3 = CEtors2 * dfn11;
          const CEtors4 = CEtors1 * p_tor2 * exp_tor2_ij * (1.0 - exp_tor2_jk) * (1.0 - exp_tor2_kl);
          const CEtors5 = CEtors1 * p_tor2 * (1.0 - exp_tor2_ij) * exp_tor2_jk * (1.0 - exp_tor2_kl);
          const CEtors6 = CEtors1 * p_tor2 * (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * exp_tor2_kl;
          const cmn = -fn10 * CV;
          const CEtors7 = cmn * sin_jkl * tan_ijk_i;
          const CEtors8 = cmn * sin_ijk * tan_jkl_i;
          const CEtors9 = fn10 * sin_ijk * sin_jkl *
            (0.5 * fbp.V1 - 2.0 * fbp.V2 * exp_tor1 * cos_omega + 1.5 * fbp.V3 * (cos2omega + 2.0 * SQR(cos_omega)));
          // 4-body conjugation energy
          const fn12 = exp_cot2_ij * exp_cot2_jk * exp_cot2_kl;
          const e_con = fbp.p_cot1 * fn12 * (1.0 + (SQR(cos_omega) - 1.0) * sin_ijk * sin_jkl);
          en.e_con += e_con;
          const Cconj = -2.0 * fn12 * fbp.p_cot1 * p_cot2 * (1.0 + (SQR(cos_omega) - 1.0) * sin_ijk * sin_jkl);
          const CEconj1 = Cconj * (BOA_ij - 1.5);
          const CEconj2 = Cconj * (BOA_jk - 1.5);
          const CEconj3 = Cconj * (BOA_kl - 1.5);
          const CEconj4 = -fbp.p_cot1 * fn12 * (SQR(cos_omega) - 1.0) * sin_jkl * tan_ijk_i;
          const CEconj5 = -fbp.p_cot1 * fn12 * (SQR(cos_omega) - 1.0) * sin_ijk * tan_jkl_i;
          const CEconj6 = 2.0 * fbp.p_cot1 * fn12 * cos_omega * sin_ijk * sin_jkl;
          // forces
          b.Cdbopi[pk] += CEtors2;
          ws.CdDelta[j] += CEtors3;
          ws.CdDelta[k] += CEtors3;
          b.Cdbo[pbondIJ] += CEtors4 + CEconj1;
          b.Cdbo[pk] += CEtors5 + CEconj2;
          b.Cdbo[plk] += CEtors6 + CEconj3;
          const dI = 3 * pi, dJ = 3 * pl;
          const c7 = CEtors7 + CEconj4, c8 = CEtors8 + CEconj5, c9 = CEtors9 + CEconj6;
          for (let c = 0; c < 3; c++) {
            // dcos_theta_ijk
            ws.f[3 * i + c] += c7 * thb.dcos_dk[dI + c];
            ws.f[3 * j + c] += c7 * thb.dcos_dj[dI + c];
            ws.f[3 * k + c] += c7 * thb.dcos_di[dI + c];
            // dcos_theta_jkl
            ws.f[3 * j + c] += c8 * thb.dcos_di[dJ + c];
            ws.f[3 * k + c] += c8 * thb.dcos_dj[dJ + c];
            ws.f[3 * l + c] += c8 * thb.dcos_dk[dJ + c];
            // dcos_omega
            ws.f[3 * i + c] += c9 * dcosOmega[c];
            ws.f[3 * j + c] += c9 * dcosOmega[3 + c];
            ws.f[3 * k + c] += c9 * dcosOmega[6 + c];
            ws.f[3 * l + c] += c9 * dcosOmega[9 + c];
          }
        }
      }
    }
  }
};

/** Hydrogen_Bonds: the donor-H ... acceptor energy over the hydrogen bond list. */
export const hydrogenBonds = (sys: RxSystem): void => {
  const { param, ws, bonds: b, en, n, hb, far } = sys;
  const type = sys.type;
  const orig = sys.orig;
  const hblist: number[] = [];
  const dvecJK = new Float64Array(3);
  const tmp = new Float64Array(9);
  for (let j = 0; j < n; j++) {
    const typeJ = type[j];
    if (typeJ < 0) continue;
    if (param.sbp[typeJ].p_hbond !== 1) continue;
    const hbStartJ = hb.start[sys.Hindex[j]];
    const hbEndJ = hb.end[sys.Hindex[j]];
    hblist.length = 0;
    for (let pi = b.start[j]; pi < b.end[j]; pi++) {
      const i = b.nbr[pi];
      const typeI = type[i];
      if (typeI < 0) continue;
      if (param.sbp[typeI].p_hbond === 2 && b.BO[pi] >= 1e-2) hblist.push(pi);
    }
    for (let pk = hbStartJ; pk < hbEndJ; pk++) {
      const k = hb.nbr[pk];
      const typeK = type[k];
      if (typeK < 0) continue;
      const fp = hb.ptr[pk];
      const r_jk = far.d[fp];
      const scl = hb.scl[pk];
      for (let c = 0; c < 3; c++) dvecJK[c] = scl * far.dvec[3 * fp + c];
      for (const pi of hblist) {
        const i = b.nbr[pi];
        if (orig[i] === orig[k]) continue;
        const typeI = type[i];
        if (typeI < 0) continue;
        const hbp = param.hbp[(typeI * param.numAtomTypes + typeJ) * param.numAtomTypes + typeK];
        if (hbp.r0_hb <= 0.0) continue;
        const { theta, cos: cosTheta } = calculateTheta(b.dvec, 3 * pi, b.d[pi], dvecJK, 0, r_jk);
        // the derivative of cos(theta): di, dj, dk
        calculateDCosTheta(b.dvec, 3 * pi, b.d[pi], dvecJK, 0, r_jk, tmp, 0, 3, 6);
        // hydrogen bond energy
        const sin_theta2 = libmSin(theta / 2.0);
        let sin_xhz4 = SQR(sin_theta2);
        sin_xhz4 *= sin_xhz4;
        const cos_xhz1 = 1.0 - cosTheta;
        const exp_hb2 = libmExp(-hbp.p_hb2 * b.BO[pi]);
        const exp_hb3 = libmExp(-hbp.p_hb3 * (hbp.r0_hb / r_jk + r_jk / hbp.r0_hb - 2.0));
        const e_hb = hbp.p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
        en.e_hb += e_hb;
        const CEhb1 = hbp.p_hb1 * hbp.p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
        const CEhb2 = -hbp.p_hb1 / 2.0 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
        const CEhb3 = -hbp.p_hb3 * (-hbp.r0_hb / SQR(r_jk) + 1.0 / hbp.r0_hb) * e_hb;
        // hydrogen bond forces
        b.Cdbo[pi] += CEhb1;
        for (let c = 0; c < 3; c++) {
          // dcos terms
          ws.f[3 * i + c] += CEhb2 * tmp[c];
          ws.f[3 * j + c] += CEhb2 * tmp[3 + c];
          ws.f[3 * k + c] += CEhb2 * tmp[6 + c];
          // dr terms
          ws.f[3 * j + c] += (-CEhb3 / r_jk) * dvecJK[c];
          ws.f[3 * k + c] += (+CEhb3 / r_jk) * dvecJK[c];
        }
      }
    }
  }
};
