/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REAXFF/reaxff_nonbonded.cpp (Compute_Polarization_Energy, vdW_Coulomb_Energy,
 * Tabulated_vdW_Coulomb_Energy, LR_vdW_Coulomb), at commit b332f28. Authors credited in the C++
 * file headers: Hasan Metin Aktulga (Purdue University, now Lawrence Berkeley National Laboratory);
 * heavily modified and adapted for LAMMPS by the LAMMPS developers. Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The far neighbor pairs of owned atoms form a half list (see core.ts buildFarList); a ghost
 * partner is taken on exactly one of its two images by the flag rule below, so each physical
 * pair contributes once.
 */
import { StyleError } from '../../engine/force/types';
import type { RxSystem } from './core';
import { C_ele, EV_to_KCALpMOL, KCALpMOL_to_EV } from './core';
import type { ControlParams, ReaxParam, TwoBody } from './ffield';
import type { LookupTables } from './lookup';
import { exp as libmExp } from '../libm/exp';

const SMALL = 0.0001;

/**
 * ACKS2 kinetic-energy terms of the far pairs (the block of vdW_Coulomb_Energy and of
 * Tabulated_vdW_Coulomb_Energy under "if (system->acks2_flag)"): the bond softness of each pair with
 * the cutoff xcut = (bcut_i + bcut_j)/2 gives an energy -1/2 k bs (s_i - s_j)^2 and its forces.
 */
const acks2Pairs = (sys: RxSystem): void => {
  const s = sys.acks2s;
  if (!s) return;
  const { far, ws, en, n, N } = sys;
  const gp34 = sys.param.gpL[34];
  for (let i = 0; i < n; i++) {
    if (sys.type[i] < 0) continue;
    for (let pj = far.start[i]; pj < far.end[i]; pj++) {
      const j = far.nbr[pj];
      if (sys.type[j] < 0) continue;
      const xcut = 0.5 * (sys.param.sbp[sys.type[i]].bcut_acks2 + sys.param.sbp[sys.type[j]].bcut_acks2);
      if (!pairFlag(sys, i, j, pj, xcut)) continue;
      const d = far.d[pj] / xcut;
      const bond_softness = gp34 * Math.pow(d, 3.0) * Math.pow(1.0 - d, 6.0);
      if (!(bond_softness > 0.0)) continue;
      const effpot_diff = s[N + i] - s[N + j];
      const e_ele = -0.5 * KCALpMOL_to_EV * bond_softness * effpot_diff * effpot_diff;
      en.e_ele += e_ele;
      let d_bond_softness = gp34 * (3.0 / xcut) * Math.pow(d, 2.0) * Math.pow(1.0 - d, 5.0) * (1.0 - 3.0 * d);
      d_bond_softness = -0.5 * d_bond_softness * effpot_diff * effpot_diff;
      d_bond_softness = (KCALpMOL_to_EV * d_bond_softness) / far.d[pj];
      for (let k = 0; k < 3; k++) {
        ws.f[3 * i + k] += -d_bond_softness * far.dvec[3 * pj + k];
        ws.f[3 * j + k] += d_bond_softness * far.dvec[3 * pj + k];
      }
    }
  }
};

/** Compute_Polarization_Energy: the charge-dependent electronegativity energy of each owned atom. */
export const polarizationEnergy = (sys: RxSystem): void => {
  sys.en.e_pol = 0.0;
  for (let i = 0; i < sys.n; i++) {
    const ti = sys.type[i];
    if (ti < 0) continue;
    const q = sys.q[i];
    const s = sys.param.sbp[ti];
    let en_tmp = KCALpMOL_to_EV * (s.chi * q + (s.eta / 2.0) * q * q);
    if (sys.acks2s) {
      // energy due to coupling with kinetic energy potential (ACKS2)
      en_tmp += KCALpMOL_to_EV * q * sys.acks2s[sys.N + i];
    }
    sys.en.e_pol += en_tmp;
  }
};

/**
 * Flag rule of vdW_Coulomb_Energy: a far pair of owned i is used when j is owned, or when j is a
 * periodic image of another atom with a larger id, or the same id and the displacement is positive
 * in z, then y, then x (exactly one of two mirrored images passes).
 */
const pairFlag = (sys: RxSystem, i: number, j: number, p: number, cut: number): boolean => {
  const far = sys.far;
  if (!(far.d[p] <= cut)) return false;
  if (j < sys.n) return true;
  const oi = sys.orig[i], oj = sys.orig[j];
  if (oi < oj) return true;
  if (oi === oj) {
    const dz = far.dvec[3 * p + 2], dy = far.dvec[3 * p + 1], dx = far.dvec[3 * p];
    if (dz > SMALL) return true;
    if (Math.abs(dz) < SMALL) {
      if (dy > SMALL) return true;
      if (Math.abs(dy) < SMALL && dx > SMALL) return true;
    }
  }
  return false;
};

/** Taper and its derivative at r (the seventh-order polynomial of core.ts taperCoefficients). */
const taperAt = (T: Float64Array, r: number): { Tap: number; dTap: number } => {
  let Tap = T[7] * r + T[6];
  Tap = Tap * r + T[5];
  Tap = Tap * r + T[4];
  Tap = Tap * r + T[3];
  Tap = Tap * r + T[2];
  Tap = Tap * r + T[1];
  Tap = Tap * r + T[0];
  let dTap = 7 * T[7] * r + 6 * T[6];
  dTap = dTap * r + 5 * T[5];
  dTap = dTap * r + 4 * T[4];
  dTap = dTap * r + 3 * T[3];
  dTap = dTap * r + 2 * T[2];
  dTap += T[1] / r;
  return { Tap, dTap };
};

/** Shared vdW (shielding, inner wall, lg) and Coulomb evaluation of one pair (vdW_Coulomb_Energy body). */
const directPair = (
  param: ReaxParam, ctrl: ControlParams, Tap: number, dTap: number, r_ij: number, twbp: TwoBody,
  gamma_w: number, p_vdW1: number,
): { e_vdW: number; e_core: number; e_lg: number; CEvd: number; e_ele0: number; CEclmb: number } => {
  const p_vdW1i = 1.0 / p_vdW1;
  let e_vdW = 0, e_core = 0, e_lg = 0, CEvd = 0;
  void gamma_w;
  if (param.vdwType === 1 || param.vdwType === 3) {
    // shielding
    const powr_vdW1 = Math.pow(r_ij, p_vdW1);
    const powgi_vdW1 = Math.pow(1.0 / twbp.gamma_w, p_vdW1);
    const fn13 = Math.pow(powr_vdW1 + powgi_vdW1, p_vdW1i);
    const exp1 = libmExp(twbp.alpha * (1.0 - fn13 / twbp.r_vdW));
    const exp2 = libmExp(0.5 * twbp.alpha * (1.0 - fn13 / twbp.r_vdW));
    e_vdW = twbp.D * (exp1 - 2.0 * exp2);
    const dfn13 = Math.pow(powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0) * Math.pow(r_ij, p_vdW1 - 2.0);
    CEvd = dTap * e_vdW - Tap * twbp.D * (twbp.alpha / twbp.r_vdW) * (exp1 - exp2) * dfn13;
  } else {
    // no shielding
    const exp1 = libmExp(twbp.alpha * (1.0 - r_ij / twbp.r_vdW));
    const exp2 = libmExp(0.5 * twbp.alpha * (1.0 - r_ij / twbp.r_vdW));
    e_vdW = twbp.D * (exp1 - 2.0 * exp2);
    CEvd = dTap * e_vdW - Tap * twbp.D * (twbp.alpha / twbp.r_vdW) * (exp1 - exp2) / r_ij;
  }
  if (param.vdwType === 2 || param.vdwType === 3) {
    // inner wall
    e_core = twbp.ecore * libmExp(twbp.acore * (1.0 - r_ij / twbp.rcore));
    const de_core = -(twbp.acore / twbp.rcore) * e_core;
    CEvd += dTap * e_core + Tap * de_core / r_ij;
    // lg correction, only if lgvdw is yes (refused by the pair style: lgflag is always 0 here)
    if (ctrl.lgflag) {
      const r_ij5 = Math.pow(r_ij, 5.0), r_ij6 = Math.pow(r_ij, 6.0), re6 = Math.pow(twbp.lgre, 6.0);
      e_lg = -(twbp.lgcij / (r_ij6 + re6));
      const de_lg = -6.0 * e_lg * r_ij5 / (r_ij6 + re6);
      CEvd += dTap * e_lg + Tap * de_lg / r_ij;
    }
  }
  // Coulomb
  const dr3gamij_1 = r_ij * r_ij * r_ij + twbp.gamma;
  const dr3gamij_3 = Math.pow(dr3gamij_1, 0.33333333333333);
  const tmp = Tap / dr3gamij_3;
  const e_ele0 = C_ele * tmp;
  const CEclmb = C_ele * (dTap - Tap * r_ij / dr3gamij_1) / dr3gamij_3;
  return { e_vdW, e_core, e_lg, CEvd, e_ele0, CEclmb };
};

/** vdW_Coulomb_Energy: direct evaluation of the van der Waals and Coulomb terms over the far pairs. */
export const vdwCoulombEnergy = (sys: RxSystem): void => {
  const { param, ctrl, ws, far, en, n } = sys;
  const p_vdW1 = param.gpL[28];
  for (let i = 0; i < n; i++) {
    if (sys.type[i] < 0) continue;
    for (let pj = far.start[i]; pj < far.end[i]; pj++) {
      const j = far.nbr[pj];
      if (sys.type[j] < 0) continue;
      if (!pairFlag(sys, i, j, pj, ctrl.nonbCut)) continue;
      const r_ij = far.d[pj];
      const twbp = param.tbp[sys.type[i] * param.numAtomTypes + sys.type[j]];
      const { Tap, dTap } = taperAt(ws_taper(sys), r_ij);
      const r = directPair(param, ctrl, Tap, dTap, r_ij, twbp, twbp.gamma_w, p_vdW1);
      const e_vdW_tot = Tap * (r.e_vdW + r.e_core + r.e_lg);
      en.e_vdW += e_vdW_tot;
      const qq = sys.q[i] * sys.q[j];
      const e_ele = qq * r.e_ele0;
      en.e_ele += e_ele;
      const coef = r.CEvd + qq * r.CEclmb;
      ws.f[3 * i] += -coef * far.dvec[3 * pj];
      ws.f[3 * i + 1] += -coef * far.dvec[3 * pj + 1];
      ws.f[3 * i + 2] += -coef * far.dvec[3 * pj + 2];
      ws.f[3 * j] += coef * far.dvec[3 * pj];
      ws.f[3 * j + 1] += coef * far.dvec[3 * pj + 1];
      ws.f[3 * j + 2] += coef * far.dvec[3 * pj + 2];
    }
  }
  acks2Pairs(sys);
  polarizationEnergy(sys);
};

/** The taper coefficients of the system (set by the pair style per evaluation). */
const ws_taper = (sys: RxSystem): Float64Array => sys.Tap;

/**
 * Tabulated_vdW_Coulomb_Energy: cubic-spline interpolation of the tables of lookup.ts. The table
 * arrays are indexed as in the C++ code (coefficient r for the interval starting at r*dx).
 */
export const tabulatedVdwCoulombEnergy = (sys: RxSystem, LR: LookupTables): void => {
  const { ctrl, far, en, n } = sys;
  for (let i = 0; i < n; i++) {
    const ti = sys.type[i];
    if (ti < 0) continue;
    for (let pj = far.start[i]; pj < far.end[i]; pj++) {
      const j = far.nbr[pj];
      const tj = sys.type[j];
      if (tj < 0) continue;
      if (!pairFlag(sys, i, j, pj, ctrl.nonbCut)) continue;
      const r_ij = far.d[pj];
      const tmin = Math.min(ti, tj), tmax = Math.max(ti, tj);
      const t = LR.table[tmin * sys.param.numAtomTypes + tmax];
      if (!t) throw new StyleError(`ReaxFF lookup table for types ${tmin + 1} ${tmax + 1} was not built`);
      // cubic spline interpolation
      let r = Math.trunc(r_ij * t.inv_dx);
      if (r === 0) ++r;
      const base = (r + 1) * t.dx;
      const dif = r_ij - base;
      const cub = (c: { a: Float64Array; b: Float64Array; c: Float64Array; d: Float64Array }) =>
        ((c.d[r] * dif + c.c[r]) * dif + c.b[r]) * dif + c.a[r];
      const e_vdW = cub(t.vdW);
      let e_ele = cub(t.ele);
      e_ele *= sys.q[i] * sys.q[j];
      en.e_vdW += e_vdW;
      en.e_ele += e_ele;
      const CEvd = cub(t.CEvd);
      let CEclmb = cub(t.CEclmb);
      CEclmb *= sys.q[i] * sys.q[j];
      const coef = CEvd + CEclmb;
      for (let c = 0; c < 3; c++) {
        sys.ws.f[3 * i + c] += -coef * far.dvec[3 * pj + c];
        sys.ws.f[3 * j + c] += coef * far.dvec[3 * pj + c];
      }
    }
  }
  acks2Pairs(sys);
  polarizationEnergy(sys);
};

/** LR_vdW_Coulomb: the table values at distance r_ij of the type pair (i, j) (used by Init_Lookup_Tables). */
export const lrVdwCoulomb = (param: ReaxParam, ctrl: ControlParams, T: Float64Array, i: number, j: number, r_ij: number): { H: number; e_vdW: number; CEvd: number; e_ele: number; CEclmb: number } => {
  const twbp = param.tbp[i * param.numAtomTypes + j];
  const p_vdW1 = param.gpL[28];
  const { Tap, dTap } = taperAt(T, r_ij);
  const r = directPair(param, ctrl, Tap, dTap, r_ij, twbp, twbp.gamma_w, p_vdW1);
  return {
    H: EV_to_KCALpMOL * (Tap / Math.pow(r_ij * r_ij * r_ij + twbp.gamma, 0.33333333333333)),
    e_vdW: Tap * (r.e_vdW + r.e_core + r.e_lg),
    CEvd: r.CEvd,
    e_ele: C_ele * (Tap / Math.pow(r_ij * r_ij * r_ij + twbp.gamma, 0.33333333333333)),
    CEclmb: C_ele * (dTap - Tap * r_ij / (r_ij * r_ij * r_ij + twbp.gamma)) / Math.pow(r_ij * r_ij * r_ij + twbp.gamma, 0.33333333333333),
  };
};
