/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_airebo.cpp (PairAIREBO::REBO_neigh, FREBO, bondorder,
 * gSpline, PijSpline, piRCSpline, TijSpline) and src/MANYBODY/pair_airebo.h, at
 * commit b332f28. The loop structure and the order of the force accumulations follow
 * the C++ text; the neighbor lists are built here from the engine's owned and ghost
 * positions (binned search) instead of the LAMMPS full neighbor list.
 * Authors credited in the C++ file headers: Ase Henry (MIT); bugfixes and optimizations:
 * Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer (Temple U), Markus Hoehnerbach
 * (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M: Thomas C. O'Connor (JHU) 2014.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { PairCompute } from '../../engine/force/types';
import { sp, sp5th, sptricubic, spbicubic, type Ref } from './splines';
import type { AireboParams, SplineTables } from './tables';
import type { Knots } from './splineInit';

export const TOL = 1.0e-9;

/** Scratch references for the derivative outputs of Sp / Sp2 (the C++ code passes double *). */
export const newRef = (): Ref => ({ v: 0 });

/** Kronecker delta of two type indices (0 = C, 1 = H). */
export const kron = (a: number, b: number): number => (a === b ? 1.0 : 0.0);

/**
 * The REBO part of PairAIREBO: REBO neighbor lists of every owned and ghost atom, the
 * two-body and bond-order energy and forces. The class state (parameters, splines) is
 * filled by the AIREBO pair (airebo.ts) and passed here as the RebCore interface.
 */
export interface ReboCore {
  P: AireboParams;
  T: SplineTables;
  K: Knots;
  /** Per atom (owned and ghost): element index 0 = C, 1 = H. */
  mt: Int32Array;
  /** Atom id of every owned and ghost atom (the image's owner id): the half-pair rule of FREBO uses it. */
  tag: Int32Array;
  /** Owned + ghost positions and forces (3 per atom). */
  x: Float64Array;
  f: Float64Array;
  /** REBO neighbor list, CSR: neighbors of atom a are rlist[rstart[a] .. rstart[a+1]). */
  rstart: Int32Array;
  rlist: Int32Array;
  /** Coordination sums of weighting functions with REBO neighbors of C and H type. */
  nC: Float64Array;
  nH: Float64Array;
  /** AIREBO-BC (bcflag): the C-C P term is bond-averaged (src/MANYBODY/pair_airebo.cpp bcflag). */
  bcflag: boolean;
  /** Energy accumulator (pc.acc), per-atom energy and per-atom virial (6 per atom; null when not requested). */
  acc: { evdwl: number };
  eatom: Float64Array | null;
  vatom: Float64Array | null;
  /** Scratch derivative of the pair cutoff Sp(r_ij). */
  dwij: Ref;
  /** Scratch derivative of the cutoff S(t) of gSpline. */
  dS: Ref;
}


/** PijSpline: the P_ij correction of the bond order; dN2 receives its derivatives. */
export const PijSpline = (
  core: ReboCore, NijC0: number, NijH0: number, typei: number, typej: number, dN2: Float64Array,
): number => {
  let NijC = NijC0, NijH = NijH0;
  let Pij = 0.0;
  dN2[0] = 0.0;
  dN2[1] = 0.0;
  const { T, K } = core;
  if (core.bcflag && typei === 0 && typej === 0) {
    // AIREBO-BC (PairAIREBO::PijSpline, bcflag branch): the half-integer grid in doubled coordinates
    // (u = 2 N_C, v = 2 N_H, domain [0,3] each); dN2 = dP/du, dP/dv unscaled, zero at the knots.
    if (NijC < 0.0) NijC = 0.0;
    if (NijC > 3.0) NijC = 3.0;
    if (NijH < 0.0) NijH = 0.0;
    if (NijH > 3.0) NijH = 3.0;
    const u = 2.0 * NijC;
    const v = 2.0 * NijH;
    let x = Math.floor(u);
    let y = Math.floor(v);
    if (Math.abs(u - Math.floor(u)) < TOL && Math.abs(v - Math.floor(v)) < TOL) {
      Pij = T.bcKnots[x * 7 + y];
    } else {
      if (u === 6.0) --x;
      if (v === 6.0) --y;
      Pij = spbicubic(u, v, T.pCCbc, (x * 6 + y) * 16, dN2);
    }
    return Pij;
  }
  if (typei === 1) return Pij;
  if (typej === 0) {
    const d = T.pCCdom;
    // if inputs are out of bounds set them back to a point in bounds
    if (NijC < d[0]) NijC = d[0];
    if (NijC > d[1]) NijC = d[1];
    if (NijH < d[2]) NijH = d[2];
    if (NijH > d[3]) NijH = d[3];
    let x = Math.floor(NijC);
    let y = Math.floor(NijH);
    if (Math.abs(NijC - Math.floor(NijC)) < TOL && Math.abs(NijH - Math.floor(NijH)) < TOL) {
      Pij = K.PCCf[x * 5 + y];
      dN2[0] = K.PCCdfdx[x * 5 + y];
      dN2[1] = K.PCCdfdy[x * 5 + y];
    } else {
      if (NijC === d[1]) --x;
      if (NijH === d[3]) --y;
      Pij = spbicubic(NijC, NijH, T.pCC, (x * 4 + y) * 16, dN2);
    }
  } else if (typej === 1) {
    const d = T.pCHdom;
    if (NijC < d[0]) NijC = d[0];
    if (NijC > d[1]) NijC = d[1];
    if (NijH < d[2]) NijH = d[2];
    if (NijH > d[3]) NijH = d[3];
    let x = Math.floor(NijC);
    let y = Math.floor(NijH);
    if (Math.abs(NijC - Math.floor(NijC)) < TOL && Math.abs(NijH - Math.floor(NijH)) < TOL) {
      Pij = K.PCHf[x * 5 + y];
      dN2[0] = K.PCHdfdx[x * 5 + y];
      dN2[1] = K.PCHdfdy[x * 5 + y];
    } else {
      if (NijC === d[1]) --x;
      if (NijH === d[3]) --y;
      Pij = spbicubic(NijC, NijH, T.pCH, (x * 4 + y) * 16, dN2);
    }
  }
  return Pij;
};

/**
 * Tricubic lookup shared by piRCSpline and TijSpline: dom holds the domain of each of the three
 * arguments (rows [min, max]); knots are [5][5][dim3] (dim3 = 11 for piRC, 10 for Tij) and the
 * patches are [4][4][9][64].
 */
const tricubicLookup = (
  a0: number, b0: number, c0: number, dom: Float64Array, knotF: Float64Array, knotDx: Float64Array,
  knotDy: Float64Array, knotDz: Float64Array, patches: Float64Array, dN3: Float64Array, dim3: number,
): number => {
  let a = a0, b = b0, c = c0;
  // if the inputs are out of bounds set them back to a point in bounds
  if (a < dom[0]) a = dom[0];
  if (a > dom[1]) a = dom[1];
  if (b < dom[2]) b = dom[2];
  if (b > dom[3]) b = dom[3];
  if (c < dom[4]) c = dom[4];
  if (c > dom[5]) c = dom[5];
  let x = Math.floor(a);
  let y = Math.floor(b);
  let z = Math.floor(c);
  if (Math.abs(a - Math.floor(a)) < TOL && Math.abs(b - Math.floor(b)) < TOL
    && Math.abs(c - Math.floor(c)) < TOL) {
    const kk = x * 5 * dim3 + y * dim3 + z;
    dN3[0] = knotDx[kk];
    dN3[1] = knotDy[kk];
    dN3[2] = knotDz[kk];
    return knotF[kk];
  }
  if (a === dom[1]) --x;
  if (b === dom[3]) --y;
  if (c === dom[5]) --z;
  return sptricubic(a, b, c, patches, ((x * 4 + y) * 9 + z) * 64, dN3);
};

/** piRCSpline: the conjugation correction of the bond order (CC, CH and HH variants). */
export const piRCSpline = (
  core: ReboCore, Nij: number, Nji: number, Nijconj: number, typei: number, typej: number, dN3: Float64Array,
): number => {
  const { T, K } = core;
  dN3[0] = 0.0;
  dN3[1] = 0.0;
  dN3[2] = 0.0;
  if (typei === 0 && typej === 0) {
    return tricubicLookup(Nij, Nji, Nijconj, T.piCCdom, K.piCCf, K.piCCdfdx, K.piCCdfdy, K.piCCdfdz, T.piCC, dN3, 11);
  }
  if ((typei === 0 && typej === 1) || (typei === 1 && typej === 0)) {
    return tricubicLookup(Nij, Nji, Nijconj, T.piCHdom, K.piCHf, K.piCHdfdx, K.piCHdfdy, K.piCHdfdz, T.piCH, dN3, 11);
  }
  if (typei === 1 && typej === 1) {
    return tricubicLookup(Nij, Nji, Nijconj, T.piHHdom, K.piHHf, K.piHHdfdx, K.piHHdfdy, K.piHHdfdz, T.piHH, dN3, 11);
  }
  return 0.0;
};

/** TijSpline: the dihedral-term spline of the bond order (C-C only). */
export const TijSpline = (
  core: ReboCore, Nij: number, Nji: number, Nijconj: number, dN3: Float64Array,
): number => {
  const { T, K } = core;
  dN3[0] = 0.0;
  dN3[1] = 0.0;
  dN3[2] = 0.0;
  return tricubicLookup(Nij, Nji, Nijconj, T.Tijdom, K.Tf, K.Tdfdx, K.Tdfdy, K.Tdfdz, T.Tijc, dN3, 10);
};

/** gSpline: the angular function g(cos theta, N) of the bond order; dgdc and dgdN receive derivatives. */
export const gSpline = (
  core: ReboCore, costh0: number, Nij: number, typei: number, dgdc: Ref, dgdN: Ref,
): number => {
  const { P, T } = core;
  let costh = costh0;
  let g = 0.0;
  let dS = 0.0;
  let dg1 = 0.0, dg2 = 0.0;
  dgdc.v = 0.0;
  dgdN.v = 0.0;
  const dd: Ref = newRef();
  // central atom is Carbon
  if (typei === 0) {
    const gCdom = T.gCdom;
    if (costh < gCdom[0]) costh = gCdom[0];
    if (costh > gCdom[4]) costh = gCdom[4];
    if (Nij >= P.NCmax) {
      // the clamped costh always falls into one of the intervals; the scan stops at the matching
      // interval and defaults to the last one otherwise
      let i = 0;
      for (i = 0; i < 3; i++) if (costh < gCdom[i + 1]) break;
      const g2 = sp5th(costh, T.gC2, i * 6, dd);
      dg2 = dd.v;
      g = g2;
      dgdc.v = dg2;
      dgdN.v = 0.0;
    }
    if (Nij <= P.NCmin) {
      let i = 0;
      for (i = 0; i < 3; i++) if (costh < gCdom[i + 1]) break;
      const g1 = sp5th(costh, T.gC1, i * 6, dd);
      dg1 = dd.v;
      g = g1;
      dgdc.v = dg1;
      dgdN.v = 0.0;
    }
    if (Nij > P.NCmin && Nij < P.NCmax) {
      let i = 0;
      for (i = 0; i < 3; i++) if (costh < gCdom[i + 1]) break;
      const g1 = sp5th(costh, T.gC1, i * 6, dd);
      dg1 = dd.v;
      const g2 = sp5th(costh, T.gC2, i * 6, dd);
      dg2 = dd.v;
      const cut = sp(Nij, P.NCmin, P.NCmax, core.dS);
      dS = core.dS.v;
      g = g2 + cut * (g1 - g2);
      dgdc.v = dg2 + cut * (dg1 - dg2);
      dgdN.v = dS * (g1 - g2);
    }
  }
  // central atom is Hydrogen
  if (typei === 1) {
    const gHdom = T.gHdom;
    if (costh < gHdom[0]) costh = gHdom[0];
    if (costh > gHdom[3]) costh = gHdom[3];
    let i = 0;
    for (i = 0; i < 2; i++) if (costh < gHdom[i + 1]) break;
    g = sp5th(costh, T.gH, i * 6, dd);
    dg1 = dd.v;
    dgdN.v = 0.0;
    dgdc.v = dg1;
  }
  return g;
};

