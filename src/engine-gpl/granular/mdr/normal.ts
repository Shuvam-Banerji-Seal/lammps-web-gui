/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRANULAR/gran_sub_mod_normal.cpp (GranSubModNormalMDR::coeffs_to_local, calculate_forces,
 * calculate_nonadhesive_mdr_force, round_up_negative_epsilon and the MDR constants), src/GRANULAR/gran_sub_mod_damping.cpp
 * (GranSubModDampingMDR::calculate_forces) and src/GRANULAR/fix_granular_mdr.h (the history layout), at commit b332f28.
 * Contributing authors of the MDR model: William Zunker (MIT), Sachith Dunatunga (MIT), Dan Bolintineanu (SNL), Joel
 * Clemmer (SNL). Copyright (LAMMPS development team, GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * One contact side of the MDR normal model (Zunker and Kamrin 2024, doi 10.1016/j.jmps.2023.105492 and .105493; the
 * pair style docs: "The *mdr* model is a mechanically-derived contact model designed to capture the contact response
 * between adhesive elastic-plastic particles into large deformation."). A particle pair is split into two sides, each
 * a rigid-flat contact (see model.ts); a wall contact has one side.
 */
import { exp as libmExp } from '../../libm/exp';

/** Indices into a contact history (27 values per contact; fix_granular_mdr.h Granular_MDR_NS::HistoryIndex). */
export const DELTA_0 = 0;
export const DELTA_MDR_0 = 4;
export const DELTA_BULK_0 = 6;
export const DELTAMAX_MDR_0 = 8;
export const YFLAG_0 = 10;
export const DELTAY_0 = 12;
export const CA_0 = 14;
export const AADH_0 = 16;
export const AC_0 = 18;
export const EPS_BAR_0 = 20;
export const PENALTY = 22;
export const DELTA_MAX = 23;
export const DELTAP_0 = 24;
export const DAMP_SCALE = 26;
export const DELTAO_0 = 2;
export const HIST_SIZE = 27;

const MDR_MAX_IT = 100;
const MDR_EPSILON1 = 1e-10;
const MDR_EPSILON2 = 1e-16;
const MDR_EPSILON3 = 1e-20;
/** Maximum contact overlap for MDR (fraction of a particle radius). */
export const MDR_OVERLAP_LIMIT = 0.95;

const PI = Math.PI;
const PISQ = 9.8696044010893579923;
const PIINV = 0.318309886183790691216;
const PITOFIVETHIRDS = 6.73880859569814116838;
const CBRT2 = 1.25992104989487319067;
const SQRTHALFPI = 1.25331413731550012081;
const CBRTHALFPI = 1.16244735150962652526;
const THIRD = 1.0 / 3.0;
const FOURTHIRDS = 1.33333333333333333333;

/** Material of one MDR normal model (pair_coeff mdr E nu Y gamma psi_b eta_n0 and damping mdr d_type). */
export interface MdrMaterial {
  E: number;
  nu: number;
  Y: number;
  /** effective surface energy Delta gamma (adhesion when > 0). */
  gamma: number;
  psib: number;
  /** eta_n0 (the damping coefficient). */
  eta: number;
  /** damping class d_type: 1 scales eta_n0 with the damping scale, 2 uses eta_n0 while adhesion is active. */
  dtype: number;
}

/** Derived constants of a material (coeffs_to_local): G, kappa and the plane strain modulus E' = E/(1 - nu^2). */
export interface MdrConsts {
  G: number;
  kappa: number;
  Eeff: number;
  gamma: number;
  Y: number;
  psib: number;
  poiss: number;
}

export function mdrConsts(m: MdrMaterial): MdrConsts {
  if (!(m.E > 0)) throw new Error("pair_style granular: normal model 'mdr' needs Young's modulus E > 0");
  return {
    G: m.E / (2 * (1 + m.nu)),
    kappa: m.E / (3 * (1 - 2 * m.nu)),
    Eeff: m.E / (1 - m.nu * m.nu),
    gamma: m.gamma,
    Y: m.Y,
    psib: m.psib,
    poiss: m.nu,
  };
}

/** Per-atom MDR state arrays (indexed by owned atom; the names are the fix granular/mdr d_ properties). */
export interface MdrAtomArrays {
  Ro: Float64Array;
  Vcaps: Float64Array;
  Vgeo: Float64Array;
  Velas: Float64Array;
  eps_bar: Float64Array;
  dRnumerator: Float64Array;
  dRdenominator: Float64Array;
  Acon0: Float64Array;
  Acon1: Float64Array;
  Atot: Float64Array;
  Atot_sum: Float64Array;
  ddelta_bar: Float64Array;
  psi: Float64Array;
  sigmaxx: Float64Array;
  sigmayy: Float64Array;
  sigmazz: Float64Array;
  dRavg: Float64Array;
}

/** Newton-Raphson-free non-adhesive MDR force of a transformed displacement (calculate_nonadhesive_mdr_force). */
export function mdrNonadhesiveForce(delta: number, Ainv: number, Eeff: number, A: number, B: number): number {
  let F = Math.acos(1 - 2 * delta * Ainv);
  F -= (2 - 4 * delta * Ainv) * Math.sqrt(delta * Ainv - (delta * Ainv) * (delta * Ainv));
  F *= 0.25 * Eeff * A * B;
  return F;
}

/** Rounds values in (-EPSILON3, 0) to zero (round_up_negative_epsilon). */
const roundUpNegativeEpsilon = (v: number): number => (v < 0 && v > -MDR_EPSILON3 ? 0 : v);

/** Inputs of one contact side: the pair (or wall) history, the atom whose state is updated and its apparent radius. */
export interface MdrSideIn {
  hist: Float64Array;
  /** 0 or 1: which history entries (DELTA_0 + side ...) this side uses. Walls use 0. */
  side: 0 | 1;
  st: MdrAtomArrays;
  /** Owned atom index whose state this side updates (gm->i of the side). */
  ai: number;
  /** Apparent radius of that atom (gm->radi). */
  R: number;
  /** Overlap of this side (the partitioned overlap for particle pairs). */
  delta: number;
  /** Unit normal of the contact, the orientation native uses for gm->nx. */
  nx: number; ny: number; nz: number;
  c: MdrConsts;
  /** Whether the history advances (not at the run setup). */
  hu: boolean;
  /** Current penalty weight w = max(1 - p, 0) of the contact. */
  wij: number;
}

/** Per-side outputs: F0 (the side's force before the penalty), the damping contact area a_damp term and the side's Ac. */
export interface MdrSideOut {
  F0: number;
  aDamp: number;
  /** The side's wij-weighted contact area (native Ac_avg). */
  wAc: number;
}

/**
 * One side of the MDR contact (GranSubModNormalMDR::calculate_forces, the loop over contactSide). Returns F_MDR + F_BULK and
 * the contact radius term of the damping; updates the side's history and the atom state when hu is set. The
 * deltamax update is not guarded by hu, as in native.
 */
export function mdrSide(c: MdrSideIn): MdrSideOut {
  const { hist, side, st, ai, R, delta, nx, ny, nz, hu, wij, c: K } = c;
  const { kappa, G, Eeff, poiss, gamma, Y, psib } = K;
  const dDELTA = DELTA_0 + side, dDELTAO = DELTAO_0 + side;
  const dMDR = DELTA_MDR_0 + side, dBULK = DELTA_BULK_0 + side;
  const dMAX = DELTAMAX_MDR_0 + side, dYF = YFLAG_0 + side, dYD = DELTAY_0 + side;
  const dCA = CA_0 + side, dAADH = AADH_0 + side, dAC = AC_0 + side, dEPS = EPS_BAR_0 + side, dP = DELTAP_0 + side;
  const Ro = st.Ro[ai];
  const deltaAll = delta;
  // kinematics
  const ddelta = deltaAll - hist[dDELTA];
  if (hu) hist[dDELTA] = deltaAll;
  const deltao = deltaAll - (R - Ro);
  const ddeltao = deltao - hist[dDELTAO];
  if (hu) hist[dDELTAO] = deltao;
  let ddeltaMDR: number, ddeltaBULK: number;
  if (st.psi[ai] < psib) {
    // bulk response triggered: the increment is split between the MDR and BULK components
    ddeltaMDR = Math.min(ddelta - st.ddelta_bar[ai], deltaAll - hist[dMDR]);
    ddeltaBULK = st.ddelta_bar[ai];
  } else {
    ddeltaBULK = 0;
    ddeltaMDR = ddelta;
  }
  const deltaMDR = hist[dMDR] + ddeltaMDR;
  if (hu) hist[dMDR] = deltaMDR;
  const deltaBULK = Math.max(0, hist[dBULK] + ddeltaBULK);
  if (hu) hist[dBULK] = deltaBULK;
  if (deltaMDR > hist[dMAX]) hist[dMAX] = deltaMDR;
  const deltamaxMDR = hist[dMAX];
  // average pressure along the yield surface
  const pY = Y * (1.75 * libmExp((-4.4 * deltamaxMDR) / R) + 1.0);
  if (hist[dYF] === 0 && deltaMDR >= deltamaxMDR) {
    const phertz = (4 * Eeff * Math.sqrt(deltaMDR)) / (3 * PI * Math.sqrt(R));
    if (hu && phertz > pY) {
      hist[dYF] = 1;
      hist[dYD] = deltaMDR;
      hist[dCA] = PI * (hist[dYD] * hist[dYD] - hist[dYD] * R);
    }
  }
  // MDR force
  let FMDR: number;
  let A: number, Ainv: number, B: number, deltae1D: number;
  let amax: number;
  const cA = hist[dCA];
  if (hist[dYF] === 0) {
    // elastic contact
    A = 4 * R;
    Ainv = 1 / A;
    B = 2 * R;
    deltae1D = deltaMDR;
    amax = Math.sqrt(deltamaxMDR * R);
  } else {
    // plastic contact
    amax = Math.sqrt(2 * deltamaxMDR * R - deltamaxMDR * deltamaxMDR + cA * PIINV);
    const amaxsq = amax * amax;
    A = 4 * pY * (1 / Eeff) * amax;
    Ainv = 1 / A;
    B = 2 * amax;
    // maximum transformed elastic displacement
    const deltae1Dmax = A * 0.5;
    // force caused by full submersion of the elliptical indenter to depth A/2
    let Fmax = Eeff * (A * B * 0.25) * Math.acos(1 - 2 * deltae1Dmax * Ainv);
    Fmax -= (2 - 4 * deltae1Dmax * Ainv) * Math.sqrt(deltae1Dmax * Ainv - (deltae1Dmax * Ainv) * (deltae1Dmax * Ainv));
    // depth of the particle centre
    const zR = R - (deltamaxMDR - deltae1Dmax);
    let deltaR = 2 * amaxsq * (-1 + poiss) - (-1 + 2 * poiss) * zR * (-zR + Math.sqrt(amaxsq + zR * zR));
    deltaR *= Fmax / (2 * PI * amaxsq * G * Math.sqrt(amaxsq + zR * zR));
    // transformed elastic displacement
    deltae1D = (deltaMDR - deltamaxMDR + deltae1Dmax + deltaR) / (1 + deltaR / deltae1Dmax);
    // rigid flat placement
    if (hu) hist[dP] = deltamaxMDR - (deltae1Dmax + deltaR);
  }
  const a_fac = 0.99;
  const a_na = deltae1D >= 0 ? (B * Math.sqrt(A - deltae1D) * Math.sqrt(deltae1D)) * Ainv : 0;
  let aAdh = hist[dAADH];
  if (aAdh > a_fac * amax) aAdh = a_fac * amax;
  const Ainvsq = Ainv * Ainv;
  const Asq = A * A;
  const A4 = Asq * Asq;
  const Binv = 1 / B;
  const Bsq = B * B;
  const B4 = Bsq * Bsq;
  const Eeffinv = 1 / Eeff;
  const Eeffsq = Eeff * Eeff;
  const Eeffsqinv = Eeffinv * Eeffinv;
  if (gamma <= 0) {
    // non-adhesive contact
    FMDR = deltae1D <= 0 ? 0 : mdrNonadhesiveForce(deltae1D, Ainv, Eeff, A, B);
    if (hu) hist[dAADH] = a_na;
  } else if (deltaMDR === deltamaxMDR || a_na >= aAdh) {
    // case 1: no tensile springs, purely compressive contact
    FMDR = deltae1D <= 0 ? 0 : mdrNonadhesiveForce(deltae1D, Ainv, Eeff, A, B);
    if (hu) hist[dAADH] = a_fac * a_na;
  } else {
    // cases 2 and 3: tensile springs
    const gamma3 = gamma * gamma * gamma, gammasq = gamma * gamma, gamma4 = gammasq * gammasq;
    const lmax = Math.sqrt(2 * PI * aAdh * gamma * Eeffinv);
    let gAdh = A * 0.5 - A * Binv * Math.sqrt(Bsq * 0.25 - aAdh * aAdh);
    gAdh = roundUpNegativeEpsilon(gAdh);
    let tmp = 27 * A4 * B4 * gamma * Eeffinv;
    // powint(B, 6) by repeated squaring (MathSpecial::powint): (B^2)(B^4)
    const B2 = Bsq, B6 = B2 * (B2 * B2);
    tmp -= 2 * B6 * gamma3 * PISQ * (Eeffinv * Eeffinv * Eeffinv);
    tmp += Math.sqrt(27) * Asq * B4 * Math.sqrt(27 * A4 * Eeffsq * gammasq - 4 * Bsq * gamma4 * PISQ) * Eeffsqinv;
    tmp = Math.cbrt(tmp);
    let acrit = (-Bsq * gamma * PI * Ainvsq) * Eeffinv;
    acrit += (CBRT2 * B4 * gammasq * PITOFIVETHIRDS) / (Asq * Eeffsq * tmp);
    acrit += CBRTHALFPI * tmp * Ainvsq;
    acrit /= 6;
    if (deltae1D + lmax - gAdh >= 0) {
      // case 2: the tensile springs do not exceed the critical length
      const deltaeAdh = gAdh;
      const Fna = mdrNonadhesiveForce(deltaeAdh, Ainv, Eeff, A, B);
      const Fadh = 2 * Eeff * (deltae1D - deltaeAdh) * aAdh;
      FMDR = Fna + Fadh;
    } else {
      // case 3: the tensile springs exceed the critical length
      if (aAdh < acrit) {
        aAdh = 0;
        FMDR = 0;
      } else {
        // Newton-Raphson for aAdh
        let aTmp = aAdh;
        for (let lv1 = 0; lv1 < MDR_MAX_IT; ++lv1) {
          const faTmp = deltae1D - A * 0.5 + (A * Math.sqrt(Bsq * 0.25 - aTmp * aTmp)) * Binv;
          const fa = faTmp + Math.sqrt(2 * PI * aTmp * gamma * Eeffinv);
          if (Math.abs(fa) < MDR_EPSILON1) break;
          let dfda = (-aTmp * A) / (B * Math.sqrt(-(aTmp * aTmp) + Bsq * 0.25));
          dfda += (gamma * SQRTHALFPI) / Math.sqrt(aTmp * gamma * Eeff);
          aTmp = aTmp - fa / dfda;
          // native: fa2 uses the pre-update fa_tmp (the sum of the old aAdh)
          const fa2 = faTmp + Math.sqrt(2 * PI * aTmp * gamma * Eeffinv);
          if (Math.abs(fa - fa2) < MDR_EPSILON2) break;
          if (lv1 === MDR_MAX_IT - 1) aTmp = 0;
        }
        aAdh = aTmp;
        if (aAdh < acrit) {
          aAdh = 0;
          FMDR = 0;
        } else {
          gAdh = A * 0.5 - A * Binv * Math.sqrt(Bsq * 0.25 - aAdh * aAdh);
          gAdh = roundUpNegativeEpsilon(gAdh);
          const deltaeAdh = gAdh;
          const Fna = mdrNonadhesiveForce(deltaeAdh, Ainv, Eeff, A, B);
          const Fadh = 2 * Eeff * (deltae1D - deltaeAdh) * aAdh;
          FMDR = Fna + Fadh;
        }
      }
      if (hu) hist[dAADH] = aAdh;
    }
  }
  // area related calculations
  let Ac = hist[dYF] === 0 ? PI * deltaAll * R : PI * (2 * deltaAll * R - deltaAll * deltaAll) + cA;
  if (Ac < 0) Ac = 0;
  if (hu) {
    st.Atot_sum[ai] += wij * (Ac - 2 * PI * R * (deltamaxMDR + deltaBULK));
    st.Acon1[ai] += wij * Ac;
  }
  // contact radius for damping
  const aDamp = gamma > 0 ? aAdh : a_na;
  // bulk force
  const FBULK = deltaBULK <= 0 ? 0 : (1 / st.Vgeo[ai]) * st.Acon0[ai] * deltaBULK * kappa * Ac;
  const F0 = FMDR + FBULK;
  if (hu) {
    // mean surface displacement and the radius update quantities
    hist[dAC] = wij * Ac;
    st.Vcaps[ai] += PI * THIRD * deltaAll * deltaAll * (3 * R - deltaAll);
  }
  const Fntmp = wij * (FMDR + FBULK);
  const fx = Fntmp * nx, fy = Fntmp * ny, fz = Fntmp * nz;
  const bx = -(Ro - deltao) * nx, by = -(Ro - deltao) * ny, bz = -(Ro - deltao) * nz;
  const epsBarContact = (fx * bx + fy * by + fz * bz) / (3 * kappa * st.Velas[ai]);
  if (hu) st.eps_bar[ai] += epsBarContact;
  if (hu && deltaMDR === deltamaxMDR && hist[dYF] > 0 && FMDR > 0) {
    const Vo = FOURTHIRDS * PI * Ro * Ro * Ro;
    st.dRnumerator[ai] -= Vo * (epsBarContact - hist[dEPS]);
    st.dRnumerator[ai] -= wij * PI * ddeltao * (2 * deltao * Ro - deltao * deltao + R * R - Ro * Ro);
    st.dRdenominator[ai] += wij * 2 * PI * R * (deltao + R - Ro);
  }
  if (hu) {
    hist[dEPS] = epsBarContact;
    st.sigmaxx[ai] += (fx * bx) / st.Velas[ai];
    st.sigmayy[ai] += (fy * by) / st.Velas[ai];
    st.sigmazz[ai] += (fz * bz) / st.Velas[ai];
  }
  return { F0, aDamp, wAc: wij * Ac };
}

/** Inputs for the damping scale of a whole contact (pair or wall). */
export interface MdrDampIn {
  hist: Float64Array;
  hu: boolean;
  aDamp: number;
  /** Damping scale computed from the mass and modulus of the contact: sqrt(m_eff 2 E' a) (pair) or sqrt(m_eff E' a) (wall). */
  dampScale: number;
  /** eta_n0. */
  eta: number;
  dtype: number;
}

/**
 * Stores the damping scale when the history advances and returns the damping prefactor of this step, as
 * GranSubModDampingMDR::calculate_forces reads it: damp_type 1 scales eta_n0 by the stored DAMP_SCALE; damp_type 2 uses eta_n0
 * wherever the stored DAMP_SCALE is non-zero.
 */
export function mdrDampPrefactor(c: MdrDampIn): number {
  if (c.hu) c.hist[DAMP_SCALE] = c.aDamp <= 0 ? 0 : c.dampScale;
  if (c.dtype === 1) return c.eta * c.hist[DAMP_SCALE];
  return c.hist[DAMP_SCALE] === 0 ? 0 : c.eta;
}
