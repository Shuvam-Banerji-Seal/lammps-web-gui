/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MESONT/pair_mesocnt.cpp (geometry, weight, finf, fsemi) and src/MESONT/pair_mesocnt.h (the switch
 * functions) at commit b332f28 (contributing author Philipp Kloza, University of Cambridge). Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The local geometry of a CNT segment pair (segment r1-r2 against the segment or chain p1-p2) and the mesoscopic
 * interaction forces: finf for two infinite CNTs (the phi and gamma tables), fsemi for an infinite and a
 * semi-infinite CNT (the usemi table, Gauss-Legendre quadrature). Lengths enter in the potential's Angstrom units
 * (the C++ scales by ang_inv), energies and forces leave in the units style.
 */
import type { MesocntTables } from './tables';
import { dspline1, dxspline2, dyspline2, ds5, dsCubic, glNodes, glWeights, s5, sCubic, spline1, spline2 } from './tables';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { acos as libmAcos } from '../libm/acos';

/** LAMMPS: static constexpr double SWITCH = 1.0e-4, SMALL = 1.0e-6, RHOMIN = 10.0 (pair_mesocnt.cpp). */
const SWITCH = 1.0e-4;
const SMALL = 1.0e-6;
const RHOMIN = 10.0;
/** LAMMPS: QUAD_FINF = 129 and QUAD_FSEMI = 10 (the quadrature orders of finf and fsemi). */
const QUAD_FINF = 129;
const QUAD_FSEMI = 10;

/** Constants of the potential in the units of the run (pair_coeff of pair_mesocnt.cpp). */
export interface MesoConst {
  tab: MesocntTables;
  ang: number;
  angInv: number;
  eunit: number;
  funit: number;
  /** Tube radius, sigma, and the derived lengths of coeff(). */
  r: number;
  rsq: number;
  d: number;
  dAng: number;
  rAng: number;
  sig: number;
  rc: number;
  cutoff: number;
  cutoffsq: number;
  cutoffAng: number;
  cutoffsqAng: number;
  comega: number;
  ctheta: number;
  delta1: number;
  delta2: number;
  glNodesFinf: Float64Array;
  glWeightsFinf: Float64Array;
  glNodesFsemi: Float64Array;
  glWeightsFsemi: Float64Array;
  warn?: (text: string) => void;
}

/** Builds the constants of a potential at the run's angstrom and energy units (PairMesoCNT::coeff). */
export const mesoConst = (tab: MesocntTables, ang: number, eunit: number, warn?: (text: string) => void): MesoConst => {
  const angInv = 1.0 / ang;
  const r = tab.rAng * ang;
  const d = 2.0 * r;
  const sig = tab.sigAng * ang;
  const rc = 3.0 * sig;
  const cutoff = rc + d;
  const cutoffsq = cutoff * cutoff;
  const cutoffAng = cutoff * angInv;
  const cutoffsqAng = cutoffAng * cutoffAng;
  const comega = 0.275 * (1.0 - 1.0 / (1.0 + 0.59 * tab.rAng));
  const ctheta = 0.35 + 0.0226 * (tab.rAng - 6.785);
  const glNodesFinf = glNodes(QUAD_FINF), glNodesFsemi = glNodes(QUAD_FSEMI);
  return {
    tab, ang, angInv, eunit, funit: eunit * angInv, r, rsq: r * r, d, dAng: 2.0 * tab.rAng, rAng: tab.rAng, sig, rc,
    cutoff, cutoffsq, cutoffAng, cutoffsqAng, comega, ctheta, delta1: tab.delta1, delta2: tab.delta2,
    glNodesFinf, glWeightsFinf: glWeights(QUAD_FINF, glNodesFinf),
    glNodesFsemi, glWeightsFsemi: glWeights(QUAD_FSEMI, glNodesFsemi), warn,
  };
};

/** Output of geometry(): the closest points p (on the chain) and m (its direction), param and the basis. */
export interface GeoOut {
  /** p (3), m (3). */
  p: Float64Array;
  m: Float64Array;
  /** h, alpha, xi1, xi2, eta1, eta2, etae (7). */
  param: Float64Array;
  /** ex, ey, ez as the three rows (9): basis[3a + k] is component k of basis vector a. */
  basis: Float64Array;
}

export const newGeoOut = (): GeoOut => ({
  p: new Float64Array(3), m: new Float64Array(3), param: new Float64Array(7), basis: new Float64Array(9),
});

const dot3 = (a: ArrayLike<number>, b: ArrayLike<number>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * PairMesoCNT::geometry: the local frame of the segment pair (r1, r2) and the chain (p1, p2). qe is the chain end
 * (semi-infinite case) or null. Writes out.p, out.m, out.param and out.basis; when the distance exceeds the cutoff only
 * param[0] is written (as in the C++), and the caller skips the pair.
 */
export const geometry = (
  c: MesoConst, r1: ArrayLike<number>, r2: ArrayLike<number>, p1: ArrayLike<number>, p2: ArrayLike<number>,
  qe: ArrayLike<number> | null, out: GeoOut,
): void => {
  const { p, m, param, basis } = out;
  const r = [0.5 * (r1[0] + r2[0]), 0.5 * (r1[1] + r2[1]), 0.5 * (r1[2] + r2[2])];
  const pc = [0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2])];
  p[0] = pc[0]; p[1] = pc[1]; p[2] = pc[2];
  const delr = [pc[0] - r[0], pc[1] - r[1], pc[2] - r[2]];
  const l = [r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]];
  normalize(l);
  m[0] = p2[0] - p1[0]; m[1] = p2[1] - p1[1]; m[2] = p2[2] - p1[2];
  normalize(m);
  let psi = dot3(l, m);
  if (psi > 1.0) psi = 1.0;
  else if (psi < -1.0) psi = -1.0;
  const denom = 1.0 - psi * psi;
  const psil = [l[0] * psi, l[1] * psi, l[2] * psi];
  const psim = [m[0] * psi, m[1] * psi, m[2] * psi];
  let rhoe = 0;
  if (qe) rhoe = dot3([p[0] - qe[0], p[1] - qe[1], p[2] - qe[2]], m);
  let taur: number, taup: number, etae: number;
  if (denom < SWITCH) {
    // parallel case
    taur = dot3(delr, l) - rhoe * psi;
    taup = -rhoe;
    etae = 0;
  } else {
    const frac = 1.0 / denom;
    const dellPsim = [l[0] - psim[0], l[1] - psim[1], l[2] - psim[2]];
    const delpsilM = [psil[0] - m[0], psil[1] - m[1], psil[2] - m[2]];
    taur = dot3(delr, dellPsim) * frac;
    taup = dot3(delr, delpsilM) * frac;
    etae = -rhoe - taup;
  }
  const rbar = [taur * l[0] + r[0], taur * l[1] + r[1], taur * l[2] + r[2]];
  const pbar = [taup * m[0] + p[0], taup * m[1] + p[1], taup * m[2] + p[2]];
  const delrbar = [pbar[0] - rbar[0], pbar[1] - rbar[1], pbar[2] - rbar[2]];
  let h = Math.sqrt(dot3(delrbar, delrbar));
  if (h > c.cutoff) {
    param[0] = h;
    return;
  }
  if (h * c.angInv < SMALL) h = SMALL * c.ang;
  const ex = [delrbar[0] * (1.0 / h), delrbar[1] * (1.0 / h), delrbar[2] * (1.0 / h)];
  const ez = l;
  // ey = ez x ex
  const ey = [ez[1] * ex[2] - ez[2] * ex[1], ez[2] * ex[0] - ez[0] * ex[2], ez[0] * ex[1] - ez[1] * ex[0]];
  const alpha = dot3(m, ey) < 0 ? libmAcos(psi) : 2 * Math.PI - libmAcos(psi);
  const delr1 = [r1[0] - rbar[0], r1[1] - rbar[1], r1[2] - rbar[2]];
  const delr2 = [r2[0] - rbar[0], r2[1] - rbar[1], r2[2] - rbar[2]];
  const delp1 = [p1[0] - pbar[0], p1[1] - pbar[1], p1[2] - pbar[2]];
  const delp2 = [p2[0] - pbar[0], p2[1] - pbar[1], p2[2] - pbar[2]];
  param[0] = h;
  param[1] = alpha;
  param[2] = dot3(delr1, l);
  param[3] = dot3(delr2, l);
  param[4] = dot3(delp1, m);
  param[5] = dot3(delp2, m);
  param[6] = etae;
  for (let k = 0; k < 3; k++) {
    basis[k] = ex[k];
    basis[3 + k] = ey[k];
    basis[6 + k] = ez[k];
  }
};

/** norm3: scales v to unit length when it is not zero (math_extra.h). */
const normalize = (v: Float64Array | number[]): void => {
  const val = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (val > 0.0) {
    const scale = 1.0 / Math.sqrt(val);
    v[0] *= scale;
    v[1] *= scale;
    v[2] *= scale;
  }
};

/** Output of weight(): the weight and its gradients with respect to the four segment ends. */
export interface WeightOut {
  w: number;
  dr1: Float64Array;
  dr2: Float64Array;
  dp1: Float64Array;
  dp2: Float64Array;
}

export const newWeightOut = (): WeightOut => ({
  w: 0, dr1: new Float64Array(3), dr2: new Float64Array(3), dp1: new Float64Array(3), dp2: new Float64Array(3),
});

/**
 * PairMesoCNT::weight: the smooth weight of a chain segment (p1, p2) for the substitute straight CNT, with its
 * gradients (dr1, dr2 with respect to the segment ends of the bond r1-r2; dp1, dp2 with respect to p1, p2).
 */
export const weight = (c: MesoConst, r1: ArrayLike<number>, r2: ArrayLike<number>, p1: ArrayLike<number>, p2: ArrayLike<number>, out: WeightOut): void => {
  const r = [0.5 * (r1[0] + r2[0]), 0.5 * (r1[1] + r2[1]), 0.5 * (r1[2] + r2[2])];
  const p = [0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2])];
  const distsq = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
  };
  const dr = Math.sqrt(0.25 * distsq(r1, r2) + c.rsq);
  const dp = Math.sqrt(0.25 * distsq(p1, p2) + c.rsq);
  const rhoc = dr + dp + c.rc;
  const rhomin = RHOMIN * c.ang;
  const rho = Math.sqrt(distsq(r, p));
  const frac = 1.0 / (rhoc - rhomin);
  const arg = frac * (rho - rhomin);
  out.w = sCubic(arg);
  if (out.w === 0.0 || out.w === 1.0) {
    out.dr1.fill(0); out.dr2.fill(0); out.dp1.fill(0); out.dp2.fill(0);
    return;
  }
  const factor = dsCubic(arg) * frac;
  const drRho = [(r[0] - p[0]) * (0.5 / rho), (r[1] - p[1]) * (0.5 / rho), (r[2] - p[2]) * (0.5 / rho)];
  const drRhoc = [(r1[0] - r2[0]) * (0.25 / dr), (r1[1] - r2[1]) * (0.25 / dr), (r1[2] - r2[2]) * (0.25 / dr)];
  const dpRhoc = [(p1[0] - p2[0]) * (0.25 / dp), (p1[1] - p2[1]) * (0.25 / dp), (p1[2] - p2[2]) * (0.25 / dp)];
  for (let k = 0; k < 3; k++) {
    out.dr1[k] = (-arg * drRhoc[k] + drRho[k]) * factor;
    out.dr2[k] = (arg * drRhoc[k] + drRho[k]) * factor;
  }
  for (let k = 0; k < 3; k++) drRho[k] = -drRho[k];
  for (let k = 0; k < 3; k++) {
    out.dp1[k] = (-arg * dpRhoc[k] + drRho[k]) * factor;
    out.dp2[k] = (arg * dpRhoc[k] + drRho[k]) * factor;
  }
};

/**
 * PairMesoCNT::finf: energy and forces of two infinite CNTs from param. f is 2 x 3 (row-major: f[3a + k]), in the
 * local basis; returns the energy.
 */
export const finf = (c: MesoConst, param: Float64Array, f: Float64Array): number => {
  const t = c.tab;
  const h = param[0] * c.angInv;
  const alpha = param[1];
  const xi1 = param[2] * c.angInv;
  const xi2 = param[3] * c.angInv;
  const sinAlpha = libmSin(alpha);
  const sinAlphasq = sinAlpha * sinAlpha;
  const dAng = c.dAng;
  // parallel case
  if (sinAlphasq < SWITCH) {
    const ubar = spline1(h, t.hstartUinf, t.delhUinf, t.uinfCoeff, t.uinfPoints);
    const delxi = xi2 - xi1;
    f[0] = 0.5 * delxi * dspline1(h, t.hstartUinf, t.delhUinf, t.uinfCoeff, t.uinfPoints) * c.funit;
    f[3] = f[0];
    f[1] = 0;
    f[4] = 0;
    f[2] = ubar * c.funit;
    f[5] = -f[2];
    return ubar * delxi * c.eunit;
  }
  // non-parallel case
  const sinAlphaInv = 1.0 / sinAlpha;
  const sinAlphasqInv = sinAlphaInv * sinAlphaInv;
  const cosAlpha = libmCos(alpha);
  const cotAlpha = cosAlpha * sinAlphaInv;
  const omega = 1.0 / (1.0 - c.comega * sinAlphasq);
  const c1 = omega * sinAlpha;
  const c1Inv = 1.0 / c1;
  const domega = 2.0 * c.comega * cosAlpha * c1 * omega;
  const gammaOrth = spline1(h, t.hstartGamma, t.delhGamma, t.gammaCoeff, t.gammaPoints);
  const dgammaOrth = dspline1(h, t.hstartGamma, t.delhGamma, t.gammaCoeff, t.gammaPoints);
  const gamma = 1.0 + (gammaOrth - 1.0) * sinAlphasq;
  const gammaInv = 1.0 / gamma;
  const dalphaGamma = 2.0 * (gammaOrth - 1.0) * sinAlpha * cosAlpha;
  const dhGamma = dgammaOrth * sinAlphasq;
  const zeta1 = xi1 * c1;
  const zeta2 = xi2 * c1;
  const smooth = s5((h - dAng - c.delta1) / (c.delta2 - c.delta1));
  const dsmooth = ds5((h - dAng - c.delta1) / (c.delta2 - c.delta1));
  const g = dAng + c.delta2;
  const hsq = h * h;
  let zetaminbar: number;
  if (h >= g) zetaminbar = 0;
  else zetaminbar = Math.sqrt(g * g - hsq);
  const zetamin = smooth * zetaminbar;
  const zetamax = Math.sqrt(c.cutoffsqAng - hsq);
  let dzetaminbar: number;
  if (h >= g) dzetaminbar = 0;
  else dzetaminbar = -h / zetaminbar;
  const dzetamin = dzetaminbar * smooth + zetaminbar * dsmooth / (c.delta2 - c.delta1);
  const dzetamax = -h / zetamax;
  const delzeta1 = Math.abs(zeta1) - zetamin;
  const delzeta2 = Math.abs(zeta2) - zetamin;
  const zetaRangeInv = 1.0 / (zetamax - zetamin);
  const psi1 = delzeta1 * zetaRangeInv;
  const psi2 = delzeta2 * zetaRangeInv;
  let deltaPhi: number, deltaDhPhi: number;
  let dzetaPhi1: number, dzetaPhi2: number;
  // out of the interpolation range: numerical integration of phi (the C++ branch with a warning)
  if (psi1 < 0 || psi2 < 0) {
    c.warn?.('Segment - infinite chain interaction outside of interpolation range. Attempting numerical integration, but performance may be poor and simulation likely unstable.');
    const scale = 0.5 * (zeta2 - zeta1);
    const shift = 0.5 * (zeta1 + zeta2);
    deltaPhi = 0.0;
    deltaDhPhi = 0.0;
    for (let i = 0; i < QUAD_FINF; i++) {
      const zeta = scale * c.glNodesFinf[i] + shift;
      const splineArg = Math.sqrt(hsq + zeta * zeta);
      deltaPhi += c.glWeightsFinf[i] * spline1(splineArg, t.hstartUinf, t.delhUinf, t.uinfCoeff, t.uinfPoints);
      deltaDhPhi += c.glWeightsFinf[i] * h * dspline1(splineArg, t.hstartUinf, t.delhUinf, t.uinfCoeff, t.uinfPoints) / splineArg;
    }
    deltaPhi *= scale;
    deltaDhPhi *= scale;
    dzetaPhi1 = spline1(Math.sqrt(hsq + zeta1 * zeta1), t.hstartUinf, t.delhUinf, t.uinfCoeff, t.uinfPoints);
    dzetaPhi2 = spline1(Math.sqrt(hsq + zeta2 * zeta2), t.hstartUinf, t.delhUinf, t.uinfCoeff, t.uinfPoints);
  } else {
    let phi1 = spline2(h, psi1, t.hstartPhi, t.psistartPhi, t.delhPhi, t.delpsiPhi, t.phiCoeff, t.phiPoints);
    let phi2 = spline2(h, psi2, t.hstartPhi, t.psistartPhi, t.delhPhi, t.delpsiPhi, t.phiCoeff, t.phiPoints);
    const dhPhibar1 = dxspline2(h, psi1, t.hstartPhi, t.psistartPhi, t.delhPhi, t.delpsiPhi, t.phiCoeff, t.phiPoints);
    const dhPhibar2 = dxspline2(h, psi2, t.hstartPhi, t.psistartPhi, t.delhPhi, t.delpsiPhi, t.phiCoeff, t.phiPoints);
    const dpsiPhibar1 = dyspline2(h, psi1, t.hstartPhi, t.psistartPhi, t.delhPhi, t.delpsiPhi, t.phiCoeff, t.phiPoints);
    const dpsiPhibar2 = dyspline2(h, psi2, t.hstartPhi, t.psistartPhi, t.delhPhi, t.delpsiPhi, t.phiCoeff, t.phiPoints);
    const dzetaRange = dzetamax - dzetamin;
    const dhPsi1 = -zetaRangeInv * (dzetamin + dzetaRange * psi1);
    const dhPsi2 = -zetaRangeInv * (dzetamin + dzetaRange * psi2);
    let dhPhi1 = dhPhibar1 + dpsiPhibar1 * dhPsi1;
    let dhPhi2 = dhPhibar2 + dpsiPhibar2 * dhPsi2;
    dzetaPhi1 = dpsiPhibar1 * zetaRangeInv;
    dzetaPhi2 = dpsiPhibar2 * zetaRangeInv;
    if (zeta1 < 0) {
      phi1 = -phi1;
      dhPhi1 = -dhPhi1;
    }
    if (zeta2 < 0) {
      phi2 = -phi2;
      dhPhi2 = -dhPhi2;
    }
    deltaPhi = phi2 - phi1;
    deltaDhPhi = dhPhi2 - dhPhi1;
  }
  const deltaDzetaPhi = dzetaPhi2 - dzetaPhi1;
  const c2 = gamma * c1Inv;
  const u = c2 * deltaPhi;
  const c3 = u * gammaInv;
  const dhU = dhGamma * c3 + c2 * deltaDhPhi;
  const dalphaU = dalphaGamma * c3 +
    c1Inv * (domega * sinAlpha + omega * cosAlpha) * (gamma * (xi2 * dzetaPhi2 - xi1 * dzetaPhi1) - u);
  const lrInv = 1.0 / (xi2 - xi1);
  const cx = h * gamma * sinAlphasqInv * deltaDzetaPhi;
  const cy = gamma * cotAlpha * deltaDzetaPhi;
  f[0] = lrInv * (xi2 * dhU - cx) * c.funit;
  f[3] = lrInv * (-xi1 * dhU + cx) * c.funit;
  f[1] = lrInv * (dalphaU - xi2 * cy) * c.funit;
  f[4] = lrInv * (-dalphaU + xi1 * cy) * c.funit;
  f[2] = gamma * dzetaPhi1 * c.funit;
  f[5] = -gamma * dzetaPhi2 * c.funit;
  return u * c.eunit;
};

/**
 * PairMesoCNT::fsemi: energy, forces (f, 2 x 3 row-major, local basis) and the end force fend of an infinite CNT and
 * a semi-infinite one from param (param[6] is the end offset etae). Returns the energy; fend is in out.fend.
 */
export const fsemi = (c: MesoConst, param: Float64Array, f: Float64Array, out: { fend: number }): number => {
  const t = c.tab;
  const h = param[0] * c.angInv;
  const alpha = param[1];
  const xi1 = param[2] * c.angInv;
  const xi2 = param[3] * c.angInv;
  const etae = param[6] * c.angInv;
  const sinAlpha = libmSin(alpha);
  const sinAlphasq = sinAlpha * sinAlpha;
  const cosAlpha = libmCos(alpha);
  const omega = 1.0 / (1.0 - c.comega * sinAlphasq);
  const omegasq = omega * omega;
  const domega = 2 * c.comega * sinAlpha * cosAlpha * omegasq;
  const theta = 1.0 - c.ctheta * sinAlphasq;
  const dtheta = -2 * c.ctheta * sinAlpha * cosAlpha;
  const c1 = omega * sinAlpha;
  const c1sq = c1 * c1;
  const c2 = theta * etae;
  const gammaOrth = spline1(h, t.hstartGamma, t.delhGamma, t.gammaCoeff, t.gammaPoints);
  const dgammaOrth = dspline1(h, t.hstartGamma, t.delhGamma, t.gammaCoeff, t.gammaPoints);
  const gamma = 1.0 + (gammaOrth - 1) * sinAlphasq;
  const gammaInv = 1.0 / gamma;
  const dalphaGamma = 2 * (gammaOrth - 1) * sinAlpha * cosAlpha;
  const dhGamma = dgammaOrth * sinAlphasq;
  const scale = 0.5 * (xi2 - xi1);
  const shift = 0.5 * (xi1 + xi2);
  const c3 = gamma * scale;
  let jh = 0, jh1 = 0, jh2 = 0, jxi = 0, jxi1 = 0, ubar = 0;
  for (let i = 0; i < QUAD_FSEMI; i++) {
    const xibar = scale * c.glNodesFsemi[i] + shift;
    const g = xibar * c1;
    const hbar = Math.sqrt(h * h + g * g);
    const thetabar = xibar * cosAlpha - c2;
    const cw = c.glWeightsFsemi[i];
    const u = cw * spline2(hbar, thetabar, t.hstartUsemi, t.xistartUsemi, t.delhUsemi, t.delxiUsemi, t.usemiCoeff, t.usemiPoints);
    let uh: number;
    if (hbar === 0) uh = 0;
    else uh = cw / hbar * dxspline2(hbar, thetabar, t.hstartUsemi, t.xistartUsemi, t.delhUsemi, t.delxiUsemi, t.usemiCoeff, t.usemiPoints);
    const uxi = cw * dyspline2(hbar, thetabar, t.hstartUsemi, t.xistartUsemi, t.delhUsemi, t.delxiUsemi, t.usemiCoeff, t.usemiPoints);
    const uh1 = xibar * uh;
    jh += uh;
    jh1 += uh1;
    jh2 += xibar * uh1;
    jxi += uxi;
    jxi1 += xibar * uxi;
    ubar += u;
  }
  jh *= c3;
  jh1 *= c3;
  jh2 *= c3;
  jxi *= c3;
  jxi1 *= c3;
  ubar *= c3;
  const c4 = gammaInv * ubar;
  const dhUbar = dhGamma * c4 + h * jh;
  const dalphaUbar = dalphaGamma * c4 + c1 * (domega * sinAlpha + omega * cosAlpha) * jh2 -
    sinAlpha * jxi1 - dtheta * etae * jxi;
  const cx = h * (omegasq * jh1 + cosAlpha * c.ctheta * jxi);
  const cy = sinAlpha * (cosAlpha * omegasq * jh1 + (c.ctheta - 1) * jxi);
  const cz1 = c1sq * jh1 + cosAlpha * jxi;
  const cz2 = c1sq * jh2 + cosAlpha * jxi1;
  const lInv = 1.0 / (xi2 - xi1);
  f[0] = lInv * (xi2 * dhUbar - cx) * c.funit;
  f[3] = lInv * (cx - xi1 * dhUbar) * c.funit;
  f[1] = lInv * (dalphaUbar - xi2 * cy) * c.funit;
  f[4] = lInv * (xi1 * cy - dalphaUbar) * c.funit;
  f[2] = lInv * (cz2 + ubar - xi2 * cz1) * c.funit;
  f[5] = lInv * (xi1 * cz1 - cz2 - ubar) * c.funit;
  out.fend = theta * jxi * c.funit;
  return ubar * c.eunit;
};
