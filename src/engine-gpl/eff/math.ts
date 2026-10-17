/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/pair_eff_inline.h, the Erf
 * polynomials (ipoly01, ipoly02, ipoly1, ierfoverx1), the electron kinetic energy (KinElec), the nucleus-electron,
 * electron-electron and Pauli terms (ElecNucNuc, ElecNucElec, ElecElecElec, PauliElecElec), the radial force
 * helpers (SmallRForce) and the seventh-order cutoff spline (cutoff, dcutoff). Contributing authors (from the
 * header): Andres Jaramillo-Botero, Hai Xiao, Julius Su (Caltech). LAMMPS is Copyright (2003) Sandia Corporation
 * and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The functions return the values that the C++ code adds (or subtracts) to its pointer arguments; the caller
 * applies them with the same sign, so every sum keeps the order of the original.
 */
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

/* pair_eff_inline.h: ERF_TERMS1 12, ERF_TERMS2 7, DERF_TERMS 13 */
const ERF_TERMS1 = 12;
const ERF_TERMS2 = 7;
const DERF_TERMS = 13;
/* pair_eff_inline.h: PAULI_RE 0.9, PAULI_RC 1.125, PAULI_RHO -0.2 */
const PAULI_RE = 0.9;
const PAULI_RC = 1.125;
const PAULI_RHO = -0.2;
/* sqrt(2), written as in pair_eff_inline.h (coeff_a) */
const COEFF_A = 1.4142135623730951;

/* pair_eff_inline.h error-function coefficient tables (generated from the header; E1, E2, DE1 — the header's DE2 table is never used there either). */
const E1 = Float64Array.from([
  1.483110564084803581889448079057, -3.01071073386594942470731046311E-1, 6.8994830689831566246603180718E-2,
  -1.3916271264722187682546525687E-2, 2.420799522433463662891678239E-3, -3.65863968584808644649382577E-4,
  4.8620984432319048282887568E-5, -5.749256558035684835054215E-6, 6.11324357843476469706758E-7,
  -5.8991015312958434390846E-8, 5.207009092068648240455E-9, -4.23297587996554326810E-10,
  3.1881135066491749748E-11, -2.236155018832684273E-12, 1.46732984799108492E-13,
  -9.044001985381747E-15, 5.25481371547092E-16, -2.8874261222849E-17,
  1.504785187558E-18, -7.4572892821E-20, 3.522563810E-21,
  -1.58944644E-22, 6.864365E-24, -2.84257E-25,
  1.1306E-26, -4.33E-28, 1.6E-29,
  -1.0E-30,
]);

const E2 = Float64Array.from([
  1.077977852072383151168335910348, -2.6559890409148673372146500904E-2, -1.487073146698099509605046333E-3,
  -1.38040145414143859607708920E-4, -1.1280303332287491498507366E-5, -1.172869842743725224053739E-6,
  -1.03476150393304615537382E-7, -1.1899114085892438254447E-8, -1.016222544989498640476E-9,
  -1.37895716146965692169E-10, -9.369613033737303335E-12, -1.918809583959525349E-12,
  -3.7573017201993707E-14, -3.7053726026983357E-14, 2.627565423490371E-15,
  -1.121322876437933E-15, 1.84136028922538E-16, -4.9130256574886E-17,
  1.0704455167373E-17, -2.671893662405E-18, 6.49326867976E-19,
  -1.65399353183E-19, 4.2605626604E-20, -1.1255840765E-20,
  3.025617448E-21, -8.29042146E-22, 2.31049558E-22,
  -6.5469511E-23, 1.8842314E-23, -5.504341E-24,
  1.630950E-24, -4.89860E-25, 1.49054E-25,
  -4.5922E-26, 1.4318E-26, -4.516E-27,
  1.440E-27, -4.64E-28, 1.51E-28,
  -5.0E-29, 1.7E-29, -6.0E-30,
  2.0E-30, -1.0E-30,
]);

const DE1 = Float64Array.from([
  -0.689379974848418501361491576718, 0.295939056851161774752959335568, -0.087237828075228616420029484096,
  0.019959734091835509766546612696, -0.003740200486895490324750329974, 0.000593337912367800463413186784,
  -0.000081560801047403878256504204, 9.886099179971884018535968E-6, -1.071209234904290565745194E-6,
  1.0490945447626050322784E-7, -9.370959271038746709966E-9, 7.6927263488753841874E-10,
  -5.8412335114551520146E-11, 4.125393291736424788E-12, -2.72304624901729048E-13,
  1.6869717361387012E-14, -9.84565340276638E-16, 5.4313471880068E-17,
  -2.840458699772E-18, 1.4120512798E-19, -6.688772574E-21,
  3.0257558E-22, -1.3097526E-23, 5.4352E-25,
  -2.1704E-26, 8.32E-28, -5.4E-29,
]);

/** pair_eff_inline.h ipoly02: P(x) in the range x > 2. */
const ipoly02 = (x0: number): number => {
  let b1 = 0.0, b0 = 0.0, b2 = 0.0;
  const x = x0 * 2;
  for (let i = ERF_TERMS2; i >= 0; i--) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + E2[i];
  }
  return 0.5 * (b0 - b2);
};

/** pair_eff_inline.h ipoly1: first derivative P'(x) in the range x < 2. */
const ipoly1 = (x0: number): number => {
  let b1 = 0.0, b0 = 0.0, b2 = 0.0;
  const x = x0 * 2;
  for (let i = DERF_TERMS; i >= 0; i--) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + DE1[i];
  }
  return 0.5 * (b0 - b2);
};

/** pair_eff_inline.h ipoly01: P(x) in the range x < 2. */
const ipoly01 = (x0: number): number => {
  let b1 = 0.0, b0 = 0.0, b2 = 0.0;
  const x = x0 * 2;
  for (let i = ERF_TERMS1; i >= 0; i--) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + E1[i];
  }
  return 0.5 * (b0 - b2);
};

/** pair_eff_inline.h ierfoverx1: Erf(x)/x and its first derivative. */
export const ierfoverx1 = (x: number): { f: number; df: number } => {
  if (x < 2.0) {
    /* erf(x) = x * y(t), t = 2 * (x/2)^2 - 1 */
    const t = 0.5 * x * x - 1;
    return { f: ipoly01(t), df: ipoly1(t) * x };
  }
  /* erf(x) = 1 - exp(-x^2)/x * y(t), t = (10.5 - x^2) / (2.5 + x^2) */
  const x2 = x * x;
  const t = (10.5 - x2) / (2.5 + x2);
  const expTerm = libmExp(-x2);
  const recipX = 1.0 / x;
  const f = 1.0 / x - (expTerm / x2) * ipoly02(t);
  const df = (1.12837916709551257389615890312 * expTerm - f) * recipX;
  return { f, df };
};

/** pair_eff_inline.h KinElec: the kinetic-energy term of an electron of radius r (eke, and its radial force). */
export const kinElec = (radius: number): { e: number; f: number } => ({
  e: 1.5 / (radius * radius),
  f: 3.0 / (radius * radius * radius),
});

/** pair_eff_inline.h ElecNucNuc: nucleus-nucleus Coulomb term (q = qi qj). */
export const elecNucNuc = (q: number, rc: number): { e: number; f: number } => ({
  e: q / rc,
  f: q / (rc * rc),
});

/** pair_eff_inline.h ElecNucElec: nucleus-electron term with the Gaussian electron of radius re1. */
export const elecNucElec = (q: number, rc: number, re1: number): { e: number; f: number; f1: number } => {
  const a = COEFF_A / re1;
  const arc = a * rc;
  const { f, df } = ierfoverx1(arc);
  const dEdr = q * a * a * df;
  const dEdr1 = -q * (a / re1) * (f + arc * df);
  const E = -q * a * f;
  return { e: E, f: dEdr, f1: dEdr1 };
};

/** pair_eff_inline.h ElecElecElec: electron-electron term of two Gaussian electrons (re1, re2). */
export const elecElecElec = (rc: number, re1: number, re2: number): { e: number; f: number; f1: number; f2: number } => {
  const re = Math.sqrt(re1 * re1 + re2 * re2);
  const a = COEFF_A / re;
  const arc = a * rc;
  const { f, df } = ierfoverx1(arc);
  const dEdr = -a * a * df;
  const fre = a * (f + arc * df) / (re * re);
  const dEdr1 = fre * re1;
  const dEdr2 = fre * re2;
  const E = a * f;
  return { e: E, f: dEdr, f1: dEdr1, f2: dEdr2 };
};

/**
 * pair_eff_inline.h PauliElecElec: Pauli repulsion between two electrons (samespin 1 or 0). Returns the Pauli energy
 * and the three force terms, which the C++ code subtracts from its force arguments (frc, fre1, fre2).
 */
export const pauliElecElec = (samespin: number, rc0: number, re10: number, re20: number): {
  ep: number; f: number; f1: number; f2: number;
} => {
  const re1 = re10 * PAULI_RE, re2 = re20 * PAULI_RE, rc = rc0 * PAULI_RC;
  const ree = re1 * re1 + re2 * re2;
  const rem = re1 * re1 - re2 * re2;
  const S = (2.82842712474619 / libmPow(re2 / re1 + re1 / re2, 1.5)) * libmExp(-rc * rc / ree);
  const t1 = 1.5 * (1 / (re1 * re1) + 1 / (re2 * re2));
  const t2 = 2.0 * (3 * ree - 2 * rc * rc) / (ree * ree);
  const tt = t1 - t2;
  const dSdr1 = (-1.5 / re1) * (rem / ree) + 2 * re1 * rc * rc / (ree * ree);
  const dSdr2 = (1.5 / re2) * (rem / ree) + 2 * re2 * rc * rc / (ree * ree);
  const dSdr = -2 * rc / ree;
  const dTdr1 = -3 / (re1 * re1 * re1) - 12 * re1 / (ree * ree) +
    8 * re1 * (-2 * rc * rc + 3 * ree) / (ree * ree * ree);
  const dTdr2 = -3 / (re2 * re2 * re2) - 12 * re2 / (ree * ree) +
    8 * re2 * (-2 * rc * rc + 3 * ree) / (ree * ree * ree);
  const dTdr = 8 * rc / (ree * ree);
  let O: number, dOdS: number;
  if (samespin === 1) {
    O = S * S / (1.0 - S * S) + (1 - PAULI_RHO) * S * S / (1.0 + S * S);
    dOdS = 2 * S / ((1.0 - S * S) * (1.0 - S * S)) +
      (1 - PAULI_RHO) * 2 * S / ((1.0 + S * S) * (1.0 + S * S));
  } else {
    O = -PAULI_RHO * S * S / (1.0 + S * S);
    dOdS = -PAULI_RHO * 2 * S / ((1.0 + S * S) * (1.0 + S * S));
  }
  const ratio = tt * dOdS * S;
  return {
    ep: tt * O,
    f: PAULI_RC * (dTdr * O + ratio * dSdr),
    f1: PAULI_RE * (dTdr1 * O + ratio * dSdr1),
    f2: PAULI_RE * (dTdr2 * O + ratio * dSdr2),
  };
};

/**
 * pair_eff_inline.h SmallRForce: the Cartesian force of a pair from its radial force `force` (force / rc along the
 * separation); for rc <= 1e-6 the components are taken from the direction cosines.
 */
export const smallRForce = (dx: number, dy: number, dz: number, rc: number, force: number): [number, number, number] => {
  if (rc > 0.000001) {
    const f = force / rc;
    return [f * dx, f * dy, f * dz];
  }
  const fx = dx !== 0 ? force / Math.sqrt(1 + (dy * dy + dz * dz) / (dx * dx)) : 0.0;
  const fy = dy !== 0 ? force / Math.sqrt(1 + (dx * dx + dz * dz) / (dy * dy)) : 0.0;
  const fz = dz !== 0 ? force / Math.sqrt(1 + (dx * dx + dy * dy) / (dz * dz)) : 0.0;
  return [fx, fy, fz];
};

/** pair_eff_inline.h cutoff: "Seventh order spline". */
export const cutoff = (x: number): number => (((20 * x - 70) * x + 84) * x - 35) * x * x * x * x + 1;

/** pair_eff_inline.h dcutoff: derivative of cutoff. */
export const dcutoff = (x: number): number => (((140 * x - 420) * x + 420) * x - 140) * x * x * x;

/**
 * pair_eff_inline.h ElecCoreNuc: the Coulomb interaction of a pseudo-core (espin 3) charge q with a nucleus, the core
 * being a Gaussian of radius re1 (no radial force on the core). Returns the energy and the pair force term.
 */
export const elecCoreNuc = (q: number, rc: number, re1: number): { e: number; f: number } => {
  const a = 1.4142135623730951 / re1;
  const arc = a * rc;
  const { f, df } = ierfoverx1(arc);
  return { e: q * a * f, f: -q * a * a * df };
};

/** pair_eff_inline.h ElecCoreCore: pseudo-core (espin 3) to pseudo-core interaction, the size re = sqrt(re1^2 + re2^2). */
export const elecCoreCore = (q: number, rc: number, re1: number, re2: number): { e: number; f: number } => {
  const re = Math.sqrt(re1 * re1 + re2 * re2);
  const a = 1.4142135623730951 / re;
  const arc = a * rc;
  const { f, df } = ierfoverx1(arc);
  return { e: q * a * f, f: -q * a * a * df };
};

/**
 * pair_eff_inline.h ElecCoreElec: pseudo-core charge q (its radius re1) with an electron of radius re2. The C++ code
 * subtracts its derivatives (`*frc -= dEdr; *fre2 -= dEdr2`), so f and f2 are returned already negated.
 */
export const elecCoreElec = (q: number, rc: number, re1: number, re2: number): { e: number; f: number; f2: number } => {
  const re = Math.sqrt(re1 * re1 + re2 * re2);
  const a = 1.4142135623730951 / re;
  const arc = a * rc;
  const { f, df } = ierfoverx1(arc);
  const E = -q * a * f;
  const dEdr = -q * a * df * a;
  const fre = q * a * (f + arc * df) / (re * re);
  const dEdr2 = fre * re2;
  return { e: E, f: -dEdr, f2: -dEdr2 };
};

/**
 * pair_eff_inline.h PauliCoreElec: the Pauli repulsion of a pseudo-core (s-type ECP, parameters A, B, C) with an
 * electron of radius re2. Returns the energy and the negated derivatives, as in the C++ `*frc -= ...` form.
 */
export const pauliCoreElec = (rc: number, re2: number, A: number, B: number, C: number): { ep: number; f: number; f2: number } => {
  const rcsq = rc * rc;
  const ssq = re2 * re2;
  const E = A * libmExp((-B * rcsq) / (ssq + C));
  const dEdrc = -2 * A * B * rc * libmExp(-B * rcsq / (ssq + C)) / (ssq + C);
  const dEdre2 = 2 * A * B * re2 * rcsq * libmExp(-B * rcsq / (ssq + C)) / ((C + ssq) * (C + ssq));
  return { ep: E, f: -dEdrc, f2: -dEdre2 };
};

/**
 * pair_eff_inline.h PauliCorePElec: the Pauli repulsion of a p-type pseudo-core (parameters A..E) with an electron
 * of radius re2. Same sign convention as pauliCoreElec.
 */
export const pauliCorePElec = (rc: number, re2: number, A: number, B: number, C: number, D: number, Ep: number): { ep: number; f: number; f2: number } => {
  const u = rc - C * re2;
  const g = libmPow(2.0 / (B / re2 + re2 / B), 5.0);
  const en = A * g * Math.pow(u, 2.0) * libmExp(-D * Math.pow(u, 2.0) / (Ep + re2 * re2));
  const dEdrc = A * g * 2.0 * u * libmExp(-D * Math.pow(u, 2.0) / (Ep + re2 * re2)) +
    en * (-D * 2.0 * u / (Ep + re2 * re2));
  const dEdre2 = en * (-5.0 / (B / re2 + re2 / B) * (-B / (re2 * re2) + 1.0 / B)) +
    A * g * 2.0 * u * (-C) * libmExp(-D * Math.pow(u, 2.0) / (Ep + re2 * re2)) +
    en * (2.0 * D * u * (C * Ep + rc * re2) / Math.pow((Ep + re2 * re2), 2.0));
  return { ep: en, f: -dEdrc, f2: -dEdre2 };
};
