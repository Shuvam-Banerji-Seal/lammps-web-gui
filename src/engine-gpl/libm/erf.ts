/*
 * Support code for the styles ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the C
 * library erf() and erfc() that the native binary calls (Ubuntu glibc 2.35 on x86_64), bit for bit. Transcribed from
 * glibc sysdeps/ieee754/dbl-64/s_erf.c (glibc-2.35), which carries this notice:
 *
 *   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *   Developed at SunPro, a Sun Microsystems, Inc. business.
 *   Permission to use, copy, modify, and distribute this software is freely granted, provided that this notice
 *   is preserved.
 *   Modified by Naohiko Shimizu/Tokai University, Japan 1997/08/25, for performance improvement on pipelined
 *   processors.
 *
 * glibc builds s_erf.c without fused multiply-adds (the installed erfc has no FMA instruction), so plain binary64
 * arithmetic in the source's evaluation order reproduces it; its two calls of __ieee754_exp go to the exact exp of
 * ./exp.ts. Checked against the installed libm.so.6 (tests/engineLibmErf.test.ts). This file is GPL-2.0 like the rest
 * of src/engine-gpl (LICENSE).
 */
import { exp } from './exp';

const F = new Float64Array(1);
const U = new Uint32Array(F.buffer); // little-endian: U[0] = low word, U[1] = high word

const tiny = 1e-300, half = 0.5, one = 1.0, two = 2.0;
/* c = (float)0.84506291151 */
const erx = 8.45062911510467529297e-01;
const efx = 1.28379167095512586316e-01;
const pp0 = 1.28379167095512558561e-01, pp1 = -3.25042107247001499370e-01, pp2 = -2.84817495755985104766e-02,
  pp3 = -5.77027029648944159157e-03, pp4 = -2.37630166566501626084e-05;
const qq1 = 3.97917223959155352819e-01, qq2 = 6.50222499887672944485e-02, qq3 = 5.08130628187576562776e-03,
  qq4 = 1.32494738004321644526e-04, qq5 = -3.96022827877536812320e-06;
const pa0 = -2.36211856075265944077e-03, pa1 = 4.14856118683748331666e-01, pa2 = -3.72207876035701323847e-01,
  pa3 = 3.18346619901161753674e-01, pa4 = -1.10894694282396677476e-01, pa5 = 3.54783043256182359371e-02,
  pa6 = -2.16637559486879084300e-03;
const qa1 = 1.06420880400844228286e-01, qa2 = 5.40397917702171048937e-01, qa3 = 7.18286544141962662868e-02,
  qa4 = 1.26171219808761642112e-01, qa5 = 1.36370839120290507362e-02, qa6 = 1.19844998467991074170e-02;
const ra0 = -9.86494403484714822705e-03, ra1 = -6.93858572707181764372e-01, ra2 = -1.05586262253232909814e+01,
  ra3 = -6.23753324503260060396e+01, ra4 = -1.62396669462573470355e+02, ra5 = -1.84605092906711035994e+02,
  ra6 = -8.12874355063065934246e+01, ra7 = -9.81432934416914548592e+00;
const sa1 = 1.96512716674392571292e+01, sa2 = 1.37657754143519042600e+02, sa3 = 4.34565877475229228821e+02,
  sa4 = 6.45387271733267880336e+02, sa5 = 4.29008140027567833386e+02, sa6 = 1.08635005541779435134e+02,
  sa7 = 6.57024977031928170135e+00, sa8 = -6.04244152148580987438e-02;
const rb0 = -9.86494292470009928597e-03, rb1 = -7.99283237680523006574e-01, rb2 = -1.77579549177547519889e+01,
  rb3 = -1.60636384855821916062e+02, rb4 = -6.37566443368389627722e+02, rb5 = -1.02509513161107724954e+03,
  rb6 = -4.83519191608651397019e+02;
const sb1 = 3.03380607434824582924e+01, sb2 = 3.25792512996573918826e+02, sb3 = 1.53672958608443695994e+03,
  sb4 = 3.19985821950859553908e+03, sb5 = 2.55305040643316442583e+03, sb6 = 4.74528541206955367215e+02,
  sb7 = -2.24409524465858183362e+01;

/** GET_HIGH_WORD as a signed 32-bit integer. */
const highWord = (x: number): number => { F[0] = x; return U[1] | 0; };

/** z = x with its low word cleared (SET_LOW_WORD (z, 0)). */
const clearLow = (x: number): number => { F[0] = x; U[0] = 0; return F[0]; };

/** The tail of erf and erfc for 1.25 <= |x| < 28: exp(-z*z - 0.5625) * exp((z - x)(z + x) + R/S), x = |x|. */
const tail = (ax: number, ix: number, split: number): number => {
  const s = one / (ax * ax);
  let R: number, S: number;
  if (ix < split) {
    const R1 = ra0 + s * ra1, s2 = s * s;
    const S1 = one + s * sa1, s4 = s2 * s2;
    const R2 = ra2 + s * ra3, s6 = s4 * s2;
    const S2 = sa2 + s * sa3, s8 = s4 * s4;
    const R3 = ra4 + s * ra5;
    const S3 = sa4 + s * sa5;
    const R4 = ra6 + s * ra7;
    const S4 = sa6 + s * sa7;
    R = R1 + s2 * R2 + s4 * R3 + s6 * R4;
    S = S1 + s2 * S2 + s4 * S3 + s6 * S4 + s8 * sa8;
  } else {
    const R1 = rb0 + s * rb1, s2 = s * s;
    const S1 = one + s * sb1, s4 = s2 * s2;
    const R2 = rb2 + s * rb3, s6 = s4 * s2;
    const S2 = sb2 + s * sb3;
    const R3 = rb4 + s * rb5;
    const S3 = sb4 + s * sb5;
    const S4 = sb6 + s * sb7;
    R = R1 + s2 * R2 + s4 * R3 + s6 * rb6;
    S = S1 + s2 * S2 + s4 * S3 + s6 * S4;
  }
  const z = clearLow(ax);
  return exp(-z * z - 0.5625) * exp((z - ax) * (z + ax) + R / S);
};

/** erf(x) exactly as the native binary's glibc computes it. */
export const erf = (x: number): number => {
  const hx = highWord(x);
  const ix = hx & 0x7fffffff;
  if (ix >= 0x7ff00000) {
    // erf(nan) = nan, erf(+-inf) = +-1
    const i = (hx >>> 31) << 1;
    return (1 - i) + one / x;
  }
  if (ix < 0x3feb0000) {
    // |x| < 0.84375
    if (ix < 0x3e300000) {
      // |x| < 2**-28
      if (ix < 0x00800000) return 0.0625 * (16.0 * x + (16.0 * efx) * x);
      return x + efx * x;
    }
    const z = x * x;
    const r1 = pp0 + z * pp1, z2 = z * z;
    const r2 = pp2 + z * pp3, z4 = z2 * z2;
    const s1 = one + z * qq1;
    const s2 = qq2 + z * qq3;
    const s3 = qq4 + z * qq5;
    const r = r1 + z2 * r2 + z4 * pp4;
    const s = s1 + z2 * s2 + z4 * s3;
    const y = r / s;
    return x + x * y;
  }
  if (ix < 0x3ff40000) {
    // 0.84375 <= |x| < 1.25
    const s = Math.abs(x) - one;
    const P1 = pa0 + s * pa1, s2 = s * s;
    const Q1 = one + s * qa1, s4 = s2 * s2;
    const P2 = pa2 + s * pa3, s6 = s4 * s2;
    const Q2 = qa2 + s * qa3;
    const P3 = pa4 + s * pa5;
    const Q3 = qa4 + s * qa5;
    const P = P1 + s2 * P2 + s4 * P3 + s6 * pa6;
    const Q = Q1 + s2 * Q2 + s4 * Q3 + s6 * qa6;
    return hx >= 0 ? erx + P / Q : -erx - P / Q;
  }
  if (ix >= 0x40180000) return hx >= 0 ? one - tiny : tiny - one; // inf > |x| >= 6
  const ax = Math.abs(x);
  const r = tail(ax, ix, 0x4006db6e); // |x| < 1/0.35 (s_erf.c erf: 0x4006DB6E)
  return hx >= 0 ? one - r / ax : r / ax - one;
};

/** erfc(x) exactly as the native binary's glibc computes it. */
export const erfc = (x: number): number => {
  const hx = highWord(x);
  const ix = hx & 0x7fffffff;
  // erfc(nan) = nan, erfc(+-inf) = 0, 2
  if (ix >= 0x7ff00000) return ((hx >>> 31) << 1) + one / x;
  if (ix < 0x3feb0000) {
    // |x| < 0.84375
    if (ix < 0x3c700000) return one - x; // |x| < 2**-56
    const z = x * x;
    const r1 = pp0 + z * pp1, z2 = z * z;
    const r2 = pp2 + z * pp3, z4 = z2 * z2;
    const s1 = one + z * qq1;
    const s2 = qq2 + z * qq3;
    const s3 = qq4 + z * qq5;
    const r = r1 + z2 * r2 + z4 * pp4;
    const s = s1 + z2 * s2 + z4 * s3;
    const y = r / s;
    if (hx < 0x3fd00000) return one - (x + x * y); // x < 1/4
    let rr = x * y;
    rr += x - half;
    return half - rr;
  }
  if (ix < 0x3ff40000) {
    // 0.84375 <= |x| < 1.25
    const s = Math.abs(x) - one;
    const P1 = pa0 + s * pa1, s2 = s * s;
    const Q1 = one + s * qa1, s4 = s2 * s2;
    const P2 = pa2 + s * pa3, s6 = s4 * s2;
    const Q2 = qa2 + s * qa3;
    const P3 = pa4 + s * pa5;
    const Q3 = qa4 + s * qa5;
    const P = P1 + s2 * P2 + s4 * P3 + s6 * pa6;
    const Q = Q1 + s2 * Q2 + s4 * Q3 + s6 * qa6;
    if (hx >= 0) {
      const z = one - erx;
      return z - P / Q;
    }
    const z = erx + P / Q;
    return one + z;
  }
  if (ix < 0x403c0000) {
    // |x| < 28
    if (hx < 0 && ix >= 0x40180000) return two - tiny; // x < -6
    const ax = Math.abs(x);
    const r = tail(ax, ix, 0x4006db6d); // |x| < 1/.35 (s_erf.c erfc: 0x4006DB6D)
    return hx > 0 ? r / ax : two - r / ax;
  }
  return hx > 0 ? tiny * tiny : two - tiny;
};
