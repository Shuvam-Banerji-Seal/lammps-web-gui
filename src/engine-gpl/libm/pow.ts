/*
 * Support code for the styles ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the C
 * library pow() that the native binary calls (Ubuntu glibc 2.35 on an x86_64 CPU with FMA and AVX2, where glibc
 * dispatches pow to __ieee754_pow_fma: sysdeps/ieee754/dbl-64/e_pow.c and e_pow_log_data.c compiled with -mfma -mavx2,
 * with the exp tables of e_exp_data.c; written by Szabolcs Nagy, Copyright (C) 2018-2022 Free Software Foundation,
 * Inc.). This file reproduces that function bit for bit: the algorithm and tables of e_pow.c, with a fused multiply-add
 * exactly where the installed libm.so.6 has one (read from its disassembly: in log_inline z invc - 1, k Ln2hi + logc,
 * k Ln2lo + logctail, the error of A0 r r, the polynomial steps and ar3 p + (lo1 + lo2 + lo3 + lo4); then y hi - ehi and
 * y lo + that; in exp_inline InvLn2N x + Shift, both reduction steps, the three polynomial steps and scale + scale tmp;
 * the k < 0 special case keeps the unfused products of the source). The log table was read from libm.so.6.
 * glibc is LGPL-2.1-or-later; used under its section 3 as GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * Fast path (powFast): log_inline runs with the exact product errors (z invc - 1, A0 r r - ar2, y hi - ehi) computed by
 * Dekker products and its polynomial unfused; its result hi = RN(hi + lo) is libm's unless hi + lo lies within the
 * bound on the unfused lo of a rounding boundary, and r = RN(elo + t) in exp_inline is libm's unless elo + t lies within
 * the propagated bound of one; kd is libm's unless x InvLn2N lies within 2^-30 of a half-integer. The steps after that
 * (the reduction step, the polynomial and scale + scale tmp) are fused exactly as libm, with the fmaF of ./log.ts
 * inlined. Anything else goes through the fully fused computation (powExact). Checked against the native libm on
 * 3,000,000 pairs (tests/engineLibmPow.test.ts keeps 4033 of them): no difference.
 */
import { fma } from './log';
import {
  InvLn2N, Shift, NegLn2hiN, NegLn2loN, C2, C3, C4, C5, SLO, SC2, SC4, TAB, TAILD, SCALE2, POW2, POW2_OFF,
} from './exp';

const F = new Float64Array(1);
const U = new Uint32Array(F.buffer); // little-endian: U[0] low word, U[1] high word
const SPLIT = 134217729;
const asDouble = (hi: number, lo: number): number => { U[1] = hi; U[0] = lo; return F[0]; };

// __pow_log_data (read from libm.so.6)
const Ln2hi = 0.6931471805598903, Ln2lo = 5.497923018708371e-14;
const A0 = -0.5, A1 = -0.6666666666666679, A2 = 0.5000000000000007, A3 = 0.7999999995323976;
const A4 = -0.6666666663487739, A5 = -1.142909628459501, A6 = 1.0000415263675542;
/** __pow_log_data.tab: invc, logc, logctail for each of the N = 128 subintervals. */
const TABL = new Float64Array([
1.4140625, -0.3464667673462145, 5.929407345889625e-15,
1.40625, -0.34092658697056777, -2.544157440035963e-14,
1.3984375, -0.3353555419211034, -3.443525940775045e-14,
1.390625, -0.3297532863724655, -2.500123826022799e-15,
1.3828125, -0.32411946865420305, -8.929337133850617e-15,
1.375, -0.31845373111855224, 1.7625431312172662e-14,
1.3671875, -0.31275571000389846, 1.5688303180062087e-15,
1.359375, -0.3070250352949415, 2.9655274673691784e-14,
1.3515625, -0.3012613305781997, 3.7923164802093147e-14,
1.34375, -0.2954642128938758, 3.993416384387844e-14,
1.3359375, -0.28963329258306203, 1.9352855826489123e-14,
1.3359375, -0.28963329258306203, 1.9352855826489123e-14,
1.328125, -0.28376817313062475, -1.9852665484979036e-14,
1.3203125, -0.27786845100342816, -2.814323765595281e-14,
1.3125, -0.2719337154836694, 2.7643769993528702e-14,
1.3046875, -0.2659635484970977, -4.025092402293806e-14,
1.296875, -0.25995752443691345, -1.2621729398885316e-14,
1.2890625, -0.25391520998095984, -3.600176732637335e-15,
1.2890625, -0.25391520998095984, -3.600176732637335e-15,
1.28125, -0.2478361639045943, 1.3029797173308663e-14,
1.2734375, -0.2417199368871934, 4.8230289429940886e-14,
1.265625, -0.23556607131274632, -2.0592242769647135e-14,
1.2578125, -0.22937410106487732, 3.149265065191484e-14,
1.25, -0.22314355131425145, 4.169796584527195e-14,
1.25, -0.22314355131425145, 4.169796584527195e-14,
1.2421875, -0.21687393830063684, 2.2477465222466186e-14,
1.234375, -0.21056476910735, 3.6507188831790577e-16,
1.2265625, -0.2042155414286526, -3.827767260205414e-14,
1.2265625, -0.2042155414286526, -3.827767260205414e-14,
1.21875, -0.19782574332987224, -4.7641388950792196e-14,
1.2109375, -0.19139485299967873, 4.9278276214647115e-14,
1.203125, -0.18492233849406148, 4.9485167661250996e-14,
1.203125, -0.18492233849406148, 4.9485167661250996e-14,
1.1953125, -0.1784076574728033, -1.5003333854266542e-14,
1.1875, -0.17185025692663203, -2.7194441649495324e-14,
1.1875, -0.17185025692663203, -2.7194441649495324e-14,
1.1796875, -0.1652495728952772, -2.99659267292569e-14,
1.171875, -0.15860503017665906, 2.0472357800461955e-14,
1.171875, -0.15860503017665906, 2.0472357800461955e-14,
1.1640625, -0.15191604202584585, 3.879296723063646e-15,
1.15625, -0.1451820098444614, -3.6506824353335045e-14,
1.1484375, -0.13840232285906495, -5.4183331379008994e-14,
1.1484375, -0.13840232285906495, -5.4183331379008994e-14,
1.140625, -0.131576357788731, 1.1729485484531301e-14,
1.140625, -0.131576357788731, 1.1729485484531301e-14,
1.1328125, -0.12470347850091912, -3.811763084710266e-14,
1.125, -0.11778303565643, 4.654729747598445e-14,
1.125, -0.11778303565643, 4.654729747598445e-14,
1.1171875, -0.11081436634026431, -2.5799991283069902e-14,
1.109375, -0.10379679368168127, 3.7700471749674615e-14,
1.109375, -0.10379679368168127, 3.7700471749674615e-14,
1.1015625, -0.09672962645856842, 1.7306161136093256e-14,
1.1015625, -0.09672962645856842, 1.7306161136093256e-14,
1.09375, -0.089612158689647, -4.012913552726574e-14,
1.0859375, -0.08244366921110213, 2.7541708360737882e-14,
1.0859375, -0.08244366921110213, 2.7541708360737882e-14,
1.078125, -0.07522342123763792, 5.0396178134370583e-14,
1.078125, -0.07522342123763792, 5.0396178134370583e-14,
1.0703125, -0.06795066190852594, 1.8195060030168815e-14,
1.0625, -0.06062462181648698, 5.213620639136504e-14,
1.0625, -0.06062462181648698, 5.213620639136504e-14,
1.0546875, -0.053244514518837605, 2.532168943117445e-14,
1.0546875, -0.053244514518837605, 2.532168943117445e-14,
1.046875, -0.045809536031242715, -5.148849572685811e-14,
1.046875, -0.045809536031242715, -5.148849572685811e-14,
1.0390625, -0.038318864302141264, 4.6652946995830086e-15,
1.0390625, -0.038318864302141264, 4.6652946995830086e-15,
1.03125, -0.03077165866670839, -4.529814257790929e-14,
1.03125, -0.03077165866670839, -4.529814257790929e-14,
1.0234375, -0.023167059281490765, -4.361324067851568e-14,
1.015625, -0.015504186535963527, -1.7274567499706107e-15,
1.015625, -0.015504186535963527, -1.7274567499706107e-15,
1.0078125, -0.0077821404420319595, -2.298941004620351e-14,
1.0078125, -0.0077821404420319595, -2.298941004620351e-14,
1.0, 0.0, 0.0,
1.0, 0.0, 0.0,
0.9921875, 0.007843177461040796, -1.4902732911301337e-14,
0.984375, 0.01574835696817445, -3.527980389655325e-14,
0.9765625, 0.023716526617363343, -4.730054772033249e-14,
0.96875, 0.03174869831457272, 7.580310369375161e-15,
0.9609375, 0.039845908547249564, -4.9893776716773285e-14,
0.953125, 0.048009219186383234, -2.262629393030674e-14,
0.9453125, 0.056239718322899535, -2.345674491018699e-14,
0.94140625, 0.06038051098892083, -1.3352588834854848e-14,
0.93359375, 0.06871389254808946, -3.765296820388875e-14,
0.92578125, 0.07711730334438016, 5.1128335719851986e-14,
0.91796875, 0.08559193033545398, -5.046674438470119e-14,
0.9140625, 0.08985632912185793, 3.1218748807418837e-15,
0.90625, 0.09844007281321865, 3.3871241029241416e-14,
0.8984375, 0.10709813555638448, -1.7376727386423858e-14,
0.89453125, 0.11145544092528326, 3.957125899799804e-14,
0.88671875, 0.12022742699821265, -5.2849453521890294e-14,
0.8828125, 0.12464244520731427, -3.767012502308738e-14,
0.875, 0.13353139262449076, 3.1859736349078334e-14,
0.87109375, 0.13800567301939282, 5.0900642926060466e-14,
0.86328125, 0.14701474296180095, 8.710783796122478e-15,
0.859375, 0.15154989812720032, 6.157896229122976e-16,
0.8515625, 0.16068238169043525, 3.821577743916796e-14,
0.84765625, 0.16528009093906348, 3.9440046718453496e-14,
0.83984375, 0.17453941635187675, 2.2924522154618074e-14,
0.8359375, 0.17920142945774842, -3.742530094732263e-14,
0.83203125, 0.18388527877016259, -2.5223102140407338e-14,
0.82421875, 0.1933193110035063, -1.0320443688698849e-14,
0.8203125, 0.19806991376208316, 1.0634128304268335e-14,
0.8125, 0.20763936477828793, -4.3425422595242564e-14,
0.80859375, 0.21245865121420593, -1.2527395755711364e-14,
0.8046875, 0.21730127569003344, -5.204008743405884e-14,
0.80078125, 0.22216746534115828, -3.979844515951702e-15,
0.79296875, 0.2319714654378231, -4.7955860343296286e-14,
0.7890625, 0.2369097470783572, 5.015686013791602e-16,
0.78515625, 0.24187253642048745, -7.252318953240293e-16,
0.78125, 0.2468600779315011, 2.4688324156011588e-14,
0.7734375, 0.2569104137850218, 5.465121253624792e-15,
0.76953125, 0.26197371574153294, 4.102651071698446e-14,
0.765625, 0.2670627852490952, -4.996736502345936e-14,
0.76171875, 0.27217788591576664, 4.903580708156347e-14,
0.7578125, 0.27731928541618345, 5.089628039500759e-14,
0.75390625, 0.28248725557466514, 1.1782016386565151e-14,
0.74609375, 0.29290401643288533, 4.727452940514406e-14,
0.7421875, 0.29815337231912054, -4.4204083338755686e-14,
0.73828125, 0.3034304294199046, 1.548345993498083e-14,
0.734375, 0.30873548164959175, 2.1522127491642888e-14,
0.73046875, 0.3140688276249648, 1.1054030169005386e-14,
0.7265625, 0.31943077076641657, -5.534326352070679e-14,
0.72265625, 0.3248216194012912, -5.351646604259541e-14,
0.71875, 0.33024168687052224, 5.4612144489920215e-14,
0.71484375, 0.3356912916381134, 2.8136969901227338e-14,
0.7109375, 0.3411707574027787, -1.156568624616423e-14,
]);
/** SIGN_BIAS = 0x800 << EXP_TABLE_BITS: added to k for a negative result (the sign bit of scale). */
const SIGN_BIAS = 0x40000;
/** The low parts of the split constants (the high parts come from exp.ts). */
const SLO_L = NegLn2loN - SLO, SC2_L = C3 - SC2, SC4_L = C5 - SC4;

/** e_pow.c checkint for a finite non-zero y: 0 not an integer, 1 odd, 2 even. */
const checkint = (y: number): number => {
  if (!Number.isInteger(y)) return 0;
  if (Math.abs(y) >= 9007199254740992) return 2;
  return y % 2 === 0 ? 2 : 1;
};

/** log_inline results: hi, its tail, and (fast path) a bound on how far lo may be from libm's lo (-1: not stable). A
 * typed array, so the doubles are not boxed on every call as module variables would be. */
const LOG = new Float64Array(2);

/** log_inline of e_pow.c, every step as libm.so.6 does it (ix = (ixHi, ixLo), normalized for subnormals). */
const logExact = (ixHi: number, ixLo: number): void => {
  const tmpHi = (ixHi - 0x3fe69555) | 0;
  const i = (tmpHi >>> 13) & 127;
  const k = tmpHi >> 20;
  const z = asDouble((ixHi - (tmpHi & 0xfff00000)) >>> 0, ixLo);
  const kd = k;
  const invc = TABL[3 * i], logc = TABL[3 * i + 1], logctail = TABL[3 * i + 2];
  const r = fma(z, invc, -1.0);
  const t1 = fma(kd, Ln2hi, logc);
  const t2 = t1 + r;
  const lo1 = fma(kd, Ln2lo, logctail);
  const lo2 = (t1 - t2) + r;
  const ar = A0 * r;
  const ar2 = r * ar;
  const ar3 = r * ar2;
  const hi = t2 + ar2;
  const lo3 = fma(ar, r, -ar2);
  const lo4 = (t2 - hi) + ar2;
  const q10 = fma(r, A2, A1);
  const q11 = fma(r, A4, A3);
  const q0 = fma(fma(r, A6, A5), ar2, q11);
  const P = fma(ar2, q0, q10);
  const lo = fma(ar3, P, ((lo1 + lo2) + lo3) + lo4);
  const y = hi + lo;
  LOG[0] = y;
  LOG[1] = (hi - y) + lo;
};

/** __math_oflow / __math_uflow of glibc: +-inf / +-0 (signBias: a negative result). */
const oflow = (signBias: number): number => (signBias ? -Infinity : Infinity);
const uflow = (signBias: number): number => (signBias ? -0 : 0);

/** exp_inline of e_pow.c, every step as libm.so.6 does it: sign * exp(x + xtail). */
const expExact = (x: number, xtail: number, signBias: number): number => {
  F[0] = x;
  let abstop = (U[1] >>> 20) & 0x7ff;
  if (((abstop - 0x3c9) >>> 0) >= 0x3f) {
    if (abstop < 0x3c9) {
      // tiny x (0 is common): 1 + x with the rounding of the sum
      const one = 1.0 + x;
      return signBias ? -one : one;
    }
    if (abstop >= 0x409) return (U[1] >>> 31) ? uflow(signBias) : oflow(signBias);
    abstop = 0; // large x: the special cases below
  }
  const ks = fma(InvLn2N, x, Shift);
  F[0] = ks;
  const kiLo = U[0];
  const kd = ks - Shift;
  let r = fma(kd, NegLn2loN, fma(kd, NegLn2hiN, x));
  r = xtail + r;
  const idx = 2 * (kiLo & 127);
  const tail = asDouble(TAB[2 * idx], TAB[2 * idx + 1]);
  const sHi = (TAB[2 * idx + 2] + ((kiLo + signBias) << 13)) >>> 0, sLo = TAB[2 * idx + 3];
  const r2 = r * r;
  const p = fma(fma(r, C3, C2), r2, tail + r);
  const tmp = fma(fma(r, C5, C4), r2 * r2, p);
  if (abstop === 0) {
    if ((kiLo & 0x80000000) === 0) {
      // k > 0: the exponent of scale might have overflowed by <= 460
      const scale = asDouble((sHi + 0xc0f00000) >>> 0, sLo);
      return fma(scale, tmp, scale) * 5.486124068793689e+303; // 0x1p1009
    }
    // k < 0: rounded in two steps near the subnormal range, unfused as compiled
    const sHi2 = (sHi + 0x3fe00000) >>> 0;
    const scale = asDouble(sHi2, sLo);
    const st = scale * tmp;
    let y = scale + st;
    if (Math.abs(y) < 1.0) {
      const one = y < 0.0 ? -1.0 : 1.0;
      let lo = (scale - y) + st;
      const hi = y + one;
      lo = ((one - hi) + y) + lo;
      y = ((lo + hi) - one);
      if (y === 0) y = (sHi2 >>> 31) ? -0 : 0;
    }
    return y * 2.2250738585072014e-308; // 0x1p-1022
  }
  const scale = asDouble(sHi, sLo);
  return fma(tmp, scale, scale);
};

/** The whole computation with every fused step (ix normalized as e_pow.c leaves it). */
const powExact = (ixHi: number, ixLo: number, y: number, signBias: number): number => {
  logExact(ixHi, ixLo);
  const hi = LOG[0], lo = LOG[1];
  const ehi = y * hi;
  const elo = fma(y, lo, fma(y, hi, -ehi));
  return expExact(ehi, elo, signBias);
};

/**
 * The fast path of pow for x > 0 normal (ix with the sign cleared) and y in the main range, in one function so nothing
 * is boxed between steps (see the header for the reasoning; the fused steps are inlined copies of fmaF).
 */
const powFast = (ixHi: number, ixLo: number, y: number, signBias: number): number => {
  // log_inline: x = 2^k z, z in [OFF, 2 OFF)
  const tmpHi = (ixHi - 0x3fe69555) | 0;
  const li = (tmpHi >>> 13) & 127;
  const k = tmpHi >> 20;
  U[1] = (ixHi - (tmpHi & 0xfff00000)) >>> 0;
  U[0] = ixLo;
  const z = F[0];
  const invc = TABL[3 * li], logc = TABL[3 * li + 1], logctail = TABL[3 * li + 2];
  // z invc - 1 is exactly representable: (pz - 1) is exact (Sterbenz) and adding the product error rounds to it
  const pz = z * invc;
  let q1 = SPLIT * z;
  const qah1 = q1 - (q1 - z), qal1 = z - qah1;
  q1 = SPLIT * invc;
  const qbh1 = q1 - (q1 - invc), qbl1 = invc - qbh1;
  const ez = ((qah1 * qbh1 - pz) + qah1 * qbl1 + qal1 * qbh1) + qal1 * qbl1;
  const r = (pz - 1.0) + ez;
  const t1 = k * Ln2hi + logc; // exact product (Ln2hi has 11 trailing zero bits): the same rounding as fused
  const t2 = t1 + r;
  const lo1 = k * Ln2lo + logctail;
  const lo2 = (t1 - t2) + r;
  const ar = A0 * r;
  const ar2 = r * ar;
  const ar3 = r * ar2;
  const hi = t2 + ar2;
  let q2 = SPLIT * ar;
  const qah2 = q2 - (q2 - ar), qal2 = ar - qah2;
  q2 = SPLIT * r;
  const qbh2 = q2 - (q2 - r), qbl2 = r - qbh2;
  const lo3 = ((qah2 * qbh2 - ar2) + qah2 * qbl2 + qal2 * qbh2) + qal2 * qbl2;
  const lo4 = (t2 - hi) + ar2;
  const P = (A1 + r * A2) + ar2 * ((A3 + r * A4) + ar2 * (A5 + r * A6));
  const lo = ar3 * P + (((lo1 + lo2) + lo3) + lo4);
  // lo differs from the fused lo by a few ulps of its terms at most; 2^-50 of their magnitudes is a generous bound
  const dev = (Math.abs(ar3) + Math.abs(lo1) + Math.abs(lo2) + Math.abs(lo3) + Math.abs(lo4) + Math.abs(lo))
    * 8.881784197001252e-16 + 2.6e-26; // 2^-50, 2^-85
  const lh = hi + lo;
  // lh = RN(hi + lo) is libm's unless hi + lo lies within dev of a rounding boundary (RN is monotonic)
  const lbb = lh - hi;
  const le = (hi - (lh - lbb)) + (lo - lbb);
  if (lh + (le + dev) !== lh || lh + (le - dev) !== lh) return powExact(ixHi, ixLo, y, signBias);
  const ltail = (hi - lh) + lo;
  // ehi = y lh (the same product as libm), elo = y ltail + (y lh - ehi)
  const ehi = y * lh;
  const yl = y * ltail;
  let q3 = SPLIT * y;
  const qah3 = q3 - (q3 - y), qal3 = y - qah3;
  q3 = SPLIT * lh;
  const qbh3 = q3 - (q3 - lh), qbl3 = lh - qbh3;
  const eerr = ((qah3 * qbh3 - ehi) + qah3 * qbl3 + qal3 * qbh3) + qal3 * qbl3;
  const elo = yl + eerr;
  // elo differs from libm's fma(y, ltail, y lh - ehi) by |y| (dev + an ulp of the tail) and the roundings of y ltail and elo
  const eloDev = Math.abs(y) * (dev + Math.abs(ltail) * 2.220446049250313e-16)
    + (Math.abs(yl) + Math.abs(elo)) * 4.440892098500626e-16 + 1e-300;
  F[0] = ehi;
  const abstop = (U[1] >>> 20) & 0x7ff;
  if (((abstop - 0x3c9) >>> 0) >= 0x3f) {
    if (abstop < 0x3c9) { const one = 1.0 + ehi; return signBias ? -one : one; } // |ehi| < 2^-54
    if (abstop >= 0x409) return (U[1] >>> 31) ? uflow(signBias) : oflow(signBias);
    return powExact(ixHi, ixLo, y, signBias); // 512 <= |ehi| < 1024: the special cases
  }
  // exp_inline: kd is libm's unless ehi InvLn2N lies within 2^-30 of a half-integer
  const zz = ehi * InvLn2N;
  const ks = zz + Shift;
  const kd = ks - Shift;
  if (0.5 - Math.abs(zz - kd) < 9.313225746154785e-10) return powExact(ixHi, ixLo, y, signBias);
  // fma(kd, NegLn2loN, ehi + kd NegLn2hiN): the inner sum is exact (kd NegLn2hiN is exact, within a factor 2 of ehi)
  const a4 = kd, b4 = NegLn2loN, c4 = ehi + kd * NegLn2hiN;
  const p4 = a4 * b4;
  let s4 = SPLIT * a4;
  const ah4 = s4 - (s4 - a4), al4 = a4 - ah4;
  const bh4 = SLO, bl4 = SLO_L;
  const ul4 = ((ah4 * bh4 - p4) + ah4 * bl4 + al4 * bh4) + al4 * bl4;
  const th4 = c4 + p4;
  const bb4 = th4 - c4;
  const tl4 = (c4 - (th4 - bb4)) + (p4 - bb4);
  const v4 = tl4 + ul4;
  const vb4 = v4 - tl4;
  const e4 = (tl4 - (v4 - vb4)) + (ul4 - vb4);
  let rt = th4 + v4;
  if (e4 !== 0) {
    const rb4 = rt - th4;
    const d4 = (th4 - (rt - rb4)) + (v4 - rb4);
    if (d4 !== 0 && (rt + 2 * d4) - rt === 2 * d4) rt = fma(a4, b4, c4);
  }
  // er = RN(elo + rt) is libm's unless elo + rt lies within eloDev of a rounding boundary
  const er = elo + rt;
  const erb = er - elo;
  const ee = (elo - (er - erb)) + (rt - erb);
  if (er + (ee + eloDev) !== er || er + (ee - eloDev) !== er) return powExact(ixHi, ixLo, y, signBias);
  // the rest exactly as libm (the same r): polynomial steps and scale + scale tmp fused
  const i = kd & 127;
  const r2 = er * er;
  const a5 = er, b5 = C3, c5 = C2;
  const p5 = a5 * b5;
  let s5 = SPLIT * a5;
  const ah5 = s5 - (s5 - a5), al5 = a5 - ah5;
  const bh5 = SC2, bl5 = SC2_L;
  const ul5 = ((ah5 * bh5 - p5) + ah5 * bl5 + al5 * bh5) + al5 * bl5;
  const th5 = c5 + p5;
  const bb5 = th5 - c5;
  const tl5 = (c5 - (th5 - bb5)) + (p5 - bb5);
  const v5 = tl5 + ul5;
  const vb5 = v5 - tl5;
  const e5 = (tl5 - (v5 - vb5)) + (ul5 - vb5);
  let c32 = th5 + v5;
  if (e5 !== 0) {
    const rb5 = c32 - th5;
    const d5 = (th5 - (c32 - rb5)) + (v5 - rb5);
    if (d5 !== 0 && (c32 + 2 * d5) - c32 === 2 * d5) c32 = fma(a5, b5, c5);
  }
  const a6 = c32, b6 = r2, c6 = TAILD[i] + er;
  const p6 = a6 * b6;
  let s6 = SPLIT * a6;
  const ah6 = s6 - (s6 - a6), al6 = a6 - ah6;
  s6 = SPLIT * b6;
  const bh6 = s6 - (s6 - b6), bl6 = b6 - bh6;
  const ul6 = ((ah6 * bh6 - p6) + ah6 * bl6 + al6 * bh6) + al6 * bl6;
  const th6 = c6 + p6;
  const bb6 = th6 - c6;
  const tl6 = (c6 - (th6 - bb6)) + (p6 - bb6);
  const v6 = tl6 + ul6;
  const vb6 = v6 - tl6;
  const e6 = (tl6 - (v6 - vb6)) + (ul6 - vb6);
  let pp = th6 + v6;
  if (e6 !== 0) {
    const rb6 = pp - th6;
    const d6 = (th6 - (pp - rb6)) + (v6 - rb6);
    if (d6 !== 0 && (pp + 2 * d6) - pp === 2 * d6) pp = fma(a6, b6, c6);
  }
  const a7 = er, b7 = C5, c7 = C4;
  const p7 = a7 * b7;
  let s7 = SPLIT * a7;
  const ah7 = s7 - (s7 - a7), al7 = a7 - ah7;
  const bh7 = SC4, bl7 = SC4_L;
  const ul7 = ((ah7 * bh7 - p7) + ah7 * bl7 + al7 * bh7) + al7 * bl7;
  const th7 = c7 + p7;
  const bb7 = th7 - c7;
  const tl7 = (c7 - (th7 - bb7)) + (p7 - bb7);
  const v7 = tl7 + ul7;
  const vb7 = v7 - tl7;
  const e7 = (tl7 - (v7 - vb7)) + (ul7 - vb7);
  let c54 = th7 + v7;
  if (e7 !== 0) {
    const rb7 = c54 - th7;
    const d7 = (th7 - (c54 - rb7)) + (v7 - rb7);
    if (d7 !== 0 && (c54 + 2 * d7) - c54 === 2 * d7) c54 = fma(a7, b7, c7);
  }
  const a8 = c54, b8 = r2 * r2, c8 = pp;
  const p8 = a8 * b8;
  let s8 = SPLIT * a8;
  const ah8 = s8 - (s8 - a8), al8 = a8 - ah8;
  s8 = SPLIT * b8;
  const bh8 = s8 - (s8 - b8), bl8 = b8 - bh8;
  const ul8 = ((ah8 * bh8 - p8) + ah8 * bl8 + al8 * bh8) + al8 * bl8;
  const th8 = c8 + p8;
  const bb8 = th8 - c8;
  const tl8 = (c8 - (th8 - bb8)) + (p8 - bb8);
  const v8 = tl8 + ul8;
  const vb8 = v8 - tl8;
  const e8 = (tl8 - (v8 - vb8)) + (ul8 - vb8);
  let tmp = th8 + v8;
  if (e8 !== 0) {
    const rb8 = tmp - th8;
    const d8 = (th8 - (tmp - rb8)) + (v8 - rb8);
    if (d8 !== 0 && (tmp + 2 * d8) - tmp === 2 * d8) tmp = fma(a8, b8, c8);
  }
  let scale = SCALE2[i] * POW2[(kd - i) / 128 + POW2_OFF];
  if (signBias) scale = -scale;
  const a9 = tmp, b9 = scale, c9 = scale;
  const p9 = a9 * b9;
  let s9 = SPLIT * a9;
  const ah9 = s9 - (s9 - a9), al9 = a9 - ah9;
  s9 = SPLIT * b9;
  const bh9 = s9 - (s9 - b9), bl9 = b9 - bh9;
  const ul9 = ((ah9 * bh9 - p9) + ah9 * bl9 + al9 * bh9) + al9 * bl9;
  const th9 = c9 + p9;
  const bb9 = th9 - c9;
  const tl9 = (c9 - (th9 - bb9)) + (p9 - bb9);
  const v9 = tl9 + ul9;
  const vb9 = v9 - tl9;
  const e9 = (tl9 - (v9 - vb9)) + (ul9 - vb9);
  let res = th9 + v9;
  if (e9 !== 0) {
    const rb9 = res - th9;
    const d9 = (th9 - (res - rb9)) + (v9 - rb9);
    if (d9 !== 0 && (res + 2 * d9) - res === 2 * d9) res = fma(a9, b9, c9);
  }
  return res;
};

/** pow(x, y) exactly as the native binary's glibc computes it (see the header). */
export const pow = (x: number, y: number): number => {
  F[0] = x;
  let ixHi = U[1];
  const ixLo = U[0];
  F[0] = y;
  const iyHi = U[1], iyLo = U[0];
  let topx = ixHi >>> 20;
  const topy = iyHi >>> 20;
  let signBias = 0;
  if (((topx - 0x001) >>> 0) >= 0x7fe || (((topy & 0x7ff) - 0x3be) >>> 0) >= 0x80) {
    // special cases: (x < 0x1p-126 or inf or nan) or (|y| < 0x1p-65 or |y| >= 0x1p63 or nan)
    const yZeroInfNan = y === 0 || !Number.isFinite(y);
    if (yZeroInfNan) {
      if (y === 0) return 1.0;
      if (x === 1) return 1.0;
      if (Number.isNaN(x) || Number.isNaN(y)) return x + y;
      if (Math.abs(x) === 1) return 1.0;
      if ((Math.abs(x) < 1) === (y > 0)) return 0.0; // |x|<1 && y==inf or |x|>1 && y==-inf
      return y * y;
    }
    if (x === 0 || !Number.isFinite(x)) {
      let x2 = x * x;
      if ((ixHi >>> 31) && checkint(y) === 1) { x2 = -x2; signBias = 1; }
      if (x === 0 && y < 0) return signBias ? -Infinity : Infinity; // __math_divzero
      return y < 0 ? 1 / x2 : x2;
    }
    // here x and y are non-zero finite
    if (ixHi >>> 31) {
      // finite x < 0
      const yint = checkint(y);
      if (yint === 0) return NaN; // __math_invalid
      if (yint === 1) signBias = SIGN_BIAS;
      ixHi &= 0x7fffffff;
      topx &= 0x7ff;
    }
    if ((((topy & 0x7ff) - 0x3be) >>> 0) >= 0x80) {
      // note: signBias == 0 here because y is not odd
      if (ixHi === 0x3ff00000 && ixLo === 0) return 1.0;
      const ax = asDouble(ixHi, ixLo);
      if ((topy & 0x7ff) < 0x3be) return ax > 1.0 ? 1.0 + y : 1.0 - y; // |y| < 2^-65, x^y ~= 1 + y log(x)
      return (ax > 1.0) === (topy < 0x800) ? oflow(0) : uflow(0);
    }
    if (topx === 0) {
      // normalize subnormal x so the exponent becomes negative
      F[0] = asDouble(ixHi, ixLo) * 4503599627370496.0; // 0x1p52
      ixHi = ((U[1] & 0x7fffffff) - 0x03400000) >>> 0; // ix -= 52 << 52
      return powExact(ixHi, U[0], y, signBias);
    }
  }
  return powFast(ixHi, ixLo, y, signBias);
};
