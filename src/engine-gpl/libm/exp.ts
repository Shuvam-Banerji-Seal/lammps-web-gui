/*
 * Support code for the styles ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the C
 * library exp() that the native binary calls (Ubuntu glibc 2.35 on an x86_64 CPU with FMA and AVX2, where glibc
 * dispatches exp to __exp_fma: sysdeps/ieee754/dbl-64/e_exp.c and e_exp_data.c compiled with -mfma -mavx2; written by
 * Szabolcs Nagy, Copyright (C) 2018-2022 Free Software Foundation, Inc.). This file reproduces that function bit for
 * bit: the algorithm and table of e_exp.c, with a fused multiply-add exactly where the installed libm.so.6 has one
 * (read from its disassembly: InvLn2N x + Shift, both reduction steps, the three polynomial steps and the final
 * scale + scale tmp; the k < 0 special case keeps the unfused products of the source). The table was read from
 * libm.so.6. glibc is LGPL-2.1-or-later; used under its section 3 as GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * Fast path: in the main range (|x| < 709) the products kd NegLn2hiN are exact (NegLn2hiN has 17 trailing zero bits,
 * |kd| < 2^17), the unfused reduction and polynomial differ from the fused ones by less than 2^-60 in tmp, and kd is
 * the same unless x InvLn2N lies within 2^-30 of a half-integer; the final scale + scale tmp is decided from its exact
 * error terms with that 2^-60 (doubled) as margin. Anything closer goes through the fully fused computation. Checked
 * against the native libm (tests/engineLibmExp.test.ts and a 5,000,000-argument comparison): no difference.
 */
import { fma, fmaS, splitHi } from './log';

const F = new Float64Array(1);
const U = new Uint32Array(F.buffer); // little-endian: U[0] low word, U[1] high word
const SPLIT = 134217729;

export const InvLn2N = 184.6649652337873;
export const Shift = 6755399441055744.0;
export const NegLn2hiN = -0.005415212348111709;
export const NegLn2loN = -1.2864023111638346e-14;
export const C2 = 0.49999999999996786, C3 = 0.16666666666665886, C4 = 0.0416666808410674, C5 = 0.008333335853059549;
export const SINV = splitHi(InvLn2N), SHI = splitHi(NegLn2hiN), SLO = splitHi(NegLn2loN);
export const SC2 = splitHi(C3), SC4 = splitHi(C5);
/** __exp_data.tab (N = 128): for i, tab[2i] = tail bits and tab[2i+1] = scale bits, as (high word, low word) pairs. */
export const TAB = new Uint32Array([
  0x00000000, 0x00000000, 0x3ff00000, 0x00000000, 0x3c9b3b4f, 0x1a88bf6e, 0x3feff63d, 0xa9fb3335,
  0xbc716013, 0x9cd8dc5d, 0x3fefec9a, 0x3e778061, 0xbc905e7a, 0x108766d1, 0x3fefe315, 0xe86e7f85,
  0x3c8cd252, 0x3567f613, 0x3fefd9b0, 0xd3158574, 0xbc8bce80, 0x23f98efa, 0x3fefd06b, 0x29ddf6de,
  0x3c60f74e, 0x61e6c861, 0x3fefc745, 0x18759bc8, 0x3c90a3e4, 0x5b33d399, 0x3fefbe3e, 0xcac6f383,
  0x3c979aa6, 0x5d837b6d, 0x3fefb558, 0x6cf9890f, 0x3c8eb51a, 0x92fdeffc, 0x3fefac92, 0x2b7247f7,
  0x3c3ebe3d, 0x702f9cd1, 0x3fefa3ec, 0x32d3d1a2, 0xbc6a0334, 0x89906e0b, 0x3fef9b66, 0xaffed31b,
  0xbc955652, 0x2a2fbd0e, 0x3fef9301, 0xd0125b51, 0xbc5080ef, 0x8c4eea55, 0x3fef8abd, 0xc06c31cc,
  0xbc91c923, 0xb9d5f416, 0x3fef829a, 0xaea92de0, 0x3c80d3e3, 0xe95c55af, 0x3fef7a98, 0xc8a58e51,
  0xbc801b15, 0xeaa59348, 0x3fef72b8, 0x3c7d517b, 0xbc8f1ff0, 0x55de323d, 0x3fef6af9, 0x388c8dea,
  0x3c8b898c, 0x3f1353bf, 0x3fef635b, 0xeb6fcb75, 0xbc96d99c, 0x7611eb26, 0x3fef5be0, 0x84045cd4,
  0x3c9aecf7, 0x3e3a2f60, 0x3fef5487, 0x3168b9aa, 0xbc8fe782, 0xcb86389d, 0x3fef4d50, 0x22fcd91d,
  0x3c8a6f41, 0x44a6c38d, 0x3fef463b, 0x88628cd6, 0x3c807a05, 0xb0e4047d, 0x3fef3f49, 0x917ddc96,
  0x3c968efd, 0xe3a8a894, 0x3fef387a, 0x6e756238, 0x3c875e18, 0xf274487d, 0x3fef31ce, 0x4fb2a63f,
  0x3c80472b, 0x981fe7f2, 0x3fef2b45, 0x65e27cdd, 0xbc96b87b, 0x3f71085e, 0x3fef24df, 0xe1f56381,
  0x3c82f7e1, 0x6d09ab31, 0x3fef1e9d, 0xf51fdee1, 0xbc3d219b, 0x1a6fbffa, 0x3fef187f, 0xd0dad990,
  0x3c8b3782, 0x720c0ab4, 0x3fef1285, 0xa6e4030b, 0x3c6e1492, 0x89cecb8f, 0x3fef0caf, 0xa93e2f56,
  0x3c834d75, 0x4db0abb6, 0x3fef06fe, 0x0a31b715, 0x3c864201, 0xe2ac744c, 0x3fef0170, 0xfc4cd831,
  0x3c8fdd39, 0x5dd3f84a, 0x3feefc08, 0xb26416ff, 0xbc86a380, 0x3b8e5b04, 0x3feef6c5, 0x5f929ff1,
  0xbc924aed, 0xcc4b5068, 0x3feef1a7, 0x373aa9cb, 0xbc9907f8, 0x1b512d8e, 0x3feeecae, 0x6d05d866,
  0xbc71d1e8, 0x3e9436d2, 0x3feee7db, 0x34e59ff7, 0xbc991919, 0xb3ce1b15, 0x3feee32d, 0xc313a8e5,
  0x3c859f48, 0xa72a4c6d, 0x3feedea6, 0x4c123422, 0xbc931260, 0x7a28698a, 0x3feeda45, 0x04ac801c,
  0xbc58a78f, 0x4817895b, 0x3feed60a, 0x21f72e2a, 0xbc7c2c9b, 0x67499a1b, 0x3feed1f5, 0xd950a897,
  0x3c4363ed, 0x60c2ac11, 0x3feece08, 0x6061892d, 0x3c966609, 0x3b0664ef, 0x3feeca41, 0xed1d0057,
  0x3c6ecce1, 0xdaa10379, 0x3feec6a2, 0xb5c13cd0, 0x3c93ff8e, 0x3f0f1230, 0x3feec32a, 0xf0d7d3de,
  0x3c7690ce, 0xbb7aafb0, 0x3feebfda, 0xd5362a27, 0x3c931dbd, 0xeb54e077, 0x3feebcb2, 0x99fddd0d,
  0xbc8f9434, 0x0071a38e, 0x3feeb9b2, 0x769d2ca7, 0xbc87decc, 0xdc93a349, 0x3feeb6da, 0xa2cf6642,
  0xbc78dec6, 0xbd0f385f, 0x3feeb42b, 0x569d4f82, 0xbc861246, 0xec7b5cf6, 0x3feeb1a4, 0xca5d920f,
  0x3c933505, 0x18fdd78e, 0x3feeaf47, 0x36b527da, 0x3c7b98b7, 0x2f8a9b05, 0x3feead12, 0xd497c7fd,
  0x3c9063e1, 0xe21c5409, 0x3feeab07, 0xdd485429, 0x3c34c785, 0x5019c6ea, 0x3feea926, 0x8a5946b7,
  0x3c9432e6, 0x2b64c035, 0x3feea76f, 0x15ad2148, 0xbc8ce44a, 0x6199769f, 0x3feea5e1, 0xb976dc09,
  0xbc8c33c5, 0x3bef4da8, 0x3feea47e, 0xb03a5585, 0xbc845378, 0x892be9ae, 0x3feea346, 0x34ccc320,
  0xbc93cedd, 0x78565858, 0x3feea238, 0x82552225, 0x3c5710aa, 0x807e1964, 0x3feea155, 0xd44ca973,
  0xbc93b3ef, 0xbf5e2228, 0x3feea09e, 0x667f3bcd, 0xbc6a12ad, 0x8734b982, 0x3feea012, 0x750bdabf,
  0xbc6367ef, 0xb86da9ee, 0x3fee9fb2, 0x3c651a2f, 0xbc80dc3d, 0x54e08851, 0x3fee9f7d, 0xf9519484,
  0xbc781f64, 0x7e5a3ecf, 0x3fee9f75, 0xe8ec5f74, 0xbc86ee4a, 0xc08b7db0, 0x3fee9f9a, 0x48a58174,
  0xbc861932, 0x1e55e68a, 0x3fee9feb, 0x564267c9, 0x3c909ccb, 0x5e09d4d3, 0x3feea069, 0x4fde5d3f,
  0xbc7b32dc, 0xb94da51d, 0x3feea114, 0x73eb0187, 0x3c94ecfd, 0x5467c06b, 0x3feea1ed, 0x0130c132,
  0x3c65ebe1, 0xabd66c55, 0x3feea2f3, 0x36cf4e62, 0xbc88a1c5, 0x2fb3cf42, 0x3feea427, 0x543e1a12,
  0xbc9369b6, 0xf13b3734, 0x3feea589, 0x994cce13, 0xbc805e84, 0x3a19ff1e, 0x3feea71a, 0x4623c7ad,
  0xbc94d450, 0xd872576e, 0x3feea8d9, 0x9b4492ed, 0x3c90ad67, 0x5b0e8a00, 0x3feeaac7, 0xd98a6699,
  0x3c8db72f, 0xc1f0eab4, 0x3feeace5, 0x422aa0db, 0xbc65b660, 0x9cc5e7ff, 0x3feeaf32, 0x16b5448c,
  0x3c7bf683, 0x59f35f44, 0x3feeb1ae, 0x99157736, 0xbc93091f, 0xa71e3d83, 0x3feeb45b, 0x0b91ffc6,
  0xbc5da9b8, 0x8b6c1e29, 0x3feeb737, 0xb0cdc5e5, 0xbc6c23f9, 0x7c90b959, 0x3feeba44, 0xcbc8520f,
  0xbc924343, 0x22f4f9aa, 0x3feebd82, 0x9fde4e50, 0xbc85ca6c, 0xd7668e4b, 0x3feec0f1, 0x70ca07ba,
  0x3c71affc, 0x2b91ce27, 0x3feec491, 0x82a3f090, 0x3c6dd235, 0xe10a73bb, 0x3feec863, 0x19e32323,
  0xbc87c504, 0x22622263, 0x3feecc66, 0x7b5de565, 0x3c8b1c86, 0xe3e231d5, 0x3feed09b, 0xec4a2d33,
  0xbc91bbd1, 0xd3bcbb15, 0x3feed503, 0xb23e255d, 0x3c90cc31, 0x9cee31d2, 0x3feed99e, 0x1330b358,
  0x3c846984, 0x6e735ab3, 0x3feede6b, 0x5579fdbf, 0xbc82dfcd, 0x978e9db4, 0x3feee36b, 0xbfd3f37a,
  0x3c8c1a77, 0x92cb3387, 0x3feee89f, 0x995ad3ad, 0xbc907b8f, 0x4ad1d9fa, 0x3feeee07, 0x298db666,
  0xbc55c3d9, 0x56dcaeba, 0x3feef3a2, 0xb84f15fb, 0xbc90a40e, 0x3da6f640, 0x3feef972, 0x8de5593a,
  0xbc68d6f4, 0x38ad9334, 0x3feeff76, 0xf2fb5e47, 0xbc91eee2, 0x6b588a35, 0x3fef05b0, 0x30a1064a,
  0x3c74ffd7, 0x0a5fddcd, 0x3fef0c1e, 0x904bc1d2, 0xbc91bdfb, 0xfa9298ac, 0x3fef12c2, 0x5bd71e09,
  0x3c736eae, 0x30af0cb3, 0x3fef199b, 0xdd85529c, 0x3c8ee332, 0x5c9ffd94, 0x3fef20ab, 0x5fffd07a,
  0x3c84e08f, 0xd10959ac, 0x3fef27f1, 0x2e57d14b, 0x3c63cdaf, 0x384e1a67, 0x3fef2f6d, 0x9406e7b5,
  0x3c676b2c, 0x6c921968, 0x3fef3720, 0xdcef9069, 0xbc808a18, 0x83ccb5d2, 0x3fef3f0b, 0x555dc3fa,
  0xbc8fad5d, 0x3ffffa6f, 0x3fef472d, 0x4a07897c, 0xbc900dae, 0x3875a949, 0x3fef4f87, 0x080d89f2,
  0x3c74a385, 0xa63d07a7, 0x3fef5818, 0xdcfba487, 0xbc82919e, 0x2040220f, 0x3fef60e3, 0x16c98398,
  0x3c8e5a50, 0xd5c192ac, 0x3fef69e6, 0x03db3285, 0x3c843a59, 0xac016b4b, 0x3fef7321, 0xf301b460,
  0xbc82d521, 0x07b43e1f, 0x3fef7c97, 0x337b9b5f, 0xbc892ab9, 0x3b470dc9, 0x3fef8646, 0x14f5a129,
  0x3c74b604, 0x603a88d3, 0x3fef902e, 0xe78b3ff6, 0x3c83c5ec, 0x519d7271, 0x3fef9a51, 0xfbc74c83,
  0xbc8ff712, 0x8fd391f0, 0x3fefa4af, 0xa2a490da, 0xbc8dae98, 0xe223747d, 0x3fefaf48, 0x2d8e67f1,
  0x3c8ec3bc, 0x41aa2008, 0x3fefba1b, 0xee615a27, 0x3c842b94, 0xc3a9eb32, 0x3fefc52b, 0x376bba97,
  0x3c8a64a9, 0x31d185ee, 0x3fefd076, 0x5b6e4540, 0xbc8e37ba, 0xe43be3ed, 0x3fefdbfd, 0xad9cbe14,
  0x3c77893b, 0x4d91cd9d, 0x3fefe7c1, 0x819e90d8, 0x3c5305c1, 0x4160cc89, 0x3feff3c2, 0x2b8f71f1,
]);

export const asDouble = (hi: number, lo: number): number => { U[1] = hi; U[0] = lo; return F[0]; };

/** tail_i = asdouble(tab[2i]) and 2^(i/N) = asdouble(tab[2i+1] + (i << 45)) for the fast path. */
export const TAILD = new Float64Array(128);
export const SCALE2 = new Float64Array(128);
for (let i = 0; i < 128; i++) {
  TAILD[i] = asDouble(TAB[4 * i], TAB[4 * i + 1]);
  SCALE2[i] = asDouble((TAB[4 * i + 2] + (i << 13)) >>> 0, TAB[4 * i + 3]);
}
/** 2^m for |m| <= 1100 (exact powers of two). */
export const POW2_OFF = 1100;
export const POW2 = new Float64Array(2201);
for (let m = -1100; m <= 1100; m++) POW2[m + POW2_OFF] = 2 ** m;

/** The k < 0 special case of e_exp.c (unfused products, as compiled). */
const specialNeg = (tmp: number, sHi: number, sLo: number): number => {
  const scale = asDouble((sHi + 0x3fe00000) >>> 0, sLo);
  const st = scale * tmp;
  let y = scale + st;
  if (1.0 > y) {
    const hi = y + 1.0;
    let lo = (scale - y) + st;
    lo = ((1.0 - hi) + y) + lo;
    y = (lo + hi) - 1.0;
    if (y === 0) y = 0.0;
  }
  return y * 2.2250738585072014e-308; // 0x1p-1022
};

/** The k > 0 special case: the fused scale + scale tmp of the rescaled scale, then 2^1009. */
const specialPos = (tmp: number, sHi: number, sLo: number): number => {
  const scale = asDouble((sHi + 0xc0f00000) >>> 0, sLo);
  return fma(scale, tmp, scale) * 5.486124068793689e+303; // 0x1p1009
};

/** The fully fused computation (every step as libm.so.6 does it). */
const expExact = (x: number, special: boolean): number => {
  const ks = fmaS(x, InvLn2N, SINV, InvLn2N - SINV, Shift);
  F[0] = ks;
  const kiLo = U[0];
  const kd = ks - Shift;
  const r = fmaS(kd, NegLn2loN, SLO, NegLn2loN - SLO, fmaS(kd, NegLn2hiN, SHI, NegLn2hiN - SHI, x));
  const idx = 2 * (kiLo & 127);
  const tail = asDouble(TAB[2 * idx], TAB[2 * idx + 1]);
  const sHi = (TAB[2 * idx + 2] + (kiLo << 13)) >>> 0, sLo = TAB[2 * idx + 3];
  const r2 = r * r;
  const t1 = fmaS(r, C3, SC2, C3 - SC2, C2);
  const t2 = fmaS(r, C5, SC4, C5 - SC4, C4);
  const p = fma(t1, r2, tail + r);
  const tmp = fma(r2 * r2, t2, p);
  if (special) return (kiLo & 0x80000000) ? specialNeg(tmp, sHi, sLo) : specialPos(tmp, sHi, sLo);
  const scale = asDouble(sHi, sLo);
  return fma(scale, tmp, scale);
};

/** exp(x) exactly as the native binary's glibc computes it (see the header). */
export const exp = (x: number): number => {
  F[0] = x;
  const hi = U[1], lo = U[0];
  const abstop = (hi >>> 20) & 0x7ff;
  if (((abstop - 0x3c9) >>> 0) > 0x3e) {
    if (abstop < 0x3c9) return 1.0 + x; // |x| < 2^-54
    if (abstop > 0x408) { // |x| >= 1024, inf, nan
      if (hi === 0xfff00000 && lo === 0) return 0.0;
      if (abstop === 0x7ff) return 1.0 + x;
      return (hi >>> 31) ? 0 : Infinity;
    }
    return expExact(x, true); // 512 <= |x| < 1024: the special cases
  }
  // fast path (see the header)
  const z = x * InvLn2N;
  const ks = z + Shift;
  const kd = ks - Shift;
  if (0.5 - Math.abs(z - kd) < 9.313225746154785e-10) return expExact(x, false); // within 2^-30 of a half-integer
  const i = kd & 127;
  const r = (x + kd * NegLn2hiN) + kd * NegLn2loN;
  const r2 = r * r;
  const tmp = (TAILD[i] + r) + r2 * (C2 + r * C3) + r2 * r2 * (C4 + r * C5);
  // scale = 2^(i/N) 2^m, m = (k - i) / N: the table's scale bits plus k << 45 (exact for these k)
  const scale = SCALE2[i] * POW2[(kd - i) / 128 + POW2_OFF];
  // the fused y = RN(scale + scale tmp): the exact sum differs from scale + p by less than |scale| 2^-60 (the fused
  // tmp) + |p| 2^-53 (the rounding of p); M doubles that. RN is monotonic, so if y + es +- M both round to y, so does
  // the exact sum.
  const p = scale * tmp;
  const y = scale + p;
  const bb = y - scale;
  const es = (scale - (y - bb)) + (p - bb);
  const M = Math.abs(scale) * 1.734723475976807e-18 + Math.abs(p) * 2.220446049250313e-16;
  if (y + (es + M) === y && y + (es - M) === y) return y;
  return expExact(x, false);
};
