/*
 * Support code for the styles ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the C
 * library cbrt() that the native binary calls (Ubuntu glibc 2.35, x86_64): sysdeps/ieee754/dbl-64/s_cbrt.c, Copyright
 * (C) 1997-2022 Free Software Foundation, Inc. The installed libm.so.6 compiles it without fused multiply-adds (read
 * from its disassembly: frexp, the polynomial in mulsd/addsd/subsd, u*u*u, the quotient and ldexp), so this file
 * reproduces it bit for bit with plain double arithmetic. glibc is LGPL-2.1-or-later; used under its section 3 as
 * GPL-2.0 (src/engine-gpl/LICENSE).
 */

/* s_cbrt.c: "#define CBRT2 1.2599210498948731648 / 2^(1/3)" and "#define SQR_CBRT2 1.5874010519681994748 / 2^(2/3)" */
const CBRT2 = 1.2599210498948731648;
const SQR_CBRT2 = 1.5874010519681994748;
const FACTOR = [1.0 / SQR_CBRT2, 1.0 / CBRT2, 1.0, CBRT2, SQR_CBRT2];

const F = new Float64Array(1);
const U = new Uint32Array(F.buffer); // little-endian: U[0] low word, U[1] high word

/** frexp(|x|): the mantissa in [0.5, 1) and the exponent (exponent 0 for zero, infinities and NaN, as glibc's). */
const frexpAbs = (x: number): { m: number; e: number } => {
  F[0] = Math.abs(x);
  let hx = U[1];
  let ix = hx & 0x7fffffff;
  let e = 0;
  if (ix >= 0x7ff00000 || (ix | U[0]) === 0) return { m: F[0] + F[0], e: 0 };
  if (ix < 0x00100000) { // subnormal
    F[0] *= 18014398509481984.0; // 2^54
    hx = U[1];
    ix = hx & 0x7fffffff;
    e = -54;
  }
  e += (ix >>> 20) - 1022;
  U[1] = (hx & 0x800fffff) | 0x3fe00000;
  return { m: F[0], e };
};

/** ldexp(y, n) for a result in the normal range or below it (rounded as the hardware scaling does). */
const ldexp = (y: number, n: number): number => {
  // two steps keep each factor finite; 2^n with |n| <= 1023 is exact
  if (n > 1023) return y * 2 ** 1023 * 2 ** (n - 1023);
  if (n < -1022) return y * 2 ** -1022 * 2 ** (n + 1022);
  return y * 2 ** n;
};

/** glibc __cbrt. */
export const cbrt = (x: number): number => {
  // Reduce X. XM now is an range 1.0 to 0.5.
  const { m: xm, e: xe } = frexpAbs(x);

  // If X is not finite or is null return it.
  if (xe === 0 && (Number.isNaN(x) || !Number.isFinite(x) || x === 0)) return x + x;

  const u = 0.354895765043919860
    + (1.50819193781584896
      + (-2.11499494167371287
        + (2.44693122563534430
          + (-1.83469277483613086
            + (0.784932344976639262 - 0.145263899385486377 * xm)
            * xm)
          * xm)
        * xm)
      * xm)
    * xm;

  const t2 = u * u * u;

  // C % truncates toward zero, so 2 + xe % 3 is 0..4
  const ym = u * (t2 + 2.0 * xm) / (2.0 * t2 + xm) * FACTOR[2 + (xe % 3)];

  return ldexp(x > 0.0 ? ym : -ym, Math.trunc(xe / 3));
};
