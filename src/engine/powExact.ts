/*
 * Correctly rounded x^y for the lattice scale factor.
 *
 * Math.pow is not required to be correctly rounded, and JavaScript engines differ in the last bit: for
 * x = 2.5, y = 1/3, Node 24 (and C's pow) give 1.3572088082974532 while Node 18-22 give 1.3572088082974534. The
 * lattice spacing scales every created position, and per-atom random streams are seeded from the exact bits of
 * the coordinates (nativeRandom.ts), so a one-ulp difference in the scale gives different velocities and
 * orientations for most atoms. This function returns the double nearest to the exact x^y (y as the given
 * double), the same in every engine.
 *
 * Method (textbook): start from Math.pow, which is within a few ulps; a candidate c is the correctly rounded
 * result when x^y lies between the midpoints to its neighbours, i.e. ln(mid_lo) <= y ln(x) < ln(mid_hi). The
 * logarithms are evaluated in 200-bit fixed point with BigInt (ln m = 2 atanh((m-1)/(m+1)) after scaling m into
 * [1, 2)), far more precision than the comparison needs.
 */

const FRAC = 200n;
const ONE = 1n << FRAC;

/** The exact value of a finite double as a BigInt in FRAC-bit fixed point (truncated below 2^-FRAC). */
const toFixed = (v: number): bigint => {
  const buf = new DataView(new ArrayBuffer(8));
  buf.setFloat64(0, v);
  const bits = buf.getBigUint64(0);
  const exp = Number((bits >> 52n) & 0x7ffn);
  const frac = bits & ((1n << 52n) - 1n);
  const mant = exp === 0 ? frac : frac | (1n << 52n);
  const e2 = (exp === 0 ? 1 : exp) - 1075; // v = mant * 2^e2
  const shift = BigInt(e2) + FRAC;
  return shift >= 0n ? mant << shift : mant >> -shift;
};

/** atanh(t) for 0 <= t <= 1/3 in fixed point, by its series t + t^3/3 + t^5/5 + ... */
const atanhFixed = (t: bigint): bigint => {
  const t2 = (t * t) >> FRAC;
  let term = t, sum = 0n;
  for (let k = 1n; term !== 0n; k += 2n) {
    sum += term / k;
    term = (term * t2) >> FRAC;
  }
  return sum;
};

const LN2 = 2n * atanhFixed(ONE / 3n);

/** ln(v) for a positive fixed-point value. */
const lnFixed = (v: bigint): bigint => {
  // v = m * 2^e with m in [1, 2)
  const e = v.toString(2).length - 1 - Number(FRAC);
  const m = e >= 0 ? v >> BigInt(e) : v << BigInt(-e);
  const t = ((m - ONE) << FRAC) / (m + ONE);
  return BigInt(e) * LN2 + 2n * atanhFixed(t);
};

const view = new DataView(new ArrayBuffer(8));
/** The next double above (dir 1) or below (dir -1) a positive finite double. */
const nextDouble = (v: number, dir: 1 | -1): number => {
  view.setFloat64(0, v);
  view.setBigUint64(0, view.getBigUint64(0) + BigInt(dir));
  return view.getFloat64(0);
};

/**
 * x^y rounded to the nearest double, for finite x > 0 and finite y whose result lies in [2^-100, 2^100] (the
 * lattice scales); outside that range, or for other arguments, Math.pow.
 */
export const powExact = (x: number, y: number): number => {
  const r0 = Math.pow(x, y);
  if (!(x > 0) || !Number.isFinite(x) || !Number.isFinite(y) || !(r0 > 2 ** -100 && r0 < 2 ** 100)) return r0;
  const z = (toFixed(y) * lnFixed(toFixed(x))) >> FRAC;
  let c = r0;
  for (let step = 0; step < 8; step++) {
    const lo = nextDouble(c, -1), hi = nextDouble(c, 1);
    const cf = toFixed(c);
    const midLo = (toFixed(lo) + cf) >> 1n, midHi = (cf + toFixed(hi)) >> 1n;
    if (z < lnFixed(midLo)) c = lo;
    else if (z >= lnFixed(midHi)) c = hi;
    else return c;
  }
  return c;
};
