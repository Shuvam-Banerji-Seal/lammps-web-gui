/*
 * Correctly rounded sin and cos for the molecule rotations of fix deposit.
 *
 * Math.sin and Math.cos are not required to be correctly rounded, and engines differ in the last bit. A molecule's
 * rotation quaternion takes cos and sin of half its angle; a one-ulp difference moves some atoms by one ulp, and an
 * atom pair at exactly the cutoff distance (a square lattice's 3-spacing pairs) then falls on the other side of the
 * cutoff, as it does in native LAMMPS, which uses the C library (correctly rounded for these arguments in practice).
 * This is the same approach as powExact.ts: the series is evaluated in 200-bit fixed point with BigInt, and the
 * result is rounded once to the nearest double.
 */

const FRAC = 200n;

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
  const signed = shift >= 0n ? mant << shift : mant >> -shift;
  return v < 0 ? -signed : signed;
};

/** The double nearest to the fixed-point value s / 2^FRAC (Number(bigint) rounds to nearest; the scaling is exact). */
const fromFixed = (s: bigint): number => Number(s) / 2 ** Number(FRAC);

/** Fixed-point multiply and divide by a small integer. */
const mulF = (a: bigint, b: bigint): bigint => (a * b) >> FRAC;

/**
 * sin(x) and cos(x) by their Taylor series in fixed point; the arguments of the molecule rotations are at most 2pi
 * in magnitude, so the series needs about 60 terms at this precision.
 */
const series = (x: number): { sin: bigint; cos: bigint } => {
  const X = toFixed(x);
  const X2 = mulF(X, X);
  let term = X; // x^(2k+1) / (2k+1)! with sign handled by the recurrence
  let sin = X;
  let k = 0;
  while (term !== 0n) {
    k++;
    term = -mulF(term, X2) / BigInt((2 * k) * (2 * k + 1));
    sin += term;
  }
  let cterm = 1n << FRAC;
  let cos = cterm;
  k = 0;
  while (cterm !== 0n) {
    k++;
    cterm = -mulF(cterm, X2) / BigInt((2 * k - 1) * (2 * k));
    cos += cterm;
  }
  return { sin, cos };
};

/** The correctly rounded sine of a double (for the rotation angles of fix deposit). */
export const sinExact = (x: number): number => (x === 0 || !Number.isFinite(x) ? Math.sin(x) : fromFixed(series(x).sin));

/** The correctly rounded cosine of a double (for the rotation angles of fix deposit). */
export const cosExact = (x: number): number => (!Number.isFinite(x) ? Math.cos(x) : fromFixed(series(x).cos));
