/*
 * erfc(x) for the real-space Ewald/PPPM sum, accurate to ~1e-12 absolute:
 * a table built once from the Taylor series of erf (x < 2, no cancellation
 * problem there) and the continued fraction for erfc (x >= 2, Abramowitz &
 * Stegun 7.1.14), read with cubic Hermite interpolation using the exact
 * derivative erfc'(x) = -2/sqrt(pi) exp(-x^2). Beyond the table erfc < 1e-17.
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** Direct evaluation (slow, for the table). */
export const erfcExact = (x: number): number => {
  if (x < 0) return 2 - erfcExact(-x);
  if (x < 2) {
    // erf(x) = 2/sqrt(pi) sum_n (-1)^n x^(2n+1) / (n! (2n+1))
    let term = x, sum = x;
    const x2 = x * x;
    for (let n = 1; n < 200; n++) {
      term *= -x2 / n;
      const t = term / (2 * n + 1);
      sum += t;
      if (Math.abs(t) < 1e-18 * Math.abs(sum)) break;
    }
    return 1 - TWO_OVER_SQRTPI * sum;
  }
  // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
  let f = x;
  for (let k = 400; k >= 1; k--) f = x + (k / 2) / f;
  return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
};

const XMAX = 6.25;
const N = 1600;            // h = 1/256
const H = XMAX / N;
const table = new Float64Array(N + 2);
for (let k = 0; k <= N + 1; k++) table[k] = erfcExact(k * H);

/** erfc(x) for x >= 0, given ex = exp(-x^2) (callers need it for the force anyway). */
export const erfcFast = (x: number, ex: number): number => {
  if (x >= XMAX) return 0;
  const u = x / H;
  const k = u | 0;
  const t = u - k;
  const x0 = k * H;
  const f0 = table[k], f1 = table[k + 1];
  const d0 = -TWO_OVER_SQRTPI * Math.exp(-x0 * x0) * H;
  const d1 = -TWO_OVER_SQRTPI * Math.exp(-(x0 + H) * (x0 + H)) * H;
  void ex;
  const t2 = t * t, t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * f0 + (t3 - 2 * t2 + t) * d0 + (-2 * t3 + 3 * t2) * f1 + (t3 - t2) * d1;
};

export const EWALD_F = TWO_OVER_SQRTPI;
