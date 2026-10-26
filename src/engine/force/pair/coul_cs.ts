import { erfcFast } from '../erfc';
import { libm } from '../../libm';

/*
 * erf(x) without cancellation at small x, for the damped Coulomb terms of born/coul/wolf, born/coul/dsf (born_coul18.ts)
 * and buck6d/coul/gauss (buck6d_gauss.ts). The core-shell styles (/cs) are GPL ports in src/engine-gpl/csdiel/.
 */

/* Exact 2/sqrt(pi) of the erf series. */
const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** erf(x) for x >= 0 without cancellation at small x (power series below 1). */
export const erfAcc = (x: number): number => {
  if (x < 1) {
    // erf(x) = 2/sqrt(pi) sum_n (-1)^n x^(2n+1) / (n! (2n+1))
    const x2 = x * x;
    let term = x, sum = x;
    for (let n = 1; n < 40; n++) {
      term *= -x2 / n;
      const t = term / (2 * n + 1);
      sum += t;
      if (Math.abs(t) < 1e-18 * Math.abs(sum)) break;
    }
    return TWO_OVER_SQRTPI * sum;
  }
  return 1 - erfcFast(x, libm.exp(-x * x));
};
