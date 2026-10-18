/*
 * The C library functions exp, log, sin, cos, erf, erfc and acos as the native binary computes them. The engine calls libm.exp,
 * libm.log, libm.sin and libm.cos instead of Math.exp, Math.log, Math.sin and Math.cos, which differ from the C library (glibc) in the last bit for
 * a few percent of arguments; a one-ulp difference in a random number or a rotation changes chaotic trajectories and
 * the atoms that land exactly on a cutoff. The exact versions are ports of glibc in the GPL module
 * (src/engine-gpl/libm, reached through the lazy group gpl_libm of lazyStyles.ts); Session.execute loads them before
 * its first command, and their module installs them here. Code run outside a session (some unit tests) gets the
 * JavaScript functions, the same mathematics to within one ulp.
 */
export const libm: {
  exp: (x: number) => number; log: (x: number) => number; sin: (x: number) => number; cos: (x: number) => number;
  erf: (x: number) => number; erfc: (x: number) => number; acos: (x: number) => number;
} = {
  exp: Math.exp,
  log: Math.log,
  sin: Math.sin,
  cos: Math.cos,
  // JavaScript has no erf or erfc: the engine's own series below until the session installs the C library's
  erf: (x: number) => erfSeries(x),
  erfc: (x: number) => erfcExact(x),
  acos: Math.acos,
};

/**
 * sqrt(pi) rounded once to double, the value native LAMMPS uses (MY_PIS in src/math_const.h at b332f28).
 * Math.sqrt(Math.PI) is one ulp below it: it takes the root of pi already rounded.
 */
export const SQRT_PI = 1.77245385090551602729;

/** erf(x) by its Maclaurin series for |x| <= 2.2 and 1 - erfc beyond; libm.erf outside a session. */
export const erfSeries = (x: number): number => {
  if (Number.isNaN(x)) return x;
  const ax = Math.abs(x);
  if (ax > 2.2) return x > 0 ? 1 - erfcExact(ax) : erfcExact(ax) - 1;
  // erf(x) = 2/sqrt(pi) sum_n (-1)^n x^(2n+1) / (n! (2n+1))
  let term = x, sum = x;
  const x2 = x * x;
  for (let n = 1; n < 200; n++) {
    term *= -x2 / n;
    const add = term / (2 * n + 1);
    sum += add;
    if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
  }
  return (2 / Math.sqrt(Math.PI)) * sum;
};

/**
 * erfc(x) evaluated directly: the Taylor series of erf for x < 2 and the continued fraction of erfc beyond
 * (Abramowitz & Stegun 7.1.14). Slow; used for force/erfc.ts's table and as libm.erfc outside a session.
 */
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
    return 1 - (2 / Math.sqrt(Math.PI)) * sum;
  }
  // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
  let f = x;
  for (let k = 400; k >= 1; k--) f = x + (k / 2) / f;
  return libm.exp(-x * x) / Math.sqrt(Math.PI) / f;
};
