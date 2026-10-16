/*
 * The C library functions exp, log, sin and cos as the native binary computes them. The engine calls libm.exp,
 * libm.log, libm.sin and libm.cos instead of Math.exp, Math.log, Math.sin and Math.cos, which differ from the C library (glibc) in the last bit for
 * a few percent of arguments; a one-ulp difference in a random number or a rotation changes chaotic trajectories and
 * the atoms that land exactly on a cutoff. The exact versions are ports of glibc in the GPL module
 * (src/engine-gpl/libm, reached through the lazy group gpl_libm of lazyStyles.ts); Session.execute loads them before
 * its first command, and their module installs them here. Code run outside a session (some unit tests) gets the
 * JavaScript functions, the same mathematics to within one ulp.
 */
export const libm: { exp: (x: number) => number; log: (x: number) => number; sin: (x: number) => number; cos: (x: number) => number } = {
  exp: Math.exp,
  log: Math.log,
  sin: Math.sin,
  cos: Math.cos,
};
