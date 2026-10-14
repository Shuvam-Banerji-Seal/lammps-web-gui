/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/min_linesearch.cpp at commit b332f28, MinLineSearch::linemin_quadratic and the constants above it
 * (ALPHA_MAX 1.0, ALPHA_REDUCE 0.5, BACKTRACK_SLOPE 0.4, QUADRATIC_TOL 0.1, EMACH 1.0e-8, EPS_QUAD 1.0e-28), and the
 * normalisation of fdothall and fh by the atom count when thermo norm is on. LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2; this file is GPL-2.0 too (see
 * src/engine-gpl/LICENSE). Original authors: the LAMMPS development team (Sandia National Laboratories).
 *
 * Not reproduced: src/min_linesearch.cpp divides fdothall and fh by the atom count when output->thermo->normflag is
 * set (thermo norm, default on in lj units). Measured with native LAMMPS (black box): thermo_modify norm yes and no
 * give the same line search (fnorm of every row equal to 1e-15, lj cluster and 4 waters), so the division is not
 * applied here.
 *
 * The default line search of min_style cg and sd (min_modify line quadratic). The core minimizer (run/min.ts) owns the
 * state: it stores the start point, moves the atoms and the box DOF for a trial alpha and evaluates the energy and the
 * force there. This routine decides the alpha, in the same order of trials as native.
 */
import type { QuadraticLinesearch, QuadraticResult } from '../../engine/run/min';

const ALPHA_MAX = 1.0;
const ALPHA_REDUCE = 0.5;
const BACKTRACK_SLOPE = 0.4;
const QUADRATIC_TOL = 0.1;
const EMACH = 1.0e-8;
const EPS_QUAD = 1.0e-28;

const dot = (a: Float64Array, b: Float64Array): number => {
  let d = 0;
  for (let k = 0; k < a.length; k++) d += a[k] * b[k];
  return d;
};

/**
 * native linemin_quadratic. The directional derivative is fdothall = f0.h at the start point; each trial gives
 * fh = f.h and the energy. The next trial is the secant (quadratic projection) step alpha0 when the
 * energy predicted by the quadratic model agrees with the energy (relerr <= QUADRATIC_TOL) and alpha0 is inside
 * (0, alphamax); it is accepted when it lowers the energy by less than EMACH. Otherwise the step is accepted when it
 * meets the Armijo condition de <= -BACKTRACK_SLOPE alpha fdothall, and if not alpha is halved. Native checks the
 * condition with the energy of the last evaluated point, which is alpha0 when the projection was tried; that is kept.
 */
export const lineminQuadratic = (eoriginal: number, ls: QuadraticLinesearch): QuadraticResult => {
  const fdothall = dot(ls.f0, ls.h);
  // not a descent direction: native returns DOWNHILL and the minimizer stops
  if (fdothall <= 0) return { status: 'downhill', energy: eoriginal };

  let alpha = Math.min(ALPHA_MAX, ls.alphaMax);
  const alphamax = alpha;
  let fhprev = fdothall;
  let engprev = eoriginal;
  let alphaprev = 0;
  for (;;) {
    let ecurrent = ls.step(alpha);
    const fh = dot(ls.force, ls.h);
    const delfh = fh - fhprev;
    // the directional force (or its change) is at round-off: reset to the start point
    if (Math.abs(fh) < EPS_QUAD || Math.abs(delfh) < EPS_QUAD) return { status: 'zeroquad', energy: ls.step(0) };

    // secant step: alpha0 is the zero of the interpolated directional force
    const relerr = Math.abs(1.0 - (0.5 * (alpha - alphaprev) * (fh + fhprev) + ecurrent) / engprev);
    const alpha0 = alpha - (alpha - alphaprev) * fh / delfh;
    if (relerr <= QUADRATIC_TOL && alpha0 > 0.0 && alpha0 < alphamax) {
      ecurrent = ls.step(alpha0);
      if (ecurrent - eoriginal < EMACH) return { status: 'ok', energy: ecurrent };
    }

    const deIdeal = -BACKTRACK_SLOPE * alpha * fdothall;
    const de = ecurrent - eoriginal;
    if (de <= deIdeal) return { status: 'ok', energy: ecurrent };

    fhprev = fh;
    engprev = ecurrent;
    alphaprev = alpha;
    alpha *= ALPHA_REDUCE;
    // backtracked to zero: reset to the start point
    if (alpha <= 0.0 || deIdeal >= -EMACH) return { status: 'zeroalpha', energy: ls.step(0) };
  }
};
