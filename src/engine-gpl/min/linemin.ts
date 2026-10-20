/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/min_linesearch.cpp at commit b332f28, MinLineSearch::linemin_quadratic, linemin_forcezero and the constants above them
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
import type { ForcezeroLinesearch, QuadraticLinesearch, QuadraticResult } from '../../engine/run/min';

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

/**
 * native linemin_forcezero (min_modify line forcezero), with its constants: the search moves along h by increments
 * of alpha until the directional derivative f.h has fallen to GRAD_TOL = 0.1 of its start. A step that raises the
 * energy by ZERO_ENERGY = 1e-12 or more, or that changes the sign of f.h, is taken back and the increment reduced
 * (by the zero of the linearised f.h, or by ALPHA_FACT = 0.1); otherwise the increment grows by the linearised f.h,
 * at most LIMIT_BOOST = 4. The first increment is min(max(EPS_QUAD, 0.1 |E| / f.h), alpha_max / 2); the search
 * ends successfully when the next alpha would pass alpha_max (the state stays at the last evaluated point), and fails
 * (ZEROALPHA, the start point restored) when hmax * increment <= MIN_ALPHA_FAC = 1e-14. The energies and f.h are
 * divided by the atom count when thermo norm is on, as native's energy_force and compute_dir_deriv do: that changes
 * the ZERO_ENERGY test. After a backtrack native leaves the atoms at the step it took back while its ecurrent is the
 * energy before it; the returned energy is that ecurrent.
 */
export const lineminForcezero = (eoriginal: number, ls: ForcezeroLinesearch): QuadraticResult => {
  const ALPHA_FACT = 0.1;
  const ZERO_ENERGY = 1e-12;
  const GRAD_TOL = 0.1;
  const MIN_ALPHA_FAC = 1e-14;
  const LIMIT_BOOST = 4.0;
  const norm = (v: number): number => (ls.natoms !== null ? v / ls.natoms : v);
  const fdothall = norm(dot(ls.f0, ls.h));
  if (fdothall <= 0.0) return { status: 'downhill', energy: eoriginal };
  const alphaMax = ls.alphaMax;
  let fhCurr = fdothall;
  const fhoriginal = fhCurr;
  let engCurr = norm(eoriginal);
  /** native's ecurrent, as the caller's (unnormalised) energy */
  let ecurrent = eoriginal;
  const alphaInit = Math.max(EPS_QUAD, 0.1 * Math.abs(norm(eoriginal)) / fdothall);
  let alpha = 0.0;
  let alphaDel = Math.min(alphaInit, 0.5 * alphaMax);
  for (;;) {
    let backtrack = false;
    const fhPrev = fhCurr;
    const engPrev = engCurr;
    const ecurrentPrev = ecurrent;
    alpha += alphaDel;
    if (alpha > alphaMax) {
      // done enough in the allowed search space: the state stays at the last evaluated point
      alpha -= alphaDel;
      return { status: 'ok', energy: ecurrent };
    }
    ecurrent = ls.step(alpha);
    engCurr = norm(ecurrent);
    fhCurr = norm(dot(ls.force, ls.h));
    const de = engCurr - engPrev;
    if (de >= ZERO_ENERGY) backtrack = true;
    if (!backtrack && Math.abs(fhCurr / fhoriginal) <= GRAD_TOL) return { status: 'ok', energy: ecurrent };
    if (fhCurr < 0.0) backtrack = true;
    if (backtrack) {
      alpha -= alphaDel;
      if (fhCurr < 0.0) alphaDel *= fhPrev / (fhPrev - fhCurr);
      else alphaDel *= ALPHA_FACT;
      engCurr = engPrev;
      ecurrent = ecurrentPrev;
      fhCurr = fhPrev;
      if (ls.hmax * alphaDel <= MIN_ALPHA_FAC) return { status: 'zeroalpha', energy: ls.step(0.0) };
    } else {
      let boostFactor = LIMIT_BOOST;
      if (fhPrev > fhCurr) boostFactor = fhCurr / (fhPrev - fhCurr);
      boostFactor = Math.min(boostFactor, LIMIT_BOOST);
      alphaDel *= boostFactor;
    }
  }
};
