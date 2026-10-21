/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/fix_wall_region.cpp (FixWallRegion::init, the coefficients and the offset of style colloid, and
 * FixWallRegion::colloid) at commit b332f28, with MathSpecial::powint (src/math_special.h). Copyright Sandia
 * Corporation (LAMMPS development team, GNU General Public License); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix wall/region style colloid (docs.lammps.org/fix_wall_region.html: "For style colloid, the energy E is given by an
 * integrated form of the pair_style colloid potential"): the force magnitude fwall and the energy eng (the offset at
 * the cutoff subtracted) of a finite-size particle of radius rad whose centre is at distance r from the wall surface.
 * The offset is the one native sets for this style, which has the form of the lj93 offset.
 */
import { log } from '../libm/log';

/** math_special.h powint(x, n): repeated squaring for integer n (the operation order of the source). */
const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

export interface ColloidWall {
  coeff1: number;
  coeff2: number;
  coeff3: number;
  coeff4: number;
  offset: number;
}

/** FixWallRegion::init for style colloid: the coefficients of epsilon and sigma, and the offset at the cutoff. */
export const colloidWall = (epsilon: number, sigma: number, cutoff: number): ColloidWall => {
  const coeff1 = -4.0 / 315.0 * epsilon * powint(sigma, 6);
  const coeff2 = -2.0 / 3.0 * epsilon;
  const coeff3 = epsilon * powint(sigma, 6) / 7560.0;
  const coeff4 = epsilon / 6.0;
  const rinv = 1.0 / cutoff;
  const r2inv = rinv * rinv;
  const r4inv = r2inv * r2inv;
  const offset = coeff3 * r4inv * r4inv * rinv - coeff4 * r2inv * rinv;
  return { coeff1, coeff2, coeff3, coeff4, offset };
};

/** FixWallRegion::colloid(r, rad): the force magnitude along the surface normal and the shifted energy. */
export const colloidWallForce = (w: ColloidWall, r: number, rad: number): { fwall: number; eng: number } => {
  const newCoeff2 = w.coeff2 * rad * rad * rad;
  const diam = 2.0 * rad;

  const rad2 = rad * rad;
  const rad4 = rad2 * rad2;
  const rad8 = rad4 * rad4;
  const delta2 = rad2 - r * r;
  const rinv = 1.0 / delta2;
  const r2inv = rinv * rinv;
  const r4inv = r2inv * r2inv;
  const r8inv = r4inv * r4inv;
  const fwall = w.coeff1 *
      (rad8 * rad + 27.0 * rad4 * rad2 * rad * r * r + 63.0 * rad4 * rad * powint(r, 4) +
        21.0 * rad2 * rad * powint(r, 6)) *
      r8inv -
    newCoeff2 * r2inv;

  const r2 = 0.5 * diam - r;
  const rinv2 = 1.0 / r2;
  const r2inv2 = rinv2 * rinv2;
  const r4inv2 = r2inv2 * r2inv2;
  const r3 = r + 0.5 * diam;
  const rinv3 = 1.0 / r3;
  const r2inv3 = rinv3 * rinv3;
  const r4inv3 = r2inv3 * r2inv3;
  const eng = w.coeff3 *
      ((-3.5 * diam + r) * r4inv2 * r2inv2 * rinv2 +
        (3.5 * diam + r) * r4inv3 * r2inv3 * rinv3) -
    w.coeff4 * ((-diam * r + r2 * r3 * (log(-r2) - log(r3))) * (-rinv2) * rinv3) - w.offset;
  return { fwall, eng };
};
