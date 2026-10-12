/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source files (commit b332f28):
 *   src/GRANSURF/pair_surf_granular.cpp (PairSurfGranular::compute, the contact loop of the sphere-surface pair),
 *   src/GRANSURF/surf_extra.cpp (SurfExtra::overlap_sphere_line),
 *   src/GRANSURF/fix_surface.cpp (FixSurface::point_connection2d, contact_compare, contact_presort, contact_sort),
 *   src/GRANULAR/granular_model.cpp (GranularModel::check_contact, calculate_forces; SURFACE contact type),
 *   src/GRANULAR/gran_sub_mod_normal.cpp (hooke), src/GRANULAR/gran_sub_mod_tangential.cpp (linear_history),
 *   src/GRANULAR/gran_sub_mod_damping.cpp (mass_velocity), src/GRANULAR/gran_sub_mod.cpp (rotate_rescale_vec).
 * Contributing authors of pair_surf_granular.cpp: Joel Clemmer (SNL), Dan Bolintineanu (SNL).
 * Copyright (2003) Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Browser-engine difference: the granular sub-models are written out for the subset the surface pair supports
 * (hooke normal, linear_history tangential, mass_velocity damping); the other models are refused by the style
 * parser (gransurf/parse.ts) with a StyleError.
 */

/** Connection flags (fix_surface.h: FLAT/NONFLAT, CONCAVE/CONVEX, SAME_SIDE/OPPOSITE_SIDE). */
export const NONFLAT = 0;
export const FLAT = 1;
export const CONCAVE = 0;
export const CONVEX = 1;
export const SAME_SIDE = 0;
export const OPPOSITE_SIDE = 1;
/** Surface-element exterior flags (pair_surf_granular.cpp: INTERNAL, EXTERNAL, UNCONNECTED). */
export const INTERNAL = 0;
export const EXTERNAL = 1;
export const UNCONNECTED = 2;

/** pair_surf_granular.cpp: EPSILON = 1e-12; fix_surface.cpp: SORTEPSILON = 1e-12; BIG = 1e20. */
export const EPSILON = 1e-12;
export const SORTEPSILON = 1e-12;
export const BIG = 1.0e20;
/** granular/gran_sub_mod_tangential.cpp: EPSILON = 1e-10 (linear_history frame-update threshold). */
const TANGENTIAL_EPSILON = 1e-10;

/** Flip of a connection side (pair_surf_granular.cpp: FLIPSIDE). */
export const flipSide = (nside: number): number => (nside === OPPOSITE_SIDE ? SAME_SIDE : OPPOSITE_SIDE);

/** Coefficients of the surface contact: normal hooke (k, damp), tangential linear_history (kt, xt, mu). */
export interface SurfCoeffs {
  k: number;
  damp: number;
  kt: number;
  xt: number;
  mu: number;
}

/** Overlap of a sphere with a line segment in 2d (surf_extra.cpp: overlap_sphere_line). */
export interface LineOverlap {
  flag: number;
  pt: [number, number, number];
  r: [number, number, number];
  rsq: number;
}

/**
 * Nearest point of a segment P1-P2 to the sphere centre (surf_extra.cpp overlap_sphere_line): flag -1 for P1, -2 for
 * P2 and 1 for the interior; 0 when the sphere does not reach the segment (the return value is then 0).
 */
export function overlapSphereLine(
  xs: ArrayLike<number>, xsOff: number, radius: number, p1: ArrayLike<number>, p2: ArrayLike<number>,
): LineOverlap | null {
  const ax = xs[xsOff] - p1[0], ay = xs[xsOff + 1] - p1[1], az = xs[xsOff + 2] - p1[2];
  const bx = p2[0] - p1[0], by = p2[1] - p1[1], bz = p2[2] - p1[2];
  // alpha = fraction of P1P2 at the projection of the centre on the infinite line
  const alpha = (ax * bx + ay * by + az * bz) / (bx * bx + by * by + bz * bz);
  let flag: number;
  let pt: [number, number, number];
  if (alpha <= 0.0) {
    flag = -1;
    pt = [p1[0], p1[1], p1[2]];
  } else if (alpha >= 1.0) {
    flag = -2;
    pt = [p2[0], p2[1], p2[2]];
  } else {
    flag = 1;
    pt = [p1[0] + alpha * bx, p1[1] + alpha * by, p1[2] + alpha * bz];
  }
  const r: [number, number, number] = [xs[xsOff] - pt[0], xs[xsOff + 1] - pt[1], xs[xsOff + 2] - pt[2]];
  const rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
  if (rsq < radius * radius) return { flag, pt, r, rsq };
  return null;
}

/** Nearest point of a segment to a point (surf_extra.cpp: nearest_point_line): -1/-2 end points, 1 interior. */
function nearestPointLine(x: ArrayLike<number>, xOff: number, p1: ArrayLike<number>, p2: ArrayLike<number>): { flag: number; pt: [number, number, number] } {
  const ax = x[xOff] - p1[0], ay = x[xOff + 1] - p1[1], az = x[xOff + 2] - p1[2];
  const bx = p2[0] - p1[0], by = p2[1] - p1[1], bz = p2[2] - p1[2];
  const alpha = (ax * bx + ay * by + az * bz) / (bx * bx + by * by + bz * bz);
  if (alpha <= 0.0) return { flag: -1, pt: [p1[0], p1[1], p1[2]] };
  if (alpha >= 1.0) return { flag: -2, pt: [p2[0], p2[1], p2[2]] };
  return { flag: 1, pt: [p1[0] + alpha * bx, p1[1] + alpha * by, p1[2] + alpha * bz] };
}

/**
 * Nearest point of a triangle to a sphere centre (surf_extra.cpp: overlap_sphere_tri): flags -1/-2/-3 for the edges
 * 12/23/31, -4/-5/-6 for the corners 1/2/3, 1 for the interior; null when the sphere does not reach the triangle.
 */
export function overlapSphereTri(
  xs: ArrayLike<number>, xsOff: number, radius: number, p1: ArrayLike<number>, p2: ArrayLike<number>,
  p3: ArrayLike<number>, norm: ArrayLike<number>,
): LineOverlap | null {
  const alpha = (xs[xsOff] - p1[0]) * norm[0] + (xs[xsOff + 1] - p1[1]) * norm[1] + (xs[xsOff + 2] - p1[2]) * norm[2];
  let pt: [number, number, number] = [xs[xsOff] - alpha * norm[0], xs[xsOff + 1] - alpha * norm[1], xs[xsOff + 2] - alpha * norm[2]];
  // the projected point is inside an edge when the edge cross product points along the normal
  // surf_extra.cpp uses its own EPSILON = 1e-14 for the edge tests
  const TRI_EPSILON = 1e-14;
  const side = (a: ArrayLike<number>, b: ArrayLike<number>, q: ArrayLike<number>): { dot: number; xn: number } => {
    const e = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const pv = [pt[0] - q[0], pt[1] - q[1], pt[2] - q[2]];
    const xp = [e[1] * pv[2] - e[2] * pv[1], e[2] * pv[0] - e[0] * pv[2], e[0] * pv[1] - e[1] * pv[0]];
    return { dot: xp[0] * norm[0] + xp[1] * norm[1] + xp[2] * norm[2], xn: xp[0] * xp[0] + xp[1] * xp[1] + xp[2] * xp[2] };
  };
  let inside = true;
  let e12 = false, e23 = false, e31 = false;
  let o12 = false, o23 = false, o31 = false;
  let sd = side(p1, p2, p1);
  if (sd.dot <= 0.0 && sd.xn > TRI_EPSILON) {
    o12 = true;
    if (sd.dot === 0.0) e12 = true;
    else inside = false;
  }
  sd = side(p2, p3, p2);
  if (sd.dot <= 0.0 && sd.xn > TRI_EPSILON) {
    o23 = true;
    if (sd.dot === 0.0) e23 = true;
    else inside = false;
  }
  sd = side(p3, p1, p3);
  if (sd.dot <= 0.0 && sd.xn > TRI_EPSILON) {
    o31 = true;
    if (sd.dot === 0.0) e31 = true;
    else inside = false;
  }
  let flag = 0;
  if (inside) {
    flag = 1;
    const esum = (e12 ? 1 : 0) + (e23 ? 1 : 0) + (e31 ? 1 : 0);
    if (esum) {
      if (esum === 1) {
        if (e12) flag = -1;
        else if (e23) flag = -2;
        else flag = -3;
      } else if (!e12) flag = -6;
      else if (!e23) flag = -4;
      else flag = -5;
    }
  } else {
    const osum = (o12 ? 1 : 0) + (o23 ? 1 : 0) + (o31 ? 1 : 0);
    if (osum === 1) {
      if (o12) {
        const l = nearestPointLine(xs, xsOff, p1, p2);
        pt = l.pt;
        flag = l.flag === 1 ? -1 : l.flag === -1 ? -4 : -5;
      } else if (o23) {
        const l = nearestPointLine(xs, xsOff, p2, p3);
        pt = l.pt;
        flag = l.flag === 1 ? -2 : l.flag === -1 ? -5 : -6;
      } else {
        const l = nearestPointLine(xs, xsOff, p3, p1);
        pt = l.pt;
        flag = l.flag === 1 ? -3 : l.flag === -1 ? -6 : -4;
      }
    } else if (!o12) {
      let l = nearestPointLine(xs, xsOff, p2, p3);
      if (l.flag === 1) {
        pt = l.pt;
        flag = -2;
      } else {
        l = nearestPointLine(xs, xsOff, p3, p1);
        if (l.flag === 1) {
          pt = l.pt;
          flag = -3;
        } else {
          pt = [p3[0], p3[1], p3[2]];
          flag = -6;
        }
      }
    } else if (!o23) {
      let l = nearestPointLine(xs, xsOff, p3, p1);
      if (l.flag === 1) {
        pt = l.pt;
        flag = -3;
      } else {
        l = nearestPointLine(xs, xsOff, p1, p2);
        if (l.flag === 1) {
          pt = l.pt;
          flag = -1;
        } else {
          pt = [p1[0], p1[1], p1[2]];
          flag = -4;
        }
      }
    } else {
      let l = nearestPointLine(xs, xsOff, p1, p2);
      if (l.flag === 1) {
        pt = l.pt;
        flag = -1;
      } else {
        l = nearestPointLine(xs, xsOff, p2, p3);
        if (l.flag === 1) {
          pt = l.pt;
          flag = -2;
        } else {
          pt = [p2[0], p2[1], p2[2]];
          flag = -5;
        }
      }
    }
  }
  const r: [number, number, number] = [xs[xsOff] - pt[0], xs[xsOff + 1] - pt[1], xs[xsOff + 2] - pt[2]];
  const rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
  if (rsq < radius * radius) return { flag, pt, r, rsq };
  return null;
}

/** The two-normal connection flags of two lines sharing endpoint iwhich/jwhich (fix_surface.cpp: point_connection2d). */
export function pointConnection2d(
  inorm: ArrayLike<number>, jnorm: ArrayLike<number>, iwhich: number, jwhich: number,
  flatthresh: number,
): { fflag: number; nside: number; aflag: number } {
  const dotnorm = inorm[0] * jnorm[0] + inorm[1] * jnorm[1] + inorm[2] * jnorm[2];
  const fflag = Math.abs(dotnorm) > 1.0 - flatthresh ? FLAT : NONFLAT;
  // icrossj = inorm x jnorm (only the z component is used)
  const icrossjz = inorm[0] * jnorm[1] - inorm[1] * jnorm[0];
  const nside = jwhich === iwhich ? OPPOSITE_SIDE : SAME_SIDE;
  const upward = jwhich === 0 ? icrossjz : -icrossjz;
  return { fflag, nside, aflag: upward > 0.0 ? CONCAVE : CONVEX };
}

/** One particle-surface contact of the current step (fix_surface.h: ContactSurf). */
export interface ContactSurf {
  /** Local (or ghost) index of the surface atom in the pair's arrays: the tie-break of the sorts. */
  index: number;
  /** Atom ID of the surface element (the key of the connectivity and of the contact history). */
  tag: number;
  type: number;
  /** Contact position on the element: 1 = interior point, -1/-2 = end points (2d); 3d: edges -1..-3, corners -4..-6. */
  flag: number;
  /** Distance from the contact point to the sphere centre. */
  rmag: number;
  nside: number;
  external: number;
  priority: number;
  convexIndex: number;
  overlap: number;
  rsqCom: number;
  weight: number;
  /** Unit vector from the contact point to the sphere centre. */
  dr: [number, number, number];
  /** Surface normal, sign set by the pre-walk. */
  surfNorm: [number, number, number];
  drForce: [number, number, number];
  /** Tags of the surface elements connected at the contact point, and their aflags. */
  cindex: number[];
  caflag: number[];
  /** End points of the element (2d). */
  ends: [[number, number, number], [number, number, number]];
  /** Corner points of the triangle (3d). */
  corners: [[number, number, number], [number, number, number], [number, number, number]];
  /** Corner-connected elements on the edge (cwhich: the corner index on the other element, 3d). */
  cwhich: number[];
  /** Neighbour chosen for the first and second contact edge (3d), -1 when none. */
  ck1: number;
  ck2: number;
  caflag1: number;
  caflag2: number;
}

/** fix_surface.cpp: contact_compare; signed selects the signed (sort) or absolute (presort) normal alignment. */
export const contactLess = (a: ContactSurf, b: ContactSurf, signed: boolean): boolean => {
  if (a.overlap > b.overlap + SORTEPSILON) return true;
  if (b.overlap > a.overlap + SORTEPSILON) return false;
  if (a.priority > b.priority) return true;
  if (b.priority > a.priority) return false;
  let dota = a.surfNorm[0] * a.dr[0] + a.surfNorm[1] * a.dr[1] + a.surfNorm[2] * a.dr[2];
  let dotb = b.surfNorm[0] * b.dr[0] + b.surfNorm[1] * b.dr[1] + b.surfNorm[2] * b.dr[2];
  if (!signed) {
    dota = Math.abs(dota);
    dotb = Math.abs(dotb);
  }
  if (dota > dotb + SORTEPSILON) return true;
  if (dotb > dota + SORTEPSILON) return false;
  if (a.rsqCom < b.rsqCom - SORTEPSILON) return true;
  if (b.rsqCom < a.rsqCom - SORTEPSILON) return false;
  return a.index < b.index;
};

/** Hooke normal, linear_history tangential and mass_velocity damping of one sphere-surface contact (SURFACE type). */
export interface SurfContactIn {
  /** Sphere centre minus contact point (dx = xi - xj). */
  dx: ArrayLike<number>;
  radi: number;
  vi: ArrayLike<number>;
  omegai: ArrayLike<number>;
  /** Velocity of the contact point. */
  vj: ArrayLike<number>;
  meff: number;
  dt: number;
  historyUpdate: boolean;
}

/** Output of one contact: total force on the sphere and its torque. */
export interface SurfContactOut {
  forces: [number, number, number];
  torque: [number, number, number];
}

/** gran_sub_mod.cpp: GranSubMod::rotate_rescale_vec. */
function rotateRescale(v: Float64Array, n: ArrayLike<number>, off: number): void {
  const rsht = v[off] * n[0] + v[off + 1] * n[1] + v[off + 2] * n[2];
  const shrmag = Math.hypot(v[off], v[off + 1], v[off + 2]);
  for (let a = 0; a < 3; a++) v[off + a] -= rsht * n[a];
  const prjmag = Math.hypot(v[off], v[off + 1], v[off + 2]);
  const temp = prjmag > 0 ? shrmag / prjmag : 0;
  for (let a = 0; a < 3; a++) v[off + a] *= temp;
}

/**
 * GranularModel::check_contact + calculate_forces for the SURFACE contact type, with hooke, linear_history and
 * mass_velocity. hist is the 3-value tangential history (updated in place when historyUpdate is set).
 */
export function surfContactForces(p: SurfCoeffs, c: SurfContactIn, hist: Float64Array, out: SurfContactOut): void {
  const { dx, radi, vi, omegai, vj, meff, dt, historyUpdate } = c;
  // radsum = radi and Reff = radi for a surface contact (radj = 0); delta = radi - r
  const r = Math.hypot(dx[0], dx[1], dx[2]);
  const rinv = 1.0 / r;
  const delta = radi - r;
  const nx = [dx[0] * rinv, dx[1] * rinv, dx[2] * rinv];
  const vr = [vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2]];
  const vnnr = vr[0] * nx[0] + vr[1] * nx[1] + vr[2] * nx[2];
  const vt = [vr[0] - vnnr * nx[0], vr[1] - vnnr * nx[1], vr[2] - vnnr * nx[2]];
  // wr = radi * omegai (the contact point is a fixed surface with zero angular velocity)
  const wr = [radi * omegai[0], radi * omegai[1], radi * omegai[2]];
  const wxn = [wr[1] * nx[2] - wr[2] * nx[1], wr[2] * nx[0] - wr[0] * nx[2], wr[0] * nx[1] - wr[1] * nx[0]];
  const vtr = [vt[0] - wxn[0], vt[1] - wxn[1], vt[2] - wxn[2]];
  // hooke normal force, mass_velocity damping (damp_prefactor = damp * meff)
  const dampPre = p.damp * meff;
  const fnormal = p.k * delta;
  const fdampN = -dampPre * vnnr;
  const fntot = fnormal + fdampN;
  const fncrit = Math.abs(fntot);
  // linear_history tangential: damp = xt * damp_prefactor; Fscrit = |Fn| mu
  const kt = p.kt;
  const damp = p.xt * dampPre;
  const fscrit = fncrit * p.mu;
  if (historyUpdate) {
    const rsht = hist[0] * nx[0] + hist[1] * nx[1] + hist[2] * nx[2];
    if (Math.abs(rsht) * kt > TANGENTIAL_EPSILON * fscrit) rotateRescale(hist, nx, 0);
    for (let a = 0; a < 3; a++) hist[a] += dt * vtr[a];
  }
  const fs = [0, 0, 0];
  for (let a = 0; a < 3; a++) fs[a] = -kt * hist[a];
  const fdamp = [-damp * vtr[0], -damp * vtr[1], -damp * vtr[2]];
  for (let a = 0; a < 3; a++) fs[a] += fdamp[a];
  const magfs = Math.hypot(fs[0], fs[1], fs[2]);
  if (magfs > fscrit) {
    const shrmag = Math.hypot(hist[0], hist[1], hist[2]);
    if (shrmag !== 0.0) {
      const scale = fscrit / magfs;
      for (let a = 0; a < 3; a++) fs[a] *= scale;
      // the elastic part of the rescaled force gives the new history (k is removed again)
      const inv = -1.0 / kt;
      for (let a = 0; a < 3; a++) hist[a] = (fs[a] - fdamp[a]) * inv;
    } else {
      fs[0] = fs[1] = fs[2] = 0;
    }
  }
  for (let a = 0; a < 3; a++) out.forces[a] = fntot * nx[a] + fs[a];
  // torquesi = -(nx x fs) * radi
  const cx = nx[1] * fs[2] - nx[2] * fs[1], cy = nx[2] * fs[0] - nx[0] * fs[2], cz = nx[0] * fs[1] - nx[1] * fs[0];
  out.torque[0] = -cx * radi;
  out.torque[1] = -cy * radi;
  out.torque[2] = -cz * radi;
}
