/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source files (commit b332f28): src/GRANSURF/fix_surface_local.cpp (FixSurfaceLocal::connectivity2d_complete,
 * epsilon_calculate, same_point, FLATTHRESH) and src/GRANSURF/fix_surface.cpp (point_connection2d).
 * Copyright (2003) Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Browser-engine difference: the neighbour lines of each line are found by comparing the end points of all owned
 * line atoms (quadratic in the number of lines, once per run) instead of the bins of assign2d. The neighbour lists
 * hold atom IDs. Periodic surfaces are refused by the pair (the end points are not unwrapped).
 */
import { EXTERNAL, FLAT, INTERNAL, UNCONNECTED, pointConnection2d } from './surf_contact';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** FixSurfaceLocal flat threshold default (fix_surface_local.cpp: FLATTHRESH). */
export const FLATTHRESH_2D = 0.00015230484360876085;
/** fix_surface_local.cpp: EPSILON = 0.001 (fraction of the shortest line for point matching). */
const MATCH_EPSILON = 0.001;

/** One line connected to an end point of a line (Connect2d: neigh_p1/p2, pwhich, nside, aflag, fflag). */
export interface Neighbour2d {
  tag: number;
  pwhich: number;
  nside: number;
  aflag: number;
  fflag: number;
}

/** Connectivity of one line (Connect2d): the lines at each end point and the exterior flags of the end points. */
export interface Connect2d {
  p: [Neighbour2d[], Neighbour2d[]];
  external: [number, number];
}

/** End points and unit normal of a line segment at atom i (pair_surf_granular.cpp: calculate_endpts). */
export function lineEnds(x: ArrayLike<number>, i: number, length: number, theta: number): {
  e0: [number, number, number]; e1: [number, number, number]; norm: [number, number, number];
} {
  const dx = 0.5 * length * libmCos(theta);
  const dy = 0.5 * length * libmSin(theta);
  const e0: [number, number, number] = [x[3 * i] - dx, x[3 * i + 1] - dy, 0.0];
  const e1: [number, number, number] = [x[3 * i] + dx, x[3 * i + 1] + dy, 0.0];
  // norm = normalize(zunit x (e1 - e0))
  const px = e1[0] - e0[0], py = e1[1] - e0[1];
  const nx = -py, ny = px;
  const len = Math.hypot(nx, ny);
  return { e0, e1, norm: [nx / len, ny / len, 0.0] };
}

/**
 * Connectivity of all owned lines. lines lists the owned atom indices of the surface lines; length and theta are
 * read by owned index (LINE_BONUS: 2 values per atom); tags are the atom IDs.
 */
export function buildConnect2d(
  x: ArrayLike<number>, lines: number[], bonus: ArrayLike<number>, tags: ArrayLike<number>, flatthresh: number,
): Map<number, Connect2d> {
  const ends = lines.map((i) => lineEnds(x, i, bonus[2 * i], bonus[2 * i + 1]));
  let minLen = Infinity;
  for (const i of lines) minLen = Math.min(minLen, bonus[2 * i]);
  const eps = minLen * MATCH_EPSILON;
  const epssq = eps * eps;
  const same = (a: [number, number, number], b: [number, number, number]): boolean => {
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz < epssq;
  };
  const out = new Map<number, Connect2d>();
  for (let a = 0; a < lines.length; a++) {
    const ea = ends[a];
    const p: [Neighbour2d[], Neighbour2d[]] = [[], []];
    for (let which = 0; which < 2; which++) {
      const pt = which === 0 ? ea.e0 : ea.e1;
      for (let b = 0; b < lines.length; b++) {
        if (b === a) continue;
        const eb = ends[b];
        // jwhich: 0 when the shared point is the first end point of line b, 1 for the second, -1 for none
        let jwhich = -1;
        if (same(pt, eb.e0)) jwhich = 0;
        else if (same(pt, eb.e1)) jwhich = 1;
        if (jwhich < 0) continue;
        const conn = pointConnection2d(ea.norm, eb.norm, which, jwhich, flatthresh);
        p[which].push({ tag: tags[lines[b]], pwhich: jwhich, nside: conn.nside, aflag: conn.aflag, fflag: conn.fflag });
      }
    }
    const external: [number, number] = [INTERNAL, INTERNAL];
    for (const nb of p[0]) if (nb.fflag !== FLAT) external[0] = EXTERNAL;
    for (const nb of p[1]) if (nb.fflag !== FLAT) external[1] = EXTERNAL;
    if (p[0].length === 0) external[0] = UNCONNECTED;
    if (p[1].length === 0) external[1] = UNCONNECTED;
    out.set(tags[lines[a]], { p, external });
  }
  return out;
}
