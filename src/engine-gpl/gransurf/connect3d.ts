/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source files (commit b332f28): src/GRANSURF/fix_surface_local.cpp (FixSurfaceLocal::connectivity3d_complete and the
 * edge/corner neighbour classes of connectivity3d_local, epsilon_calculate for 3d), src/GRANSURF/fix_surface.cpp
 * (FixSurface::edge_connection3d, corner_connection3d), src/GRANSURF/surf_extra.cpp (SurfExtra::overlap_sphere_tri,
 * nearest_point_line; see surf_contact.ts), src/GRANSURF/pair_surf_granular.cpp (calculate_corners, corners2norm).
 * Copyright (2003) Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Browser-engine difference: the triangles that touch a corner point are found by comparing the corner points of all
 * owned triangles (quadratic in the number of triangles, once per run) instead of the bins of connectivity3d_local.
 * Neighbour lists hold atom IDs in owned-index order.
 */
import { quatToMatrix } from '../../engine/compute/asphere';
import { StyleError } from '../../engine/force/types';
import { EXTERNAL, FLAT, INTERNAL, UNCONNECTED, CONCAVE, CONVEX, OPPOSITE_SIDE, SAME_SIDE } from './surf_contact';

/** Tolerance for matching corner points, as a fraction of the smallest triangle diameter (fix_surface_local.cpp). */
const MATCH_EPSILON = 0.001;

type V3 = [number, number, number];

/** A triangle sharing an edge with this one (Connect3d: neigh_e, aflag_e, fflag_e, nside_e). */
export interface EdgeNeighbour {
  tag: number;
  nside: number;
  aflag: number;
  fflag: number;
}

/** A triangle sharing only a corner point with this one (Connect3d: neigh_c, fflag_c, nside_c). */
export interface CornerNeighbour {
  tag: number;
  nside: number;
  fflag: number;
}

/** Connectivity of one triangle: edges 1-2, 2-3, 3-1 and corners 1, 2, 3 (Connect3d). */
export interface Connect3d {
  e: [EdgeNeighbour[], EdgeNeighbour[], EdgeNeighbour[]];
  c: [CornerNeighbour[], CornerNeighbour[], CornerNeighbour[]];
  externalEdge: [number, number, number];
  externalCor: [number, number, number];
}

/** fix_surface.cpp: FixSurface::edge_connection3d (jpfirst/jpsecond are the 1-based corners matching the edge ends). */
export function edgeConnection3d(
  inorm: V3, jnorm: V3, iedge: V3, jpfirst: number, jpsecond: number, flatthresh: number,
): { fflag: number; nside: number; aflag: number } {
  if (jpfirst < 0 || jpsecond < 0) throw new StyleError('Inconsistent surface connectivity');
  const dotnorm = inorm[0] * jnorm[0] + inorm[1] * jnorm[1] + inorm[2] * jnorm[2];
  const fflag = Math.abs(dotnorm) > 1.0 - flatthresh ? FLAT : 0;
  const icrossj: V3 = [
    inorm[1] * jnorm[2] - inorm[2] * jnorm[1], inorm[2] * jnorm[0] - inorm[0] * jnorm[2], inorm[0] * jnorm[1] - inorm[1] * jnorm[0],
  ];
  const d = icrossj[0] * iedge[0] + icrossj[1] * iedge[1] + icrossj[2] * iedge[2];
  const cyclic = (jpfirst === 1 && jpsecond === 2) || (jpfirst === 2 && jpsecond === 3) || (jpfirst === 3 && jpsecond === 1);
  if (cyclic) return { fflag, nside: OPPOSITE_SIDE, aflag: d > 0.0 ? CONCAVE : CONVEX };
  return { fflag, nside: SAME_SIDE, aflag: d < 0.0 ? CONCAVE : CONVEX };
}

/** fix_surface.cpp: FixSurface::corner_connection3d. */
export function cornerConnection3d(inorm: V3, jnorm: V3, cwhich: number, flatthresh: number): { fflag: number; nside: number } {
  if (cwhich < 0) throw new StyleError('Inconsistent surface connectivity');
  const dotnorm = inorm[0] * jnorm[0] + inorm[1] * jnorm[1] + inorm[2] * jnorm[2];
  return {
    fflag: Math.abs(dotnorm) > 1.0 - flatthresh ? FLAT : 0,
    nside: dotnorm < 0.0 ? OPPOSITE_SIDE : SAME_SIDE,
  };
}

/** Corner points and unit normal of a triangle at atom i (pair_surf_granular.cpp: calculate_corners, corners2norm). */
export function triCorners(
  x: ArrayLike<number>, i: number, o: number, quat: ArrayLike<number>, bonus: ArrayLike<number>, cols: number,
): { c: [V3, V3, V3]; norm: V3 } {
  // i: the position (x) of the atom, o: its owner (quaternion and body corners)
  const R = quatToMatrix(quat[4 * o], quat[4 * o + 1], quat[4 * o + 2], quat[4 * o + 3]);
  const corner = (k: number): V3 => {
    const b = [bonus[cols * o + 3 * k], bonus[cols * o + 3 * k + 1], bonus[cols * o + 3 * k + 2]];
    return [0, 1, 2].map((r) => x[3 * i + r] + R[3 * r] * b[0] + R[3 * r + 1] * b[1] + R[3 * r + 2] * b[2]) as V3;
  };
  const c: [V3, V3, V3] = [corner(0), corner(1), corner(2)];
  return { c, norm: normOf(c) };
}

/** Unit normal of the triangle (p2 - p1) x (p3 - p1) (corners2norm). */
export function normOf(c: [V3, V3, V3]): V3 {
  const a = [c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]];
  const b = [c[2][0] - c[0][0], c[2][1] - c[0][1], c[2][2] - c[0][2]];
  const n: V3 = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = Math.hypot(n[0], n[1], n[2]);
  return [n[0] / len, n[1] / len, n[2] / len];
}

/**
 * Connectivity of all owned triangles. tris lists the owned atom indices of the surface triangles; quat is the
 * per-atom quaternion and bonus the TRI_BONUS body corners (cols values per atom); radius is the atom radius
 * (the circumradius of a triangle), used for the matching tolerance.
 */
export function buildConnect3d(
  x: ArrayLike<number>, tris: number[], quat: ArrayLike<number>, bonus: ArrayLike<number>, cols: number,
  radius: ArrayLike<number>, tags: ArrayLike<number>, flatthresh: number,
): Map<number, Connect3d> {
  const pts = tris.map((i) => triCorners(x, i, i, quat, bonus, cols));
  let minRad = Infinity;
  for (const i of tris) minRad = Math.min(minRad, radius[i]);
  const eps = 2.0 * minRad * MATCH_EPSILON;
  const epssq = eps * eps;
  const same = (a: V3, b: V3): boolean => {
    const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz < epssq;
  };
  // the triangles (indices into tris) that have a corner point equal to corner k of triangle a
  const touching = (a: number, k: number): number[] => {
    const out: number[] = [];
    for (let b = 0; b < tris.length; b++) {
      if (b === a) continue;
      if (pts[b].c.some((p) => same(pts[a].c[k], p))) out.push(b);
    }
    return out;
  };
  // the 1-based corner of triangle b equal to point p, or -1
  const match1 = (b: number, p: V3): number => {
    for (let k = 0; k < 3; k++) if (same(p, pts[b].c[k])) return k + 1;
    return -1;
  };
  const match0 = (b: number, p: V3): number => match1(b, p) - 1;
  const out = new Map<number, Connect3d>();
  for (let a = 0; a < tris.length; a++) {
    const pa = pts[a];
    const N = [touching(a, 0), touching(a, 1), touching(a, 2)];
    const inSet = (s: number[], b: number): boolean => s.includes(b);
    const e1 = N[0].filter((b) => inSet(N[1], b));
    const e2 = N[1].filter((b) => inSet(N[2], b));
    const e3 = N[2].filter((b) => inSet(N[0], b));
    const c1 = N[0].filter((b) => !inSet(N[1], b) && !inSet(N[2], b));
    const c2 = N[1].filter((b) => !inSet(N[2], b) && !inSet(N[0], b));
    const c3 = N[2].filter((b) => !inSet(N[0], b) && !inSet(N[1], b));
    const iedge: [V3, V3, V3] = [
      [pa.c[1][0] - pa.c[0][0], pa.c[1][1] - pa.c[0][1], pa.c[1][2] - pa.c[0][2]],
      [pa.c[2][0] - pa.c[1][0], pa.c[2][1] - pa.c[1][1], pa.c[2][2] - pa.c[1][2]],
      [pa.c[0][0] - pa.c[2][0], pa.c[0][1] - pa.c[2][1], pa.c[0][2] - pa.c[2][2]],
    ];
    // the edge pairs: (first, second) corner of this triangle and the matching corners of the other one
    const edgeCorners: [number, number][] = [[0, 1], [1, 2], [2, 0]];
    const edgeLists = [e1, e2, e3];
    const e: [EdgeNeighbour[], EdgeNeighbour[], EdgeNeighbour[]] = [[], [], []];
    for (let s = 0; s < 3; s++) {
      const [ka, kb] = edgeCorners[s];
      for (const b of edgeLists[s]) {
        const jf = match1(b, pa.c[ka]), js = match1(b, pa.c[kb]);
        const conn = edgeConnection3d(pa.norm, pts[b].norm, iedge[s], jf, js, flatthresh);
        e[s].push({ tag: tags[tris[b]], nside: conn.nside, aflag: conn.aflag, fflag: conn.fflag });
      }
    }
    const cornerLists = [c1, c2, c3];
    const cc: [CornerNeighbour[], CornerNeighbour[], CornerNeighbour[]] = [[], [], []];
    for (let k = 0; k < 3; k++) {
      for (const b of cornerLists[k]) {
        const conn = cornerConnection3d(pa.norm, pts[b].norm, match0(b, pa.c[k]), flatthresh);
        cc[k].push({ tag: tags[tris[b]], nside: conn.nside, fflag: conn.fflag });
      }
    }
    const externalEdge: [number, number, number] = [INTERNAL, INTERNAL, INTERNAL];
    for (let s = 0; s < 3; s++) {
      for (const nb of e[s]) if (nb.fflag !== FLAT) externalEdge[s] = EXTERNAL;
      if (e[s].length === 0) externalEdge[s] = UNCONNECTED;
    }
    const externalCor: [number, number, number] = [
      Math.max(externalEdge[0], externalEdge[2]), Math.max(externalEdge[0], externalEdge[1]), Math.max(externalEdge[1], externalEdge[2]),
    ];
    out.set(tags[tris[a]], { e, c: cc, externalEdge, externalCor });
  }
  return out;
}
