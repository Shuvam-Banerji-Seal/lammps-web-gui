/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/GRAPHICS/image_objects.cpp, the
 * ConvexHullObj class (build, build_hull: the coplanar fan, the 3D Bowyer-Watson Delaunay triangulation with the
 * deterministic insertion order and point perturbation, the auto alpha from the mean nearest-neighbour distance, the
 * alpha-shape boundary and the smooth or flat normals) and the helpers vec3dot, vec3cross, vec3len, vec3norm and
 * compute_circumsphere. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public
 * License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The output is a list of triangles with their normals and, per vertex, the index of the input point it came from
 * (the caller maps that index to an atom). The C++ keeps the faces of its std::map in key order; the map is kept in
 * the same order here, so the triangle order is the C++ order as well.
 */
import type { Vec3 } from './objects';

export type Triangle = [Vec3, Vec3, Vec3];

export interface HullResult {
  triangles: Triangle[];
  normals: Triangle[];
  /** Index into the input point list for each vertex of each triangle. */
  colorIdx: Array<[number, number, number]>;
}

const SMALL = 1.0e-10;

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function len(v: Vec3): number {
  return Math.sqrt(dot(v, v));
}
/** vec3norm: "(n > 0.0) ? (1.0 / n) * v" (times the reciprocal, not divided by n: the rounding differs). */
function norm(v: Vec3): Vec3 {
  const n = len(v);
  if (!(n > 0.0)) return [0.0, 0.0, 0.0];
  const inv = 1.0 / n;
  return [inv * v[0], inv * v[1], inv * v[2]];
}
function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function scale(s: number, v: Vec3): Vec3 {
  return [s * v[0], s * v[1], s * v[2]];
}

/** compute_circumsphere: the circumcentre and squared radius of a tetrahedron; false when degenerate. */
function circumsphere(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3): { center: Vec3; radiusSq: number } | null {
  const a = sub(p1, p0);
  const b = sub(p2, p0);
  const c = sub(p3, p0);
  const a2 = dot(a, a);
  const b2 = dot(b, b);
  const c2 = dot(c, c);
  const bxc = cross(b, c);
  const cxa = cross(c, a);
  const axb = cross(a, b);
  const denom = 2.0 * dot(a, bxc);
  if (Math.abs(denom) < SMALL * SMALL) return null;
  const inv = 1.0 / denom;
  const rel: Vec3 = [
    (a2 * bxc[0] + b2 * cxa[0] + c2 * axb[0]) * inv,
    (a2 * bxc[1] + b2 * cxa[1] + c2 * axb[1]) * inv,
    (a2 * bxc[2] + b2 * cxa[2] + c2 * axb[2]) * inv,
  ];
  return { center: add(rel, p0), radiusSq: dot(rel, rel) };
}

/** The sorted triple used as the key of the C++ std::map of faces. */
function sortedKey(v: [number, number, number]): string {
  const s = [...v].sort((x, y) => x - y);
  return `${s[0]},${s[1]},${s[2]}`;
}

/** Keys iterated in std::map order: lexicographic on the sorted triple. */
function inKeyOrder<T>(m: Map<string, T>): Array<[string, T]> {
  return [...m.entries()].sort((a, b) => {
    const ka = a[0].split(',').map(Number);
    const kb = b[0].split(',').map(Number);
    for (let k = 0; k < 3; k++) if (ka[k] !== kb[k]) return ka[k] - kb[k];
    return 0;
  });
}

interface Tet {
  v: [number, number, number, number];
  cc: Vec3;
  crSq: number;
  valid: boolean;
}

/** Build the triangulated surface of a point cloud (ConvexHullObj::build). */
export function buildConvexHull(points: Vec3[], smooth: boolean, alpha: number): HullResult {
  const empty: HullResult = { triangles: [], normals: [], colorIdx: [] };
  if (points.length < 4) return empty;
  const npts = points.length;

  // centroid and bounding box
  const centroid: Vec3 = [0.0, 0.0, 0.0];
  const bbmin: Vec3 = [points[0][0], points[0][1], points[0][2]];
  const bbmax: Vec3 = [points[0][0], points[0][1], points[0][2]];
  for (const p of points) {
    centroid[0] += p[0];
    centroid[1] += p[1];
    centroid[2] += p[2];
    for (let d = 0; d < 3; d++) {
      bbmin[d] = Math.min(bbmin[d], p[d]);
      bbmax[d] = Math.max(bbmax[d], p[d]);
    }
  }
  centroid[0] /= npts;
  centroid[1] /= npts;
  centroid[2] /= npts;
  const maxext = Math.max(bbmax[0] - bbmin[0], bbmax[1] - bbmin[1], bbmax[2] - bbmin[2]);

  // seed points: extremes along the longest box axis
  let i0 = 0;
  let i1 = 0;
  {
    let axis = 0;
    if (bbmax[1] - bbmin[1] > bbmax[axis] - bbmin[axis]) axis = 1;
    if (bbmax[2] - bbmin[2] > bbmax[axis] - bbmin[axis]) axis = 2;
    let lo = bbmax[axis];
    let hi = bbmin[axis];
    for (let i = 0; i < npts; i++) {
      if (points[i][axis] < lo) {
        lo = points[i][axis];
        i0 = i;
      }
      if (points[i][axis] > hi) {
        hi = points[i][axis];
        i1 = i;
      }
    }
  }
  if (i0 === i1) i1 = (i0 + 1) % npts;
  {
    const d = sub(points[i1], points[i0]);
    if (dot(d, d) < SMALL * SMALL) return empty;
  }

  // point farthest from the line i0-i1
  const line = sub(points[i1], points[i0]);
  let i2 = -1;
  let maxdist = 0.0;
  for (let i = 0; i < npts; i++) {
    if (i === i0 || i === i1) continue;
    const cr = cross(line, sub(points[i], points[i0]));
    const dist = dot(cr, cr);
    if (dist > maxdist) {
      maxdist = dist;
      i2 = i;
    }
  }
  if (i2 < 0 || maxdist < SMALL * SMALL) return empty;

  // point farthest from the plane of i0, i1, i2
  let normal = cross(sub(points[i1], points[i0]), sub(points[i2], points[i0]));
  const nlen = len(normal);
  if (nlen > SMALL) normal = scale(1.0 / nlen, normal);

  let i3 = -1;
  maxdist = 0.0;
  for (let i = 0; i < npts; i++) {
    if (i === i0 || i === i1 || i === i2) continue;
    const dist = Math.abs(dot(sub(points[i], points[i0]), normal));
    if (dist > maxdist) {
      maxdist = dist;
      i3 = i;
    }
  }

  if (i3 < 0 || maxdist < SMALL) {
    return coplanarFan(points, centroid, normal, i1, i0);
  }

  // === 3D Bowyer-Watson Delaunay triangulation on a perturbed copy, with a super-tetrahedron ===
  const R = 10.0 * maxext;
  const TWO_SQRT2_OVER_3 = 0.9428090415820634;
  const SQRT6_OVER_3 = 0.8164965809277261;
  const SQRT2_OVER_3 = 0.4714045207910317;
  const ONE_THIRD = 1.0 / 3.0;
  const pts: Vec3[] = points.map((p) => [p[0], p[1], p[2]] as Vec3);
  {
    const PERT_SCALE = 1.0e-5;
    const GOLD = 0.6180339887498949;
    const SQRT2 = 1.4142135623730951;
    const SQRT3 = 1.7320508075688772;
    const mag = PERT_SCALE * Math.max(maxext, SMALL);
    for (let i = 0; i < npts; i++) {
      pts[i][0] += mag * ((((i + 1) * GOLD) % 1.0) - 0.5);
      pts[i][1] += mag * ((((i + 1) * SQRT2) % 1.0) - 0.5);
      pts[i][2] += mag * ((((i + 1) * SQRT3) % 1.0) - 0.5);
    }
  }
  pts.push([centroid[0], centroid[1], centroid[2] + R]);
  pts.push([centroid[0], centroid[1] + R * TWO_SQRT2_OVER_3, centroid[2] - R * ONE_THIRD]);
  pts.push([centroid[0] - R * SQRT6_OVER_3, centroid[1] - R * SQRT2_OVER_3, centroid[2] - R * ONE_THIRD]);
  pts.push([centroid[0] + R * SQRT6_OVER_3, centroid[1] - R * SQRT2_OVER_3, centroid[2] - R * ONE_THIRD]);
  const sv0 = npts;
  const sv1 = npts + 1;
  const sv2 = npts + 2;
  const sv3 = npts + 3;

  const tets: Tet[] = [];
  {
    const v: [number, number, number, number] = [sv0, sv1, sv2, sv3];
    const ab = sub(pts[sv1], pts[sv0]);
    const ac = sub(pts[sv2], pts[sv0]);
    const ad = sub(pts[sv3], pts[sv0]);
    if (dot(ab, cross(ac, ad)) < 0.0) [v[2], v[3]] = [v[3], v[2]];
    const cs = circumsphere(pts[v[0]], pts[v[1]], pts[v[2]], pts[v[3]]);
    tets.push({ v, cc: cs ? cs.center : [0, 0, 0], crSq: cs ? cs.radiusSq : 0, valid: true });
  }

  const INSPHERE_REL_EPS = 1.0e-7;
  // deterministic insertion order: Fisher-Yates with a 32-bit multiplicative hash (unsigned arithmetic)
  const order: number[] = [];
  for (let j = 0; j < npts; j++) order.push(j);
  for (let j = npts - 1; j > 0; j--) {
    const h = Math.imul(j, 2654435761) >>> 0;
    const k = h % (j + 1);
    [order[j], order[k]] = [order[k], order[j]];
  }

  for (let idx = 0; idx < npts; idx++) {
    const i = order[idx];
    const p = pts[i];
    const bad: number[] = [];
    for (let t = 0; t < tets.length; t++) {
      const tet = tets[t];
      if (!tet.valid) continue;
      const diff = sub(p, tet.cc);
      if (dot(diff, diff) < tet.crSq * (1.0 + INSPHERE_REL_EPS)) bad.push(t);
    }
    if (bad.length === 0) continue;

    // boundary faces of the cavity: faces that occur once among the bad tetrahedra
    const faceMap = new Map<string, { v: [number, number, number]; count: number }>();
    for (const tIdx of bad) {
      const tet = tets[tIdx];
      for (let skip = 0; skip < 4; skip++) {
        const fv: [number, number, number] = [0, 0, 0];
        let k = 0;
        for (let j = 0; j < 4; j++) if (j !== skip) fv[k++] = tet.v[j];
        const key = sortedKey(fv);
        const found = faceMap.get(key);
        if (found) found.count++;
        else faceMap.set(key, { v: [fv[0], fv[1], fv[2]], count: 1 });
      }
    }
    for (const tIdx of bad) tets[tIdx].valid = false;
    for (const [, cf] of inKeyOrder(faceMap)) {
      if (cf.count !== 1) continue;
      const v: [number, number, number, number] = [cf.v[0], cf.v[1], cf.v[2], i];
      const ab = sub(pts[v[1]], pts[v[0]]);
      const ac = sub(pts[v[2]], pts[v[0]]);
      const ad = sub(pts[v[3]], pts[v[0]]);
      if (dot(ab, cross(ac, ad)) < 0.0) [v[0], v[1]] = [v[1], v[0]];
      const cs = circumsphere(pts[v[0]], pts[v[1]], pts[v[2]], pts[v[3]]);
      if (cs) tets.push({ v, cc: cs.center, crSq: cs.radiusSq, valid: true });
      else {
        const cc = scale(0.25, add(add(pts[v[0]], pts[v[1]]), add(pts[v[2]], pts[v[3]])));
        tets.push({ v, cc, crSq: 1.0e30, valid: true });
      }
    }
  }

  // drop tetrahedra that touch the super-tetrahedron
  for (const tet of tets) {
    if (!tet.valid) continue;
    for (let k = 0; k < 4; k++) {
      if (tet.v[k] >= npts) {
        tet.valid = false;
        break;
      }
    }
  }

  // === alpha ===
  let alphaSq: number;
  if (alpha > 0.0) {
    alphaSq = alpha * alpha;
  } else {
    const avgNn = meanNearestDistance(points, bbmin, bbmax, maxext);
    const ALPHA_MULTIPLIER = 2.5;
    alphaSq = ALPHA_MULTIPLIER * ALPHA_MULTIPLIER * avgNn * avgNn;
  }

  let faces = boundaryFaces(tets, pts, alphaSq);
  if (faces.length === 0) faces = boundaryFaces(tets, pts, Infinity);

  // === triangles with normals ===
  const triangles: Triangle[] = [];
  const normals: Triangle[] = [];
  const colorIdx: Array<[number, number, number]> = [];
  if (smooth) {
    const faceNormals = faces.map((f) =>
      norm(cross(sub(points[f[1]], points[f[0]]), sub(points[f[2]], points[f[0]]))),
    );
    const vertexNormals: Vec3[] = points.map((): Vec3 => [0.0, 0.0, 0.0]);
    faces.forEach((f, fi) => {
      for (let k = 0; k < 3; k++) vertexNormals[f[k]] = add(vertexNormals[f[k]], faceNormals[fi]);
    });
    for (let i = 0; i < npts; i++) vertexNormals[i] = norm(vertexNormals[i]);
    faces.forEach((f) => {
      triangles.push([points[f[0]], points[f[1]], points[f[2]]]);
      normals.push([vertexNormals[f[0]], vertexNormals[f[1]], vertexNormals[f[2]]]);
      colorIdx.push([f[0], f[1], f[2]]);
    });
  } else {
    for (const f of faces) {
      const p0 = points[f[0]];
      const p1 = points[f[1]];
      const p2 = points[f[2]];
      const fn = norm(cross(sub(p1, p0), sub(p2, p0)));
      triangles.push([p0, p1, p2]);
      normals.push([fn, fn, fn]);
      colorIdx.push([f[0], f[1], f[2]]);
    }
  }
  return { triangles, normals, colorIdx };
}

/** The alpha-interior tetrahedra (or all of them when alphaSq is Infinity) and their outward boundary faces. */
function boundaryFaces(tets: Tet[], pts: Vec3[], alphaSq: number): Array<[number, number, number]> {
  const faceMap = new Map<string, { v: [number, number, number]; count: number }>();
  for (const tet of tets) {
    if (!tet.valid) continue;
    if (tet.crSq > alphaSq) continue;
    for (let skip = 0; skip < 4; skip++) {
      const fv: [number, number, number] = [0, 0, 0];
      let k = 0;
      for (let j = 0; j < 4; j++) if (j !== skip) fv[k++] = tet.v[j];
      const fn = cross(sub(pts[fv[1]], pts[fv[0]]), sub(pts[fv[2]], pts[fv[0]]));
      const toOpp = sub(pts[tet.v[skip]], pts[fv[0]]);
      if (dot(fn, toOpp) > 0.0) [fv[1], fv[2]] = [fv[2], fv[1]];
      const key = sortedKey(fv);
      const found = faceMap.get(key);
      if (found) found.count++;
      else faceMap.set(key, { v: [fv[0], fv[1], fv[2]], count: 1 });
    }
  }
  const faces: Array<[number, number, number]> = [];
  for (const [, af] of inKeyOrder(faceMap)) if (af.count === 1) faces.push(af.v);
  return faces;
}

/** Mean of the nearest-neighbour distances (brute force below 64 points, a 3x3x3 cell search above). */
function meanNearestDistance(points: Vec3[], bbmin: Vec3, bbmax: Vec3, maxext: number): number {
  const npts = points.length;
  let sum = 0.0;
  if (npts < 64) {
    for (let i = 0; i < npts; i++) {
      let minDsq = 1.0e30;
      for (let j = 0; j < npts; j++) {
        if (i === j) continue;
        const d = sub(points[j], points[i]);
        const dsq = dot(d, d);
        if (dsq < minDsq) minDsq = dsq;
      }
      sum += Math.sqrt(minDsq);
    }
    return sum / npts;
  }
  let vol = (bbmax[0] - bbmin[0]) * (bbmax[1] - bbmin[1]) * (bbmax[2] - bbmin[2]);
  if (vol < SMALL) vol = maxext * maxext * maxext;
  let cell = Math.cbrt(vol / npts) * 2.0;
  if (cell < SMALL) cell = maxext;
  const MAX_CELLS = 256;
  const nx = Math.min(Math.max(1, Math.trunc((bbmax[0] - bbmin[0]) / cell) + 1), MAX_CELLS);
  const ny = Math.min(Math.max(1, Math.trunc((bbmax[1] - bbmin[1]) / cell) + 1), MAX_CELLS);
  const nz = Math.min(Math.max(1, Math.trunc((bbmax[2] - bbmin[2]) / cell) + 1), MAX_CELLS);
  const cellOf = (p: Vec3): [number, number, number] => [
    Math.min(Math.trunc((p[0] - bbmin[0]) / cell), nx - 1),
    Math.min(Math.trunc((p[1] - bbmin[1]) / cell), ny - 1),
    Math.min(Math.trunc((p[2] - bbmin[2]) / cell), nz - 1),
  ];
  const grid = new Map<number, number[]>();
  const flat = (c: [number, number, number]) => c[0] * ny * nz + c[1] * nz + c[2];
  for (let i = 0; i < npts; i++) {
    const k = flat(cellOf(points[i]));
    const list = grid.get(k);
    if (list) list.push(i);
    else grid.set(k, [i]);
  }
  for (let i = 0; i < npts; i++) {
    const [cx, cy, cz] = cellOf(points[i]);
    let minDsq = 1.0e30;
    for (let dx = -1; dx <= 1; dx++) {
      const gx = cx + dx;
      if (gx < 0 || gx >= nx) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const gy = cy + dy;
        if (gy < 0 || gy >= ny) continue;
        for (let dz = -1; dz <= 1; dz++) {
          const gz = cz + dz;
          if (gz < 0 || gz >= nz) continue;
          for (const j of grid.get(gx * ny * nz + gy * nz + gz) ?? []) {
            if (i === j) continue;
            const dd = sub(points[j], points[i]);
            const dsq = dot(dd, dd);
            if (dsq < minDsq) minDsq = dsq;
          }
        }
      }
    }
    sum += Math.sqrt(minDsq);
  }
  return sum / npts;
}

/** The coplanar branch: a fan from the centroid over the points sorted by angle, facing both ways. */
function coplanarFan(points: Vec3[], centroid: Vec3, normal: Vec3, i1: number, i0: number): HullResult {
  const triangles: Triangle[] = [];
  const normals: Triangle[] = [];
  const colorIdx: Array<[number, number, number]> = [];
  const pu = norm(sub(points[i1], points[i0]));
  const pv = cross(normal, pu);
  const proj: Array<{ angle: number; idx: number }> = points.map((p, i) => {
    const d = sub(p, centroid);
    return { angle: Math.atan2(dot(d, pv), dot(d, pu)), idx: i };
  });
  proj.sort((a, b) => a.angle - b.angle);
  const unique: Array<{ angle: number; idx: number }> = [proj[0]];
  for (let i = 1; i < proj.length; i++) {
    const back = unique[unique.length - 1];
    if (Math.abs(proj[i].angle - back.angle) < 1e-10) {
      const d1 = sub(points[back.idx], centroid);
      const d2 = sub(points[proj[i].idx], centroid);
      if (dot(d2, d2) > dot(d1, d1)) unique[unique.length - 1] = proj[i];
    } else {
      unique.push(proj[i]);
    }
  }
  const np = unique.length;
  const bnorm = scale(-1.0, normal);
  for (let i = 0; i < np; i++) {
    const j = (i + 1) % np;
    const pa = points[unique[i].idx];
    const pb = points[unique[j].idx];
    triangles.push([centroid, pa, pb]);
    normals.push([normal, normal, normal]);
    colorIdx.push([unique[i].idx, unique[i].idx, unique[j].idx]);
    triangles.push([centroid, pb, pa]);
    normals.push([bnorm, bnorm, bnorm]);
    colorIdx.push([unique[j].idx, unique[j].idx, unique[i].idx]);
  }
  return { triangles, normals, colorIdx };
}
