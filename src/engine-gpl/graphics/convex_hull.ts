/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/GRAPHICS/image_objects.cpp, the
 * ConvexHullObj class (build, build_hull: the coplanar fan, the 3D Bowyer-Watson Delaunay triangulation with the
 * deterministic insertion order and point perturbation, the auto alpha from the mean nearest-neighbour distance, the
 * alpha-shape boundary and the smooth or flat normals) and the helpers vec3dot, vec3cross, vec3len, vec3norm and
 * compute_circumsphere. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public
 * License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The output is a list of triangles with their normals and, per vertex, the index of the input point it came from
 * (the caller maps that index to an atom). The C++ keeps the faces of its std::map in key order; the faces are sorted
 * by the same key (the sorted vertex triple) here, so the triangle order is the C++ order as well.
 *
 * The tetrahedra are flat typed arrays, and the valid ones are listed in creation order (the bad ones of each insertion
 * are dropped from the list in the same pass that finds them): the C++ scans its vector in index order and skips the
 * invalid entries, so the valid tetrahedra are visited in the same order and every result is the same. The in-sphere
 * test is written out with the same operations as vec3dot of the difference. (The arrays of small Vec3 objects and the
 * string face keys, re-split at every comparison of the key sort, made the Delaunay step of the rhodopsin
 * graphics/chunk example take about 30 s.)
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

/** A face of the C++ std::map of faces: its first-seen vertex order, its count, and the sorted triple (the map key). */
interface FaceCount {
  v0: number; v1: number; v2: number;
  s0: number; s1: number; s2: number;
  count: number;
}

/**
 * The faces seen while scanning tetrahedra, keyed by their sorted vertex triple: a number s0 N^2 + s1 N + s2 while that
 * is exact in a double (N = the number of points plus the 4 super vertices), else the string of the triple.
 */
class FaceTable {
  private readonly map = new Map<number | string, FaceCount>();
  constructor(private readonly n: number) {}
  private readonly exact = this.n * this.n * this.n < 2 ** 53;

  clear(): void { this.map.clear(); }

  add(a: number, b: number, c: number): FaceCount {
    // the sorted triple
    let s0 = a, s1 = b, s2 = c, t: number;
    if (s0 > s1) { t = s0; s0 = s1; s1 = t; }
    if (s1 > s2) { t = s1; s1 = s2; s2 = t; }
    if (s0 > s1) { t = s0; s0 = s1; s1 = t; }
    const key = this.exact ? (s0 * this.n + s1) * this.n + s2 : `${s0},${s1},${s2}`;
    const found = this.map.get(key);
    if (found) { found.count++; return found; }
    const f: FaceCount = { v0: a, v1: b, v2: c, s0, s1, s2, count: 1 };
    this.map.set(key, f);
    return f;
  }

  /** The faces seen once, in std::map order (lexicographic on the sorted triple). */
  singles(): FaceCount[] {
    const out: FaceCount[] = [];
    for (const f of this.map.values()) if (f.count === 1) out.push(f);
    out.sort((x, y) => (x.s0 - y.s0) || (x.s1 - y.s1) || (x.s2 - y.s2));
    return out;
  }
}

/** The tetrahedra: vertices, circumcentres and squared radii by index, and the valid ones in creation order. */
class TetStore {
  v = new Int32Array(0);
  cc = new Float64Array(0);
  crSq = new Float64Array(0);
  count = 0;
  active = new Int32Array(0);
  nactive = 0;
  constructor(cap: number) { this.grow(Math.max(64, cap)); }

  private grow(cap: number): void {
    const v = new Int32Array(4 * cap), cc = new Float64Array(3 * cap), crSq = new Float64Array(cap), active = new Int32Array(cap);
    v.set(this.v); cc.set(this.cc); crSq.set(this.crSq); active.set(this.active.subarray(0, this.nactive));
    this.v = v; this.cc = cc; this.crSq = crSq; this.active = active;
  }

  add(v0: number, v1: number, v2: number, v3: number, cx: number, cy: number, cz: number, r: number): void {
    if (this.count === this.crSq.length) this.grow(2 * this.count);
    const t = this.count++;
    this.v[4 * t] = v0; this.v[4 * t + 1] = v1; this.v[4 * t + 2] = v2; this.v[4 * t + 3] = v3;
    this.cc[3 * t] = cx; this.cc[3 * t + 1] = cy; this.cc[3 * t + 2] = cz;
    this.crSq[t] = r;
    this.active[this.nactive++] = t;
  }
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

  const tets = new TetStore(8 * npts);
  {
    const v: [number, number, number, number] = [sv0, sv1, sv2, sv3];
    const ab = sub(pts[sv1], pts[sv0]);
    const ac = sub(pts[sv2], pts[sv0]);
    const ad = sub(pts[sv3], pts[sv0]);
    if (dot(ab, cross(ac, ad)) < 0.0) [v[2], v[3]] = [v[3], v[2]];
    const cs = circumsphere(pts[v[0]], pts[v[1]], pts[v[2]], pts[v[3]]);
    const c: Vec3 = cs ? cs.center : [0, 0, 0];
    tets.add(v[0], v[1], v[2], v[3], c[0], c[1], c[2], cs ? cs.radiusSq : 0);
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

  const faces0 = new FaceTable(npts + 4);
  const bad: number[] = [];
  for (let idx = 0; idx < npts; idx++) {
    const i = order[idx];
    const p = pts[i];
    const px = p[0], py = p[1], pz = p[2];
    // the tetrahedra whose circumsphere holds the point (in index order), dropped from the valid list as found
    bad.length = 0;
    const act = tets.active, tcc = tets.cc, tcr = tets.crSq;
    const na = tets.nactive;
    let w = 0;
    for (let a = 0; a < na; a++) {
      const t = act[a];
      const dx = px - tcc[3 * t], dy = py - tcc[3 * t + 1], dz = pz - tcc[3 * t + 2];
      if (dx * dx + dy * dy + dz * dz < tcr[t] * (1.0 + INSPHERE_REL_EPS)) bad.push(t);
      else act[w++] = t;
    }
    tets.nactive = w;
    if (bad.length === 0) continue;

    // boundary faces of the cavity: faces that occur once among the bad tetrahedra
    faces0.clear();
    const tv = tets.v;
    for (const t of bad) {
      for (let skip = 0; skip < 4; skip++) {
        const fv: number[] = [];
        for (let j = 0; j < 4; j++) if (j !== skip) fv.push(tv[4 * t + j]);
        faces0.add(fv[0], fv[1], fv[2]);
      }
    }
    for (const cf of faces0.singles()) {
      const v: [number, number, number, number] = [cf.v0, cf.v1, cf.v2, i];
      const ab = sub(pts[v[1]], pts[v[0]]);
      const ac = sub(pts[v[2]], pts[v[0]]);
      const ad = sub(pts[v[3]], pts[v[0]]);
      if (dot(ab, cross(ac, ad)) < 0.0) [v[0], v[1]] = [v[1], v[0]];
      const cs = circumsphere(pts[v[0]], pts[v[1]], pts[v[2]], pts[v[3]]);
      if (cs) tets.add(v[0], v[1], v[2], v[3], cs.center[0], cs.center[1], cs.center[2], cs.radiusSq);
      else {
        const cc = scale(0.25, add(add(pts[v[0]], pts[v[1]]), add(pts[v[2]], pts[v[3]])));
        tets.add(v[0], v[1], v[2], v[3], cc[0], cc[1], cc[2], 1.0e30);
      }
    }
  }

  // drop tetrahedra that touch the super-tetrahedron
  {
    const act = tets.active, tv = tets.v;
    let w = 0;
    for (let a = 0; a < tets.nactive; a++) {
      const t = act[a];
      if (tv[4 * t] < npts && tv[4 * t + 1] < npts && tv[4 * t + 2] < npts && tv[4 * t + 3] < npts) act[w++] = t;
    }
    tets.nactive = w;
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
function boundaryFaces(tets: TetStore, pts: Vec3[], alphaSq: number): Array<[number, number, number]> {
  const table = new FaceTable(pts.length);
  const tv = tets.v;
  for (let a = 0; a < tets.nactive; a++) {
    const t = tets.active[a];
    if (tets.crSq[t] > alphaSq) continue;
    for (let skip = 0; skip < 4; skip++) {
      const fv: [number, number, number] = [0, 0, 0];
      let k = 0;
      for (let j = 0; j < 4; j++) if (j !== skip) fv[k++] = tv[4 * t + j];
      const fn = cross(sub(pts[fv[1]], pts[fv[0]]), sub(pts[fv[2]], pts[fv[0]]));
      const toOpp = sub(pts[tv[4 * t + skip]], pts[fv[0]]);
      if (dot(fn, toOpp) > 0.0) [fv[1], fv[2]] = [fv[2], fv[1]];
      table.add(fv[0], fv[1], fv[2]);
    }
  }
  return table.singles().map((f): [number, number, number] => [f.v0, f.v1, f.v2]);
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
