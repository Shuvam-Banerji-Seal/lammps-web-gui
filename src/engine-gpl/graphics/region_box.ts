/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/dump_image.cpp at b332f28 (DumpImage::create_image, "render regions, if supported": the POINTS
 * branch, the block and prism corner branch with its FRAME, FILLED and TRANSPARENT faces and open faces, the sphere
 * and ellipsoid branches and the plane branch), src/GRAPHICS/image_objects.cpp at b332f28 (PlaneObj: the unit square
 * of four triangles refined to its level, PlaneObj::draw, and the file-local transform that orients a template mesh
 * along a direction) and src/random_mars.h (RanMars, seed POINT_CLOUD_SEED = 19660405 of dump_image.cpp).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Scope: static regions (the core refuses moving and rotating ones) with the identity forward transform. The hull_points
 * style (ConvexHullObj) and compound regions drawn as a point cloud's hull are not ported; the core names them.
 */
import type { RasterFrame, RegionHook, BoxRegionSpec, SphereRegionSpec, EllipsoidRegionSpec, PlaneRegionSpec, PointsRegionSpec } from '../../engine/output/image_objects';
import { drawSphere, drawCylinder, drawTriangle } from './draw';
import { drawEllipsoid } from './ellipsoid_mesh';
import { drawConeRegion } from './region_cone';
import { RanMars } from '../random_mars';

type V = [number, number, number];
type Tri = [V, V, V];

const IDENTITY_QUAT = [1, 0, 0, 0];
/** dump_image.cpp POINT_CLOUD_SEED: the seed of the RanMars stream of the points and hull_points styles. */
const POINT_CLOUD_SEED = 19660405;
/** The refinement level of PlaneObj(6) (dump_image.cpp, the plane branch). */
const PLANE_LEVEL = 6;

const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V): V => {
  const l = Math.hypot(a[0], a[1], a[2]);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [a[0], a[1], a[2]];
};

/**
 * The corners of a block or prism region clamped to the box (dump_image.cpp: xlo = MAX(region xlo, boxlo[0]) and so on,
 * then the eight corners; a prism shifts the top corners by xz, yz and the tilt xy). Corner numbering is the native one.
 */
function boxCorners(spec: BoxRegionSpec, boxlo: V, boxhi: V): V[] {
  const b = spec.b;
  const xlo = Math.max(b[0], boxlo[0]), xhi = Math.min(b[1], boxhi[0]);
  const ylo = Math.max(b[2], boxlo[1]), yhi = Math.min(b[3], boxhi[1]);
  const zlo = Math.max(b[4], boxlo[2]), zhi = Math.min(b[5], boxhi[2]);
  if (spec.kind === 'block') {
    return [[xlo, ylo, zlo], [xlo, ylo, zhi], [xlo, yhi, zhi], [xlo, yhi, zlo], [xhi, ylo, zlo], [xhi, ylo, zhi], [xhi, yhi, zhi], [xhi, yhi, zlo]];
  }
  const xy = b[6], xz = b[7], yz = b[8];
  return [
    [xlo, ylo, zlo], [xlo + xz, ylo + yz, zhi], [xlo + xy + xz, yhi + yz, zhi], [xlo + xy, yhi, zlo],
    [xhi, ylo, zlo], [xhi + xz, ylo + yz, zhi], [xhi + xy + xz, yhi + yz, zhi], [xhi + xy, yhi, zlo],
  ];
}

/**
 * The six faces of the corner list, in the native order of the FILLED branch (two triangles per face). The native code
 * skips face i when open_faces[i] is set; open_faces[i] is the region's "open" number i + 1.
 */
const FACES: [number, number, number][][] = [
  [[0, 1, 2], [2, 3, 0]],
  [[4, 5, 6], [6, 7, 4]],
  [[0, 4, 7], [7, 3, 0]],
  [[1, 2, 6], [6, 5, 1]],
  [[0, 1, 5], [5, 4, 0]],
  [[3, 2, 6], [6, 7, 3]],
];
/** The twelve edges of the FRAME branch (DRAW_CYLINDER pairs in native order). */
const EDGES: [number, number][] = [[0, 1], [1, 2], [0, 3], [2, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [5, 6], [4, 7], [6, 7]];

/** Block and prism regions: a frame of cylinders, or the triangles of the faces that are not open (dump image). */
export function drawBoxRegion(f: RasterFrame, spec: BoxRegionSpec, boxlo: V, boxhi: V): void {
  const c = boxCorners(spec, boxlo, boxhi);
  const col = spec.color;
  if (spec.drawstyle === 'frame') {
    for (const [a, b] of EDGES) drawCylinder(f, c[a], c[b], col, spec.framediam, 3, 1);
    return;
  }
  // native sets opacity = reg.opacity for TRANSPARENT and 1.0 for FILLED before the faces
  const opacity = spec.drawstyle === 'transparent' ? spec.opacity : 1;
  FACES.forEach((tris, i) => {
    if (spec.openFaces.includes(i + 1)) return;
    for (const [a, b, d] of tris) drawTriangle(f, c[a], c[b], c[d], col, opacity);
  });
}

/** Sphere regions: a filled sphere of diameter 2 radius, or the wireframe of the ellipsoid mesh at level 4. */
export function drawSphereRegion(f: RasterFrame, spec: SphereRegionSpec): void {
  const col = spec.color;
  const r = spec.radius;
  if (spec.drawstyle === 'frame') {
    drawEllipsoid(f, 2, col, spec.center, [r, r, r], IDENTITY_QUAT, null, spec.framediam, 1, 4);
    return;
  }
  drawSphere(f, spec.center, col, 2 * r, spec.drawstyle === 'transparent' ? spec.opacity : 1);
}

/** Ellipsoid regions: the wireframe at mesh level 4 or the filled surface at level 5 (dump_image.cpp, the ellipsoid branch). */
export function drawEllipsoidRegion(f: RasterFrame, spec: EllipsoidRegionSpec): void {
  const col = spec.color;
  if (spec.drawstyle === 'frame') {
    drawEllipsoid(f, 2, col, spec.center, spec.shape, IDENTITY_QUAT, null, spec.framediam, 1, 4);
    return;
  }
  drawEllipsoid(f, 1, col, spec.center, spec.shape, IDENTITY_QUAT, null, spec.framediam, spec.drawstyle === 'transparent' ? spec.opacity : 1, 5);
}

/**
 * The unit square of PlaneObj: four triangles around the centre, refined to level PLANE_LEVEL (each triangle becomes four,
 * with the edge midpoints as new corners). The square lies in the plane x = 0 with corners at y, z = +-1.
 */
function planeTemplate(): Tri[] {
  const SQ1: V = [0, 1, 1], SQ2: V = [0, 1, -1], SQ3: V = [0, -1, -1], SQ4: V = [0, -1, 1], CEN: V = [0, 0, 0];
  let tris: Tri[] = [[SQ2, CEN, SQ1], [SQ3, CEN, SQ2], [SQ4, CEN, SQ3], [SQ1, CEN, SQ4]];
  for (let level = 1; level < PLANE_LEVEL; level++) {
    const next: Tri[] = [];
    for (const [t0, t1, t2] of tris) {
      const posa: V = [0.5 * (t0[0] + t2[0]), 0.5 * (t0[1] + t2[1]), 0.5 * (t0[2] + t2[2])];
      const posb: V = [0.5 * (t0[0] + t1[0]), 0.5 * (t0[1] + t1[1]), 0.5 * (t0[2] + t1[2])];
      const posc: V = [0.5 * (t1[0] + t2[0]), 0.5 * (t1[1] + t2[1]), 0.5 * (t1[2] + t2[2])];
      next.push([t0, posb, posa], [posb, t1, posc], [posa, posb, posc], [posa, posc, t2]);
    }
    tris = next;
  }
  return tris;
}

/**
 * The plane region: the template is oriented along the unit normal (PlaneObj's transform: basis u = normal, v = u x a with
 * a = x or y, w = u x v), scaled by the box diagonal and moved to the plane's point. A triangle is skipped when all of its
 * corners lie outside the box on the same side of one axis. Filled and transparent planes are triangles; frames are
 * cylinders on the triangle edges (dump_image.cpp, the plane branch).
 */
export function drawPlaneRegion(f: RasterFrame, spec: PlaneRegionSpec, boxlo: V, boxhi: V): void {
  const col = spec.color;
  const frame = spec.drawstyle === 'frame';
  const opacity = spec.drawstyle === 'transparent' ? spec.opacity : 1;
  const u = norm(spec.normal);
  const a: V = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const v = norm(cross(u, a));
  const w = cross(u, v);
  const scale = spec.scale;
  const off = spec.point;
  const place = (t: V): V => [
    scale * t[1] * v[0] + scale * t[2] * w[0] + off[0],
    scale * t[1] * v[1] + scale * t[2] * w[1] + off[1],
    scale * t[1] * v[2] + scale * t[2] * w[2] + off[2],
  ];
  for (const t of planeTemplate()) {
    const p = [place(t[0]), place(t[1]), place(t[2])];
    let outside = 0;
    for (let i = 0; i < 3; i++) {
      if (((p[0][i] < boxlo[i]) || (p[0][i] > boxhi[i])) && ((p[1][i] < boxlo[i]) || (p[1][i] > boxhi[i])) && ((p[2][i] < boxlo[i]) || (p[2][i] > boxhi[i]))) outside++;
    }
    if (outside) continue;
    if (frame) {
      drawCylinder(f, p[0], p[1], col, spec.framediam, 3, 1);
      drawCylinder(f, p[0], p[2], col, spec.framediam, 3, 1);
      drawCylinder(f, p[1], p[2], col, spec.framediam, 3, 1);
    } else {
      drawTriangle(f, p[0], p[1], p[2], col, opacity);
    }
  }
}

/**
 * The points style (dump_image.cpp, POINTS branch, for any region): npoints uniform positions in the box from RanMars
 * (three draws per point, x then y then z), each one inside the region drawn as a sphere of the point diameter.
 */
export function drawPointsRegion(f: RasterFrame, spec: PointsRegionSpec, boxlo: V, boxhi: V, match: (x: number, y: number, z: number) => boolean): void {
  const rand = new RanMars(POINT_CLOUD_SEED);
  const col = spec.color;
  const len = [boxhi[0] - boxlo[0], boxhi[1] - boxlo[1], boxhi[2] - boxlo[2]];
  for (let i = 0; i < spec.npoints; i++) {
    const pos: V = [rand.uniform() * len[0] + boxlo[0], rand.uniform() * len[1] + boxlo[1], rand.uniform() * len[2] + boxlo[2]];
    if (match(pos[0], pos[1], pos[2])) drawSphere(f, pos, col, spec.framediam, 1);
  }
}

/** The lazy HOOKS entry 'region' (registry/graphics.ts): the cone, box, sphere, ellipsoid, plane and points drawing. */
export const regionHook: RegionHook = {
  drawConeRegion,
  drawBoxRegion,
  drawSphereRegion,
  drawEllipsoidRegion,
  drawPlaneRegion,
  drawPointsRegion,
};

