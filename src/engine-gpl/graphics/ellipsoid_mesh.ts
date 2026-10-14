/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/image_objects.cpp at b332f28: EllipsoidObj (the icosahedron of level 1, the refine step that
 * replaces each triangle by four and projects the new vertices onto the unit sphere, the rotation that puts the
 * Cartesian axes through face centres, and the two draw methods: the one with a region and the one with a quaternion
 * and an optional superellipsoid block), and the file-local radscale, superscale and supernormal helpers; and the
 * ellipsoid branch of DumpImage::create_image in src/GRAPHICS/dump_image.cpp (the per-atom colour and the choice of
 * the rounded triangle style, estyle 1, or the wireframe, estyle 2). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The triangle mesh of a level is built once and reused for every atom (native builds one per atom; the mesh is the
 * same, so only the time differs).
 */
import type { EllipsoidAtom, RasterFrame, Rgb255 } from '../../engine/output/image_objects';
import { drawSphere, drawTrinorm, drawCylinder } from './draw';

type V = [number, number, number];
type Tri = [V, V, V];

const dot = (a: V, b: V): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const add = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: V, s: number): V => [a[0] * s, a[1] * s, a[2] * s];
const vlen = (a: V): number => Math.hypot(a[0], a[1], a[2]);
/** vec3norm of image_objects.cpp: a zero vector stays zero. */
const vnorm = (a: V): V => {
  const l = vlen(a);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
};
const unit = (c: Rgb255): V => [c[0] / 255, c[1] / 255, c[2] / 255];

/** EllipsoidObj::refine: each triangle becomes four, the new vertices on the unit sphere (the picture of the native comment). */
const refine = (tris: Tri[]): Tri[] => {
  const out: Tri[] = [];
  for (const [t0, t1, t2] of tris) {
    const posa = vnorm(add(t0, t2));
    const posb = vnorm(add(t0, t1));
    const posc = vnorm(add(t1, t2));
    out.push([t0, posb, posa]);
    out.push([posb, t1, posc]);
    out.push([posa, posb, posc]);
    out.push([posa, posc, t2]);
  }
  return out;
};

/**
 * The unit-sphere mesh of a refinement level (EllipsoidObj::EllipsoidObj): the 20-triangle icosahedron refined level - 1
 * times, then rotated so that the face centres closest to +x and +y become the frame's first two axes (Gram-Schmidt).
 */
export const icosphereMesh = (level: number): Tri[] => {
  // A and B are the normalised coordinates from the golden ratio (phi = (1 + sqrt 5) / 2)
  const A = 0.5257311121191336, B = 0.8506508083520399;
  const ICO: V[] = [
    [-A, B, 0], [A, B, 0], [-A, -B, 0], [A, -B, 0],
    [0, -A, B], [0, A, B], [0, -A, -B], [0, A, -B],
    [B, 0, -A], [B, 0, A], [-B, 0, -A], [-B, 0, A],
  ];
  const i = (k: number): V => ICO[k];
  let tris: Tri[] = [
    [i(0), i(5), i(11)], [i(0), i(1), i(5)], [i(0), i(7), i(1)], [i(0), i(10), i(7)],
    [i(0), i(11), i(10)], [i(1), i(9), i(5)], [i(5), i(4), i(11)], [i(11), i(2), i(10)],
    [i(10), i(6), i(7)], [i(7), i(8), i(1)], [i(3), i(4), i(9)], [i(3), i(2), i(4)],
    [i(3), i(6), i(2)], [i(3), i(8), i(6)], [i(3), i(9), i(8)], [i(4), i(5), i(9)],
    [i(2), i(11), i(4)], [i(6), i(10), i(2)], [i(8), i(7), i(6)], [i(9), i(1), i(8)],
  ];
  for (let k = 1; k < level; k++) tris = refine(tris);
  if (tris.length === 0) return tris;
  // face centres closest to +x and +y (best starts at -1, strict comparisons as in native)
  const ax: V = [1, 0, 0], ay: V = [0, 1, 0];
  let bestDx = -1, bestDy = -1;
  let cx: V = ax, cy: V = ay;
  for (const t of tris) {
    const c = vnorm(add(add(t[0], t[1]), t[2]));
    const dx = dot(c, ax), dy = dot(c, ay);
    if (dx > bestDx) { bestDx = dx; cx = c; }
    if (dy > bestDy) { bestDy = dy; cy = c; }
  }
  const e1 = cx;
  const e2 = vnorm(sub(cy, scale(e1, dot(cy, e1))));
  const e3 = cross(e1, e2);
  const R = [e1, e2, e3];
  const rot = (v: V): V => [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
  return tris.map(([a, b, c]) => [rot(a), rot(b), rot(c)] as Tri);
};

/** radscale: the factor that moves a unit-sphere point onto the ellipsoid with half-axes shape. */
const radscale = (shape: V, p: V): number =>
  Math.sqrt(1.0 / ((p[0] / shape[0]) * (p[0] / shape[0]) + (p[1] / shape[1]) * (p[1] / shape[1]) + (p[2] / shape[2]) * (p[2] / shape[2])));

/** superscale: the factor for a superellipsoid with block exponents (n1 = block[0], n2 = block[1]). */
const superscale = (shape: V, block: [number, number], p: V): number => {
  const n = block[0], e = block[1];
  const a = Math.pow(Math.abs(p[0] / shape[0]), e) + Math.pow(Math.abs(p[1] / shape[1]), e);
  const b = Math.pow(Math.abs(p[2] / shape[2]), n);
  return Math.pow(Math.pow(a, n / e) + b, -1.0 / n);
};

/** supernormal: the unit normal of a superellipsoid at a unit-sphere point. */
const supernormal = (shape: V, block: [number, number], p: V): V => {
  const n = block[0], e = block[1];
  const xa = Math.abs(p[0] / shape[0]);
  const yb = Math.abs(p[1] / shape[1]);
  const zc = Math.abs(p[2] / shape[2]);
  const u = Math.pow(xa, e) + Math.pow(yb, e);
  const ufactor = u > 0 ? Math.pow(u, n / e - 1.0) : 0;
  const nx = xa > 0 ? (ufactor * Math.pow(xa, e - 1.0) * Math.sign(p[0] || 1)) / shape[0] : 0;
  const ny = yb > 0 ? (ufactor * Math.pow(yb, e - 1.0) * Math.sign(p[1] || 1)) / shape[1] : 0;
  const nz = zc > 0 ? (Math.pow(zc, n - 1.0) * Math.sign(p[2] || 1)) / shape[2] : 0;
  return vnorm([nx, ny, nz]);
};

/** quat_to_mat (MathExtra): the rotation matrix of a unit quaternion (w, i, j, k), row-major. */
const quatMatrix = (q: readonly number[]): number[] => {
  const [w, i, j, k] = q;
  return [
    1 - 2 * (j * j + k * k), 2 * (i * j - k * w), 2 * (i * k + j * w),
    2 * (i * j + k * w), 1 - 2 * (i * i + k * k), 2 * (j * k - i * w),
    2 * (i * k - j * w), 2 * (j * k + i * w), 1 - 2 * (i * i + j * j),
  ];
};
const matvec = (m: number[], v: V): V => [
  m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
  m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
  m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
];

const meshes = new Map<number, Tri[]>();
const meshOf = (level: number): Tri[] => {
  let m = meshes.get(level);
  if (!m) {
    m = icosphereMesh(level);
    meshes.set(level, m);
  }
  return m;
};

/**
 * EllipsoidObj::draw with a quaternion (the atom's orientation) and an optional block: flag 1 draws the rounded
 * triangles, 2 the wireframe of diameter `diameter`. A sphere of equal half-axes without a block is drawn as a sphere.
 */
export function drawEllipsoid(f: RasterFrame, flag: number, col: V, center: V, shape: V, quat: readonly number[], block: [number, number] | null, diameter: number, opacity: number, level: number): void {
  const dotri = flag === 1;
  const doframe = flag === 2 && diameter > 0;
  if (!dotri && !doframe) return;
  if (dotri && !block && shape[0] === shape[1] && shape[0] === shape[2]) {
    drawSphere(f, center, col, 2.0 * shape[0], opacity);
    return;
  }
  const mesh = meshOf(level);
  if (mesh.length === 0) return;
  const p = quatMatrix(quat);
  for (const t of mesh) {
    let n: [V, V, V] | null = null;
    if (dotri) {
      n = block
        ? [supernormal(shape, block, t[0]), supernormal(shape, block, t[1]), supernormal(shape, block, t[2])]
        : [t[0], t[1], t[2]].map((q) => vnorm([q[0] / (shape[0] * shape[0]), q[1] / (shape[1] * shape[1]), q[2] / (shape[2] * shape[2])])) as [V, V, V];
    }
    const sc = (q: V): V => (block ? scale(q, superscale(shape, block, q)) : scale(q, radscale(shape, q)));
    const e1 = add(matvec(p, sc(t[0])), center);
    const e2 = add(matvec(p, sc(t[1])), center);
    const e3 = add(matvec(p, sc(t[2])), center);
    if (dotri && n) {
      drawTrinorm(f, e1, e2, e3, matvec(p, n[0]), matvec(p, n[1]), matvec(p, n[2]), col, col, col, opacity);
    }
    if (doframe) {
      drawCylinder(f, e1, e2, col, diameter, 3, opacity);
      drawCylinder(f, e2, e3, col, diameter, 3, opacity);
      drawCylinder(f, e3, e1, col, diameter, 3, opacity);
    }
  }
}

/** The ellipsoid atoms of the "ellipsoid" keyword, in atom order: estyle 1 rounded triangles, 2 wireframe. */
export function drawEllipsoidAtoms(f: RasterFrame, atoms: readonly EllipsoidAtom[], estyle: number, level: number, diameter: number): void {
  for (const a of atoms) {
    drawEllipsoid(f, estyle, unit(a.color), a.center, a.shape, a.quat, a.block, diameter, a.opacity, level);
  }
}
