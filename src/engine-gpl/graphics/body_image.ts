/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BODY/body_rounded_polyhedron.cpp at b332f28 (BodyRoundedPolyhedron::image: the sphere of a one-vertex body,
 * the edges as lines, the faces as rounded or flat triangles with their normals, quadrilateral faces split into two
 * triangles, and the outward shift by half the diameter when edges are drawn; rounded_radius), and the body branch of
 * DumpImage::create_image in src/GRAPHICS/dump_image.cpp (the SPHERE, LINE, TRI and TRINORM element kinds and their
 * drawing with draw_sphere, draw_cylinder (sflag 3), draw_triangle and draw_trinorm). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * flag2 selects the faces (dump_image.cpp reads it as the image() flag): 2 edges only, 3 edges and flat triangles,
 * 4 flat triangles only, anything else rounded triangles without edges. flag1 is the edge and sphere diameter, and
 * 2 * rounded radius when it is 0 or less.
 */
import type { BodyAtom, RasterFrame } from '../../engine/output/image_objects';
import { drawCylinder, drawSphere, drawTrinorm, drawTriangle } from './draw';

type V = [number, number, number];

const unit = (c: readonly number[]): V => [c[0] / 255, c[1] / 255, c[2] / 255];
const vlen = (a: V): number => Math.hypot(a[0], a[1], a[2]);
/** MathExtra::norm3 (in place in native): a zero vector is left as it is. */
const norm3 = (a: V): V => {
  const l = vlen(a);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : a;
};

/** MathExtra::matvec with the rotation matrix of quat_to_mat (row-major 3 x 3). */
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

/** The elements image() returns: a sphere, a line, a flat triangle or a triangle with per-vertex normals. */
type Element =
  | { kind: 'sphere'; c: V; d: number }
  | { kind: 'line'; p1: V; p2: V; d: number }
  | { kind: 'tri'; p: [V, V, V] }
  | { kind: 'trinorm'; p: [V, V, V]; n: [V, V, V] };

/**
 * BodyRoundedPolyhedron::image for one body (the element list in native order). vertices are the body-frame corners,
 * edges and faces index them (a face's fourth index is negative for a triangle).
 */
export function bodyElements(b: BodyAtom, flag1: number, flag2: number): Element[] {
  const p = quatMatrix(b.quat);
  const x = b.center;
  const diam = flag1 <= 0 ? 2.0 * b.radius : flag1;
  const vert = (i: number): V => [b.vertices[3 * i], b.vertices[3 * i + 1], b.vertices[3 * i + 2]];
  const out: Element[] = [];
  if (b.nsub === 1) {
    out.push({ kind: 'sphere', c: add3(matvec(p, vert(0)), x), d: diam });
    return out;
  }
  const edgeflag = flag2 === 2 || flag2 === 3;
  const triflag = flag2 !== 2;
  const flatflag = flag2 === 3 || flag2 === 4;
  const nedges = b.nsub === 2 ? 1 : b.nedge;
  if (edgeflag || nedges === 1) {
    for (let e = 0; e < nedges; e++) {
      const p1 = add3(matvec(p, vert(Math.trunc(b.edges[2 * e]))), x);
      const p2 = add3(matvec(p, vert(Math.trunc(b.edges[2 * e + 1]))), x);
      out.push({ kind: 'line', p1, p2, d: diam });
    }
  }
  if (!triflag) return out;
  for (let f = 0; f < b.nface; f++) {
    const pt = [0, 1, 2, 3].map((q) => Math.trunc(b.faces[4 * f + q]));
    const v = (i: number): V => add3(matvec(p, vert(i)), x);
    const nv = (i: number): V => matvec(p, vert(i));
    if (pt[3] >= 0) {
      // a quadrilateral: two triangles (pt1 pt2 pt3) and (pt3 pt4 pt1) sharing the normal tilted from the four corners
      const tmp = add3(matvec(p, vert(pt[3])), add3(add3(nv(pt[0]), nv(pt[1])), nv(pt[2])));
      const half = scale3(tmp, 0.5);
      const n1 = norm3(add3(nv(pt[0]), half)), n2 = norm3(add3(nv(pt[1]), half)), n3 = norm3(add3(nv(pt[2]), half));
      const a: [V, V, V] = [v(pt[0]), v(pt[1]), v(pt[2])];
      const bq: [V, V, V] = [v(pt[2]), v(pt[3]), v(pt[0])];
      const na: [V, V, V] = [n1, n2, n3];
      const nb: [V, V, V] = [norm3(add3(nv(pt[2]), half)), norm3(add3(nv(pt[3]), half)), norm3(add3(nv(pt[0]), half))];
      const tri1: Element = flatflag ? { kind: 'tri', p: a } : { kind: 'trinorm', p: a, n: na };
      const tri2: Element = flatflag ? { kind: 'tri', p: bq } : { kind: 'trinorm', p: bq, n: nb };
      if (edgeflag) {
        // the face centre of the quad (four corners), pushed out by half the diameter along its direction from the body centre
        const c = scale3([bq[0][0] + bq[1][0] + bq[2][0] + a[1][0], bq[0][1] + bq[1][1] + bq[2][1] + a[1][1], bq[0][2] + bq[1][2] + bq[2][2] + a[1][2]], 0.25);
        const shift = shiftOut(c, x, diam);
        for (const t of [a, bq]) for (let q = 0; q < 3; q++) t[q] = add3(t[q], shift);
      }
      out.push(tri1, tri2);
    } else {
      const a: [V, V, V] = [v(pt[0]), v(pt[1]), v(pt[2])];
      const tmp = scale3(add3(nv(pt[0]), add3(nv(pt[1]), nv(pt[2]))), 0.5);
      const na: [V, V, V] = [norm3(add3(nv(pt[0]), tmp)), norm3(add3(nv(pt[1]), tmp)), norm3(add3(nv(pt[2]), tmp))];
      if (edgeflag) {
        const c = scale3([a[0][0] + a[1][0] + a[2][0], a[0][1] + a[1][1] + a[2][1], a[0][2] + a[1][2] + a[2][2]], 1 / 3);
        const shift = shiftOut(c, x, diam);
        for (let q = 0; q < 3; q++) a[q] = add3(a[q], shift);
      }
      out.push(flatflag ? { kind: 'tri', p: a } : { kind: 'trinorm', p: a, n: na });
    }
  }
  return out;
}

/** The outward shift of a face: its centre relative to the body centre, scaled to half the diameter (snormalize3). */
const shiftOut = (c: V, x: V, diam: number): V => {
  const v = [c[0] - x[0], c[1] - x[1], c[2] - x[2]] as V;
  const l = vlen(v);
  const s = (0.5 * diam) / l;
  return [v[0] * s, v[1] * s, v[2] * s];
};
const add3 = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale3 = (a: V, s: number): V => [a[0] * s, a[1] * s, a[2] * s];

/** Draws the elements of the body atoms in colour b.color (dump_image.cpp, the bodyflag branch). */
export function drawBodyAtoms(f: RasterFrame, atoms: readonly BodyAtom[], flag1: number, flag2: number, opacity: number): void {
  for (const b of atoms) {
    const col = unit(b.color);
    for (const e of bodyElements(b, flag1, flag2)) {
      if (e.kind === 'sphere') drawSphere(f, e.c, col, e.d, opacity);
      else if (e.kind === 'line') drawCylinder(f, e.p1, e.p2, col, e.d, 3, opacity);
      else if (e.kind === 'tri') drawTriangle(f, e.p[0], e.p[1], e.p[2], col, opacity);
      else drawTrinorm(f, e.p[0], e.p[1], e.p[2], e.n[0], e.n[1], e.n[2], col, col, col, opacity);
    }
  }
}
